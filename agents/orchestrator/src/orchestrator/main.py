"""
Sovereign Edge orchestrator — task routing and daily schedule.

Experts register at startup; the APScheduler cron fires the morning
pipeline at CT times. Each expert gets its own timed delivery slot so
morning_brief() is called exactly once per expert per day — eliminating
the double-call bug where prefetch steps discarded results and the 05:30
digest called everything again.

Schedule:
  05:00 — health check + startup validation
  05:15 — spiritual brief → Telegram
  05:30 — intelligence brief → Telegram
  06:00 — career brief → Telegram
  07:00 — creative brief → Telegram
  18:00 — career rescan → Telegram

All task completions are traced to SQLite for /stats observability.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from core.config import get_settings, log_startup_warnings
from core.expert import BaseExpert
from core.types import Intent, RoutingDecision, ExpertName, TaskRequest, TaskResult
from observability.logging import get_logger, setup_logging
from observability.traces import TraceStore

logger = get_logger(__name__, component="orchestrator")


class Orchestrator:
    """Central dispatcher and morning-pipeline scheduler."""

    def __init__(self, use_director: bool = False) -> None:
        self._settings = get_settings()
        self._experts: dict[str, BaseExpert] = {}
        self._scheduler = AsyncIOScheduler(timezone=self._settings.timezone)
        self._running = False
        self._send_fn: Callable[[str], Awaitable[None]] | None = None
        self._trace_store = TraceStore()
        self._use_director = use_director
        self._director: Any = None  # DirectorGraph — loaded lazily after experts register

    # ------------------------------------------------------------------ #
    # Public properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def expert_names(self) -> list[str]:
        """Sorted list of registered expert names (safe public accessor)."""
        return sorted(self._experts.keys())

    @property
    def running(self) -> bool:
        """True while the orchestrator is active."""
        return self._running

    # ------------------------------------------------------------------ #
    # Expert registry                                                       #
    # ------------------------------------------------------------------ #

    def register(self, expert: BaseExpert) -> None:
        """Add a expert to the routing table."""
        self._experts[expert.name] = expert
        logger.info("expert_registered", expert=expert.name)

    def register_send_fn(self, fn: Callable[[str], Awaitable[None]]) -> None:
        """Register the Telegram send function for proactive brief delivery."""
        self._send_fn = fn
        logger.info("send_fn_registered")

    # ------------------------------------------------------------------ #
    # Task dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def dispatch(self, request: TaskRequest) -> TaskResult:
        """Route *request* to the appropriate expert and trace the result."""
        chat_id = request.context.get("chat_id", "")

        # ── 1. Inject conversation history as prior message turns ───────
        if chat_id:
            try:
                from memory.conversation import get_conversation_store

                history_json = get_conversation_store().get_recent_json(chat_id)
                if history_json:
                    new_ctx = dict(request.context)
                    new_ctx["history"] = history_json
                    request = request.model_copy(update={"context": new_ctx})
            except Exception:
                logger.debug("conversation_history_inject_failed", exc_info=True)

        # ── 2. Semantic cache check ─────────────────────────────────────
        # Skip cache for INTELLIGENCE — research queries always want live data.
        # Caching paper digests for 24h defeats the expert's entire purpose.
        if request.routing == RoutingDecision.CLOUD and request.intent != Intent.INTELLIGENCE:
            try:
                from memory.semantic_cache import get_cache

                cached = await get_cache().lookup(request.content, expert="")
                if cached is not None:
                    result = TaskResult(
                        task_id=request.task_id,
                        expert=request.expert,
                        content=cached["content"],
                        model_used="cache",
                        routing=RoutingDecision.CACHE,
                        cached=True,
                    )
                    self._trace_store.record(result)
                    if chat_id:
                        from memory.conversation import get_conversation_store

                        store = get_conversation_store()
                        store.add_turn(chat_id, "user", request.content)
                        store.add_turn(chat_id, "assistant", result.content)
                    return result
            except Exception:
                logger.debug("semantic_cache_lookup_failed", exc_info=True)

        # ── 3. Dispatch — director graph or single-expert ────────────────
        if self._director is not None:
            result = await self._director.run(request)
        else:
            expert_name = self._resolve_expert(request)
            expert = (
                self._experts.get(expert_name)
                or self._experts.get(ExpertName.INTELLIGENCE)
                or next(iter(self._experts.values()), None)
            )
            if expert is None:
                logger.warning(
                    "no_experts_registered",
                    expert_name=expert_name,
                    intent=request.intent.value,
                )
                return TaskResult(
                    task_id=request.task_id,
                    expert=ExpertName.GENERAL,
                    content="No experts are registered.",
                    model_used="none",
                    routing=RoutingDecision.LOCAL,
                )
            result = await expert.process(request)

        # ── 4. Store result in semantic cache ───────────────────────────
        if request.routing == RoutingDecision.CLOUD and not result.cached and request.intent != Intent.INTELLIGENCE:
            try:
                from memory.semantic_cache import get_cache

                await get_cache().store(request.content, result.content, expert=str(result.expert))
            except Exception:
                logger.debug("semantic_cache_store_failed", exc_info=True)

        # ── 5. Update conversation history ──────────────────────────────
        if chat_id:
            try:
                from memory.conversation import get_conversation_store

                store = get_conversation_store()
                store.add_turn(chat_id, "user", request.content)
                store.add_turn(chat_id, "assistant", result.content, expert=str(result.expert))
            except Exception:
                logger.debug("conversation_history_store_failed", exc_info=True)

        # ── 6. Extract long-term facts into episodic memory ─────────────
        if chat_id:
            try:
                from memory.episodic import get_episodic_memory

                text = f"User: {request.content}\nAssistant: {result.content}"
                await get_episodic_memory().add_async(text, user_id=chat_id)
            except Exception:
                logger.debug("episodic_memory_add_failed", exc_info=True)

        # TraceStore.record() has internal try/except — observability failures
        # must never crash the dispatch path.
        self._trace_store.record(result)
        return result

    async def stream_dispatch(self, request: TaskRequest) -> AsyncGenerator[str, None]:
        """Like dispatch(), but streams LLM chunks as they arrive.

        Runs the full pipeline (history inject → cache check → expert stream →
        cache store → history store → episodic memory) yielding text chunks
        from the expert's stream_process() so the bot can edit messages live.
        Cache hits and local-routing results are yielded as single chunks.
        """
        chat_id = request.context.get("chat_id", "")

        # ── 1. Inject conversation history ──────────────────────────────
        if chat_id:
            try:
                from memory.conversation import get_conversation_store

                history_json = get_conversation_store().get_recent_json(chat_id)
                if history_json:
                    new_ctx = dict(request.context)
                    new_ctx["history"] = history_json
                    request = request.model_copy(update={"context": new_ctx})
            except Exception:
                logger.debug("stream_history_inject_failed", exc_info=True)

        # ── 2. Semantic cache check — yield immediately on hit ───────────
        if request.routing == RoutingDecision.CLOUD and request.intent != Intent.INTELLIGENCE:
            try:
                from memory.semantic_cache import get_cache

                cached = await get_cache().lookup(request.content, expert="")
                if cached is not None:
                    result = TaskResult(
                        task_id=request.task_id,
                        expert=request.expert,
                        content=cached["content"],
                        model_used="cache",
                        routing=RoutingDecision.CACHE,
                        cached=True,
                    )
                    self._trace_store.record(result)
                    yield cached["content"]
                    return
            except Exception:
                logger.debug("stream_cache_lookup_failed", exc_info=True)

        # ── 3. Resolve expert and stream ──────────────────────────────────
        expert_name = self._resolve_expert(request)
        expert = (
            self._experts.get(expert_name)
            or self._experts.get(ExpertName.INTELLIGENCE)
            or next(iter(self._experts.values()), None)
        )
        if expert is None:
            yield "No experts are registered."
            return

        full_content = ""
        async for chunk in expert.stream_process(request):
            full_content += chunk
            yield chunk

        # ── 4–6. Post-stream bookkeeping (same as dispatch) ─────────────
        if request.routing == RoutingDecision.CLOUD and request.intent != Intent.INTELLIGENCE:
            try:
                from memory.semantic_cache import get_cache

                await get_cache().store(request.content, full_content, expert=expert_name)
            except Exception:
                logger.debug("stream_cache_store_failed", exc_info=True)

        if chat_id:
            try:
                from memory.conversation import get_conversation_store

                store = get_conversation_store()
                store.add_turn(chat_id, "user", request.content)
                store.add_turn(chat_id, "assistant", full_content, expert=expert_name)
            except Exception:
                logger.debug("stream_history_store_failed", exc_info=True)

        if chat_id:
            try:
                from memory.episodic import get_episodic_memory

                await get_episodic_memory().add_async(
                    f"User: {request.content}\nAssistant: {full_content}",
                    user_id=chat_id,
                )
            except Exception:
                logger.debug("stream_episodic_add_failed", exc_info=True)

        result = TaskResult(
            task_id=request.task_id,
            expert=ExpertName(expert_name),
            content=full_content,
            model_used="stream",
            routing=request.routing,
        )
        self._trace_store.record(result)

    def _resolve_expert(self, request: TaskRequest) -> str:
        intent_map = {
            Intent.SPIRITUAL: ExpertName.SPIRITUAL,
            Intent.CAREER: ExpertName.CAREER,
            Intent.INTELLIGENCE: ExpertName.INTELLIGENCE,
            Intent.CREATIVE: ExpertName.CREATIVE,
            Intent.GENERAL: ExpertName.INTELLIGENCE,  # general → intelligence
        }
        return intent_map.get(request.intent, ExpertName.INTELLIGENCE)

    # ------------------------------------------------------------------ #
    # Observability                                                        #
    # ------------------------------------------------------------------ #

    def get_daily_stats(self) -> dict:
        """Today's aggregated trace stats — used by the /stats Telegram command."""
        return self._trace_store.get_daily_stats()

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #

    async def health_check_all(self) -> dict[str, bool]:
        """Run all expert health checks in parallel."""
        names = list(self._experts.keys())
        experts = list(self._experts.values())

        async def _check(expert: BaseExpert) -> bool:
            try:
                return await expert.health_check()
            except Exception:
                logger.error("health_check_failed", expert=expert.name, exc_info=True)
                return False

        results = await asyncio.gather(*[_check(s) for s in experts])
        return dict(zip(names, results, strict=True))

    # ------------------------------------------------------------------ #
    # Morning pipeline helpers                                             #
    # ------------------------------------------------------------------ #

    async def _send_brief(self, expert_name: str, icon: str, header: str) -> None:
        """Call morning_brief() for one expert and push the result to Telegram.

        Each expert's pipeline step calls this exactly once — no double calls.
        """
        expert = self._experts.get(expert_name)
        if expert is None:
            logger.warning("pipeline_expert_not_found expert=%s", expert_name)
            return

        logger.info("pipeline_step step=%s", expert_name)
        try:
            async with asyncio.timeout(90):
                brief = await expert.morning_brief()
        except TimeoutError:
            logger.error("pipeline_brief_timeout step=%s", expert_name)
            brief = f"[{expert_name} brief timed out]"
        except Exception:
            logger.error("pipeline_brief_failed step=%s", expert_name, exc_info=True)
            brief = f"[{expert_name} brief unavailable]"

        if not brief:
            return

        if self._send_fn is not None:
            message = f"{icon} *{header}*\n\n{brief}"
            try:
                for chunk in _split_message(message, 4000):
                    await self._send_fn(chunk)
                logger.info("brief_sent expert=%s", expert_name)
            except Exception:
                logger.error("brief_send_failed expert=%s", expert_name, exc_info=True)
        else:
            logger.warning("brief_no_send_fn expert=%s — not delivered", expert_name)

    # ── Individual pipeline steps ──────────────────────────────────────────

    async def _morning_health_check(self) -> None:
        """05:00 — validate all experts are healthy before the morning briefs."""
        health = await self.health_check_all()
        unhealthy = [k for k, v in health.items() if not v]
        if unhealthy:
            logger.warning("morning_health_unhealthy experts=%s", unhealthy)
        else:
            logger.info("morning_health_ok experts=%d", len(health))

    async def _spiritual_brief(self) -> None:
        """05:15 — morning devotional."""
        await self._send_brief("spiritual", "✝️", "Morning Devotional")

    async def _intelligence_brief(self) -> None:
        """05:30 — AI/ML papers and news."""
        await self._send_brief("intelligence", "🧠", "AI/ML Intelligence")

    async def _career_brief(self) -> None:
        """06:00 — morning job listings."""
        await self._send_brief("career", "💼", "Career & Jobs")

    async def _creative_brief(self) -> None:
        """07:00 — content strategy direction."""
        await self._send_brief("creative", "✍️", "Creative Direction")

    async def _career_rescan_brief(self) -> None:
        """18:00 — evening job scan."""
        await self._send_brief("career", "💼", "Evening Job Scan")

    # ------------------------------------------------------------------ #
    # Scheduler setup                                                      #
    # ------------------------------------------------------------------ #

    def _register_jobs(self) -> None:
        tz = self._settings.timezone
        jobs: list[tuple[Any, str, str]] = [
            (self._morning_health_check, "05", "00"),
            (self._spiritual_brief, "05", "15"),
            (self._intelligence_brief, "05", "30"),
            (self._career_brief, "06", "00"),
            (self._creative_brief, "07", "00"),
            (self._career_rescan_brief, "18", "00"),
        ]
        for fn, hour, minute in jobs:
            self._scheduler.add_job(
                fn,
                trigger=CronTrigger(hour=int(hour), minute=int(minute), timezone=tz),
                id=fn.__name__,
                replace_existing=True,
                misfire_grace_time=300,  # 5 min tolerance for delayed wake
            )
        logger.info("scheduler_jobs_registered count=%d", len(jobs))

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        log_startup_warnings()
        self._register_jobs()
        self._scheduler.start()
        self._running = True

        # Initialise director after experts are registered (lazy — requires langgraph)
        if self._use_director and self._experts:
            try:
                from director.graph import DirectorGraph

                self._director = DirectorGraph(experts=self._experts)
                logger.info("director_initialized experts=%s", list(self._experts.keys()))
            except ImportError:
                logger.warning(
                    "director_import_failed — install sovereign-edge-director for multi-expert chains"
                )
        logger.info("orchestrator_started director=%s", self._use_director)

    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        try:
            from search.arxiv import aclose as arxiv_close
            from search.bible import aclose as bible_close
            from search.hf import aclose as hf_close
            from search.jina import aclose as jina_close

            await asyncio.gather(
                jina_close(),
                arxiv_close(),
                bible_close(),
                hf_close(),
                return_exceptions=True,
            )
            logger.info("http_clients_closed")
        except ImportError:
            logger.debug("search_modules_not_available — skip aclose")
        self._running = False
        logger.info("orchestrator_stopped")


def _split_message(text: str, size: int) -> list[str]:
    """Split long text into chunks without breaking mid-word."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    while text:
        chunk = text[:size]
        split_at = chunk.rfind("\n") if "\n" in chunk else chunk.rfind(" ")
        if split_at > 0 and len(text) > size:
            chunk = text[:split_at]
        chunks.append(chunk.strip())
        text = text[len(chunk) :].strip()
    return chunks


# ---------------------------------------------------------------------- #
# Entry point                                                             #
# ---------------------------------------------------------------------- #


async def run() -> None:
    setup_logging(debug=get_settings().debug_mode)
    orchestrator = Orchestrator()

    loop = asyncio.get_running_loop()
    # Keep a reference to the shutdown task so it isn't garbage-collected mid-run
    _shutdown_tasks: set[asyncio.Task[None]] = set()

    def _handle_signal() -> None:
        logger.info("shutdown_signal_received")
        task = loop.create_task(orchestrator.stop())
        _shutdown_tasks.add(task)
        task.add_done_callback(_shutdown_tasks.discard)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await orchestrator.start()

    while orchestrator.running:
        await asyncio.sleep(1)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
