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
  07:30 — goals check-in → Telegram
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
from core.types import ExpertName, Intent, RoutingDecision, TaskRequest, TaskResult
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
        # Tracks the last handled intent per chat — used by the feedback signal loop
        # to know which skill to reinforce when the user says "thanks" or "wrong".
        self._last_intent_by_chat: dict[str, str] = {}
        # Background tasks (skill extraction) — kept alive to prevent GC before completion.
        self._bg_tasks: set[asyncio.Task[None]] = set()

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
        # Sanitize user input before any prompt injection reaches an expert or cache
        from core.security import sanitize_input

        request = request.model_copy(update={"content": sanitize_input(request.content)})

        chat_id = request.context.get("chat_id", "")

        # ── 0. Implicit feedback detection ──────────────────────────────────
        # Check if this message is a feedback signal ("thanks", "wrong", etc.)
        # before routing — update SkillLibrary scores without blocking dispatch.
        if chat_id and chat_id in self._last_intent_by_chat:
            try:
                from memory.feedback import detect_feedback_signal
                from memory.skill_library import SkillLibrary

                signal = detect_feedback_signal(request.content)
                if signal is not None:
                    last_intent = self._last_intent_by_chat[chat_id]
                    SkillLibrary().record_outcome(last_intent, success=(signal == "positive"))
                    logger.info(
                        "feedback_signal_recorded chat_id=%s intent=%s signal=%s",
                        chat_id,
                        last_intent,
                        signal,
                    )
            except Exception:
                logger.debug("feedback_signal_handling_failed", exc_info=True)

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
                logger.warning("conversation_history_inject_failed", exc_info=True)

        # ── 2. Semantic cache check ─────────────────────────────────────
        # Skip cache for INTELLIGENCE — research queries always want live data.
        # Caching paper digests for 24h defeats the expert's entire purpose.
        if request.routing == RoutingDecision.CLOUD and request.intent != Intent.INTELLIGENCE:
            try:
                from memory.semantic_cache import get_cache

                cached = await get_cache().lookup(
                    request.content, expert=self._resolve_expert(request)
                )
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
                    try:
                        from observability.audit import AuditEventType, get_audit_store

                        get_audit_store().record(
                            event_type=AuditEventType.CACHE_HIT,
                            expert=str(request.expert),
                            task_id=str(request.task_id),
                            input_text=request.content,
                            routing=str(request.routing),
                        )
                    except Exception:
                        logger.debug("audit_cache_hit_failed", exc_info=True)
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

            # Track intent so the next message can be scored as feedback
            if chat_id:
                self._last_intent_by_chat[chat_id] = request.intent.value

            # ── 3a. Reflection + skill auto-extraction ───────────────────
            # Only for CLOUD-routed, non-cached responses (latency acceptable).
            # Gated by SE_REFLECT_ENABLED so operator controls the extra LLM cost.
            if (
                self._settings.reflect_enabled
                and request.routing == RoutingDecision.CLOUD
                and not result.cached
                and result.content
            ):
                try:
                    from core.reflection import ReflectionResult, reflect_response

                    reflection: ReflectionResult | None = await reflect_response(
                        query=request.content,
                        response=result.content,
                        intent=request.intent.value,
                        expert=expert_name,
                    )
                    if reflection is not None:
                        if reflection.improved and reflection.improved_response.strip():
                            result = result.model_copy(
                                update={"content": reflection.improved_response}
                            )
                        # Auto-extract skill pattern when quality is high
                        if reflection.score >= 4:
                            task = asyncio.create_task(
                                self._extract_and_store_skill(result.content, request.intent.value)
                            )
                            self._bg_tasks.add(task)
                            task.add_done_callback(self._bg_tasks.discard)
                except Exception:
                    logger.debug("reflection_integration_failed", exc_info=True)

        # ── 4. Store result in semantic cache ───────────────────────────
        if (
            request.routing == RoutingDecision.CLOUD
            and not result.cached
            and request.intent != Intent.INTELLIGENCE
        ):
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

        # ── 7. Audit trail — granular event log for compliance/debugging ──
        try:
            from observability.audit import AuditEventType, get_audit_store

            get_audit_store().record(
                event_type=AuditEventType.DISPATCH,
                expert=str(result.expert),
                task_id=str(result.task_id),
                model=result.model_used,
                input_text=request.content,
                output_text=result.content,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                routing=str(request.routing),
                metadata={"intent": request.intent.value, "cached": str(result.cached)},
            )
        except Exception:
            logger.debug("audit_record_failed", exc_info=True)

        return result

    async def stream_dispatch(self, request: TaskRequest) -> AsyncGenerator[str, None]:
        """Like dispatch(), but streams LLM chunks as they arrive.

        Runs the full pipeline (history inject → cache check → expert stream →
        cache store → history store → episodic memory) yielding text chunks
        from the expert's stream_process() so the bot can edit messages live.
        Cache hits and local-routing results are yielded as single chunks.
        """
        # Sanitize user input before any prompt injection reaches an expert or cache
        from core.security import sanitize_input

        request = request.model_copy(update={"content": sanitize_input(request.content)})

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

                cached = await get_cache().lookup(
                    request.content, expert=self._resolve_expert(request)
                )
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

        # ── 3a. Director path — non-streaming; yields final chunk ─────────
        # Multi-expert chains (e.g. intelligence → creative) are inherently
        # multi-step and cannot be streamed. We await the full result and
        # yield it as one chunk so the Telegram placeholder still updates.
        if self._director is not None:
            result = await self._director.run(request)

            if (
                request.routing == RoutingDecision.CLOUD
                and not result.cached
                and request.intent != Intent.INTELLIGENCE
            ):
                try:
                    from memory.semantic_cache import get_cache

                    await get_cache().store(
                        request.content, result.content, expert=str(result.expert)
                    )
                except Exception:
                    logger.debug("stream_cache_store_failed", exc_info=True)

            if chat_id:
                try:
                    from memory.conversation import get_conversation_store

                    store = get_conversation_store()
                    store.add_turn(chat_id, "user", request.content)
                    store.add_turn(chat_id, "assistant", result.content, expert=str(result.expert))
                except Exception:
                    logger.debug("stream_history_store_failed", exc_info=True)

            if chat_id:
                try:
                    from memory.episodic import get_episodic_memory

                    await get_episodic_memory().add_async(
                        f"User: {request.content}\nAssistant: {result.content}",
                        user_id=chat_id,
                    )
                except Exception:
                    logger.debug("stream_episodic_add_failed", exc_info=True)

            self._trace_store.record(result)
            yield result.content
            return

        # ── 3b. Single-expert streaming path (director off / unavailable) ─
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
        try:
            async for chunk in expert.stream_process(request):
                full_content += chunk
                yield chunk
        finally:
            # ── 4-6. Post-stream bookkeeping — runs on normal completion AND exception ──
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
            Intent.GOALS: ExpertName.GOALS,
            Intent.GENERAL: ExpertName.INTELLIGENCE,  # general → intelligence
        }
        return intent_map.get(request.intent, ExpertName.INTELLIGENCE)

    # ------------------------------------------------------------------ #
    # Self-improvement helpers                                            #
    # ------------------------------------------------------------------ #

    async def _extract_and_store_skill(self, response: str, intent: str) -> None:
        """Background task: extract a reusable technique and add to SkillLibrary.

        Called as a fire-and-forget asyncio task after reflection scores >= 4.
        Failure is completely silent — never impacts the user-facing response path.
        """
        try:
            from memory.feedback import extract_skill_pattern
            from memory.skill_library import SkillLibrary

            pattern = await extract_skill_pattern(response, intent)
            if pattern:
                SkillLibrary().add_skill(intent, pattern)
        except Exception:
            logger.debug("skill_auto_extraction_failed intent=%s", intent, exc_info=True)

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
                try:
                    from observability.audit import AuditEventType, get_audit_store

                    get_audit_store().record(
                        event_type=AuditEventType.MORNING_BRIEF,
                        expert=expert_name,
                        output_text=brief,
                    )
                except Exception:
                    logger.debug("audit_morning_brief_failed", exc_info=True)
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

    async def _goals_brief(self) -> None:
        """07:30 — personal goals check-in (skipped when no active goals)."""
        await self._send_brief("goals", "🎯", "Goals Check-in")

    async def _career_rescan_brief(self) -> None:
        """18:00 — evening job scan."""
        await self._send_brief("career", "💼", "Evening Job Scan")

    async def _prune_storage(self) -> None:
        """04:00 — delete old traces and conversation turns to extend Jetson SSD life."""
        traces_deleted = self._trace_store.prune(
            max_age_days=self._settings.storage_prune_traces_days
        )
        try:
            from memory.conversation import get_conversation_store

            convos_deleted = get_conversation_store().prune_old_chats(
                max_age_days=self._settings.storage_prune_conversations_days
            )
        except Exception:
            logger.error("prune_conversations_failed", exc_info=True)
            convos_deleted = 0

        logger.info(
            "storage_pruned traces_deleted=%d conversations_deleted=%d",
            traces_deleted,
            convos_deleted,
        )

    # ------------------------------------------------------------------ #
    # Scheduler setup                                                      #
    # ------------------------------------------------------------------ #

    def _register_jobs(self) -> None:
        tz = self._settings.timezone
        wake_h = self._settings.morning_wake_hour
        wake_m = self._settings.morning_wake_minute

        def _offset(extra_minutes: int) -> tuple[int, int]:
            total = wake_h * 60 + wake_m + extra_minutes
            return total // 60 % 24, total % 60

        h0, m0 = _offset(0)
        h15, m15 = _offset(15)
        h30, m30 = _offset(30)
        h60, m60 = _offset(60)
        h120, m120 = _offset(120)
        h150, m150 = _offset(150)

        # Career rescan is absolute 18:00 (regardless of wake hour)
        # Storage prune is absolute 04:00 — runs before morning briefs
        jobs: list[tuple[Any, int, int]] = [
            (self._prune_storage, 4, 0),
            (self._morning_health_check, h0, m0),
            (self._spiritual_brief, h15, m15),
            (self._intelligence_brief, h30, m30),
            (self._career_brief, h60, m60),
            (self._creative_brief, h120, m120),
            (self._goals_brief, h150, m150),
            (self._career_rescan_brief, 18, 0),
        ]
        for fn, hour, minute in jobs:
            self._scheduler.add_job(
                fn,
                trigger=CronTrigger(hour=hour, minute=minute, timezone=tz),
                id=fn.__name__,
                replace_existing=True,
                misfire_grace_time=300,  # 5 min tolerance for delayed wake
            )
        logger.info(
            "scheduler_jobs_registered count=%d wake=%02d:%02d tz=%s",
            len(jobs),
            wake_h,
            wake_m,
            tz,
        )

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
                    "director_import_failed — install sovereign-edge-director for multi-expert chains"  # noqa: E501
                )
        logger.info("orchestrator_started director=%s", self._use_director)

    async def stop(self) -> None:
        # Cancel any in-flight skill-extraction background tasks
        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)

        self._scheduler.shutdown(wait=True)
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

        # Close SQLite connections for clean shutdown
        try:
            self._trace_store.close()
            logger.info("trace_store_closed")
        except Exception:
            logger.debug("trace_store_close_failed", exc_info=True)
        try:
            from observability.audit import get_audit_store

            get_audit_store().close()
            logger.info("audit_store_closed")
        except Exception:
            logger.debug("audit_store_close_failed", exc_info=True)
        try:
            from memory.conversation import get_conversation_store

            get_conversation_store().close()
            logger.info("conversation_store_closed")
        except Exception:
            logger.debug("conversation_store_close_failed", exc_info=True)

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
    orchestrator = Orchestrator(use_director=True)

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
