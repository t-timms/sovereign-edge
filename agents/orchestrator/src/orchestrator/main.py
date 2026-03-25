"""
Sovereign Edge orchestrator — task routing and daily schedule.

Squads register at startup; the APScheduler cron fires the morning
pipeline at CT times. Each squad gets its own timed delivery slot so
morning_brief() is called exactly once per squad per day — eliminating
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
from collections.abc import Awaitable, Callable
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from core.config import get_settings, log_startup_warnings
from core.squad import BaseSquad
from core.types import Intent, RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger, setup_logging
from observability.traces import TraceStore

logger = get_logger(__name__, component="orchestrator")


class Orchestrator:
    """Central dispatcher and morning-pipeline scheduler."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._squads: dict[str, BaseSquad] = {}
        self._scheduler = AsyncIOScheduler(timezone=self._settings.timezone)
        self._running = False
        self._send_fn: Callable[[str], Awaitable[None]] | None = None
        self._trace_store = TraceStore()

    # ------------------------------------------------------------------ #
    # Public properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def squad_names(self) -> list[str]:
        """Sorted list of registered squad names (safe public accessor)."""
        return sorted(self._squads.keys())

    @property
    def running(self) -> bool:
        """True while the orchestrator is active."""
        return self._running

    # ------------------------------------------------------------------ #
    # Squad registry                                                       #
    # ------------------------------------------------------------------ #

    def register(self, squad: BaseSquad) -> None:
        """Add a squad to the routing table."""
        self._squads[squad.name] = squad
        logger.info("squad_registered", squad=squad.name)

    def register_send_fn(self, fn: Callable[[str], Awaitable[None]]) -> None:
        """Register the Telegram send function for proactive brief delivery."""
        self._send_fn = fn
        logger.info("send_fn_registered")

    # ------------------------------------------------------------------ #
    # Task dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def dispatch(self, request: TaskRequest) -> TaskResult:
        """Route *request* to the appropriate squad and trace the result."""
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
        if request.routing == RoutingDecision.CLOUD:
            try:
                from memory.semantic_cache import get_cache

                cached = await get_cache().lookup(request.content, squad="")
                if cached is not None:
                    result = TaskResult(
                        task_id=request.task_id,
                        squad=request.squad,
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

        # ── 3. Normal squad dispatch ────────────────────────────────────
        squad_name = self._resolve_squad(request)
        squad = (
            self._squads.get(squad_name)
            or self._squads.get(SquadName.INTELLIGENCE)
            or next(iter(self._squads.values()), None)
        )

        if squad is None:
            logger.warning(
                "no_squads_registered",
                squad_name=squad_name,
                intent=request.intent.value,
            )
            return TaskResult(
                task_id=request.task_id,
                squad=SquadName.GENERAL,
                content="No squads are registered.",
                model_used="none",
                routing=RoutingDecision.LOCAL,
            )

        result = await squad.process(request)

        # ── 4. Store result in semantic cache ───────────────────────────
        if request.routing == RoutingDecision.CLOUD and not result.cached:
            try:
                from memory.semantic_cache import get_cache

                await get_cache().store(request.content, result.content, squad=str(result.squad))
            except Exception:
                logger.debug("semantic_cache_store_failed", exc_info=True)

        # ── 5. Update conversation history ──────────────────────────────
        if chat_id:
            try:
                from memory.conversation import get_conversation_store

                store = get_conversation_store()
                store.add_turn(chat_id, "user", request.content)
                store.add_turn(chat_id, "assistant", result.content, squad=str(result.squad))
            except Exception:
                logger.debug("conversation_history_store_failed", exc_info=True)

        # TraceStore.record() has internal try/except — observability failures
        # must never crash the dispatch path.
        self._trace_store.record(result)
        return result

    def _resolve_squad(self, request: TaskRequest) -> str:
        intent_map = {
            Intent.SPIRITUAL: SquadName.SPIRITUAL,
            Intent.CAREER: SquadName.CAREER,
            Intent.INTELLIGENCE: SquadName.INTELLIGENCE,
            Intent.CREATIVE: SquadName.CREATIVE,
            Intent.GENERAL: SquadName.INTELLIGENCE,  # general → intelligence
        }
        return intent_map.get(request.intent, SquadName.INTELLIGENCE)

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
        """Run all squad health checks in parallel."""
        names = list(self._squads.keys())
        squads = list(self._squads.values())

        async def _check(squad: BaseSquad) -> bool:
            try:
                return await squad.health_check()
            except Exception:
                logger.error("health_check_failed", squad=squad.name, exc_info=True)
                return False

        results = await asyncio.gather(*[_check(s) for s in squads])
        return dict(zip(names, results, strict=True))

    # ------------------------------------------------------------------ #
    # Morning pipeline helpers                                             #
    # ------------------------------------------------------------------ #

    async def _send_brief(self, squad_name: str, icon: str, header: str) -> None:
        """Call morning_brief() for one squad and push the result to Telegram.

        Each squad's pipeline step calls this exactly once — no double calls.
        """
        squad = self._squads.get(squad_name)
        if squad is None:
            logger.warning("pipeline_squad_not_found squad=%s", squad_name)
            return

        logger.info("pipeline_step step=%s", squad_name)
        try:
            async with asyncio.timeout(90):
                brief = await squad.morning_brief()
        except TimeoutError:
            logger.error("pipeline_brief_timeout step=%s", squad_name)
            brief = f"[{squad_name} brief timed out]"
        except Exception:
            logger.error("pipeline_brief_failed step=%s", squad_name, exc_info=True)
            brief = f"[{squad_name} brief unavailable]"

        if not brief:
            return

        if self._send_fn is not None:
            message = f"{icon} *{header}*\n\n{brief}"
            try:
                for chunk in _split_message(message, 4000):
                    await self._send_fn(chunk)
                logger.info("brief_sent squad=%s", squad_name)
            except Exception:
                logger.error("brief_send_failed squad=%s", squad_name, exc_info=True)
        else:
            logger.warning("brief_no_send_fn squad=%s — not delivered", squad_name)

    # ── Individual pipeline steps ──────────────────────────────────────────

    async def _morning_health_check(self) -> None:
        """05:00 — validate all squads are healthy before the morning briefs."""
        health = await self.health_check_all()
        unhealthy = [k for k, v in health.items() if not v]
        if unhealthy:
            logger.warning("morning_health_unhealthy squads=%s", unhealthy)
        else:
            logger.info("morning_health_ok squads=%d", len(health))

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
        logger.info("orchestrator_started")

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
