"""
Sovereign Edge Telegram bot — primary user interface.

Accepts text messages, routes them through the orchestrator, and
returns responses. Only the owner chat ID is authorised.

Commands:
  /start   — greeting
  /health  — squad health status
  /stats   — today's usage and cost stats (TraceStore)
  /status  — current squad + routing info
  /squads  — list registered squads
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from core.config import get_settings, log_startup_warnings
from core.types import TaskPriority, TaskRequest
from observability.logging import get_logger, setup_logging
from router.classifier import IntentRouter
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from telegram import Update

if TYPE_CHECKING:
    from orchestrator.main import Orchestrator

logger = get_logger(__name__, component="telegram")

_WELCOME = (
    "👁️ *Sovereign Edge online.*\n\n"
    "Send any message and I'll route it to the right squad.\n"
    "Use /health to check system status.\n"
    "Use /stats to see today's usage."
)

# Maximum characters accepted from user input — prevents context-window flooding
_MAX_INPUT_CHARS = 2000

# Per-chat rate limiting: max 1 request per N seconds
_RATE_LIMIT_SECONDS = 2.0
# chat_id → monotonic timestamp of last allowed request
_last_request: dict[str, float] = {}


def _auth(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator — reject requests from anyone but the owner."""

    @functools.wraps(func)
    async def wrapper(
        self: SovereignEdgeBot, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.effective_user is None or update.effective_chat is None:
            return
        if str(update.effective_chat.id) != str(self._settings.telegram_owner_chat_id):
            # Guard: message may be None for callback queries / channel posts
            if update.message is not None:
                await update.message.reply_text("⛔ Unauthorised.")
            logger.critical(
                "unauthorised_access_attempt",
                chat_id=update.effective_chat.id,
                user=getattr(update.effective_user, "username", "unknown"),
            )
            return
        await func(self, update, ctx)

    return wrapper


class SovereignEdgeBot:
    """Telegram bot wired to the Sovereign Edge orchestrator."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._settings = get_settings()
        self._orchestrator = orchestrator
        self._router = IntentRouter()
        self._app: Application | None = None

    # ------------------------------------------------------------------ #
    # Bot lifecycle                                                        #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        token = self._settings.telegram_bot_token
        if not token:
            logger.warning("telegram_token_missing — bot disabled")
            return

        from telegram.request import HTTPXRequest

        # Short connect timeout so a hung TLS handshake never blocks reply delivery
        _req = HTTPXRequest(connect_timeout=8.0, read_timeout=60.0, write_timeout=30.0)
        self._app = Application.builder().token(token).request(_req).build()

        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("health", self._cmd_health))
        self._app.add_handler(CommandHandler("stats", self._cmd_stats))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("squads", self._cmd_squads))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("telegram_bot_started")

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("telegram_bot_stopped")

    async def send_message(self, text: str) -> None:
        """Push a proactive message to the owner (used by morning briefs)."""
        if not (self._app and self._settings.telegram_owner_chat_id):
            return
        try:
            await self._app.bot.send_message(
                chat_id=self._settings.telegram_owner_chat_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception:
            logger.warning("telegram_markdown_failed — retrying as plain text")
            try:
                await self._app.bot.send_message(
                    chat_id=self._settings.telegram_owner_chat_id,
                    text=text,
                )
            except Exception:
                logger.error("telegram_send_failed", exc_info=True)

    # ------------------------------------------------------------------ #
    # Commands                                                             #
    # ------------------------------------------------------------------ #

    @_auth
    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(_WELCOME, parse_mode="Markdown")

    @_auth
    async def _cmd_health(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("🔍 Checking squads…")
        health = await self._orchestrator.health_check_all()
        lines = [f"{'✅' if ok else '❌'} *{name}*" for name, ok in sorted(health.items())]
        text = "🏥 *Squad Health*\n\n" + "\n".join(lines) if lines else "No squads registered."
        await update.message.reply_text(text, parse_mode="Markdown")

    @_auth
    async def _cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Today's LLM usage stats from TraceStore."""
        stats = self._orchestrator.get_daily_stats()
        if not stats:
            await update.message.reply_text("No stats recorded today yet.")
            return

        avg_ms = stats.get("avg_latency_ms", 0.0)
        text = (
            "📊 *Today's Stats*\n\n"
            f"Requests: {stats.get('total_requests', 0)}\n"
            f"Cache hits: {stats.get('cache_hits', 0)}\n"
            f"Errors: {stats.get('errors', 0)}\n"
            f"Avg latency: {avg_ms:.0f}ms\n"
            f"Tokens in: {stats.get('total_tokens_in', 0):,}\n"
            f"Tokens out: {stats.get('total_tokens_out', 0):,}\n"
            f"Total cost: ${stats.get('total_cost_usd', 0.0):.4f}\n"
            f"Models: {stats.get('models_used') or 'none'}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    @_auth
    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        names = self._orchestrator.squad_names  # public property — no private access
        text = (
            "⚡ *Sovereign Edge Status*\n\n"
            f"Squads: {len(names)}\n"
            f"Registered: {', '.join(names) or 'none'}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    @_auth
    async def _cmd_squads(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        names = self._orchestrator.squad_names  # public property — no private access
        text = "🤖 *Registered Squads*\n\n" + "\n".join(f"• {n}" for n in names)
        await update.message.reply_text(text, parse_mode="Markdown")

    # ------------------------------------------------------------------ #
    # Message handler                                                      #
    # ------------------------------------------------------------------ #

    @_auth
    async def _on_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        chat_id = str(update.effective_chat.id)
        tg_chat_id = update.effective_chat.id

        # Per-chat rate limit — reject if too soon after last request
        now = time.monotonic()
        last = _last_request.get(chat_id, 0.0)
        if now - last < _RATE_LIMIT_SECONDS:
            await update.message.reply_text("⏳ Slow down — one message at a time.")
            return
        _last_request[chat_id] = now

        # Input length guard — prevent context-window flooding
        if len(user_text) > _MAX_INPUT_CHARS:
            logger.info("input_truncated chat_id=%s original_len=%d", chat_id, len(user_text))
            user_text = user_text[:_MAX_INPUT_CHARS]

        intent, confidence, routing = await self._router.aroute(user_text)

        request = TaskRequest(
            content=user_text,
            intent=intent,
            priority=TaskPriority.HIGH,
            routing=routing,
            context={"chat_id": chat_id},
        )

        # Send typing immediately — guaranteed visible even on cache hits
        await ctx.bot.send_chat_action(chat_id=tg_chat_id, action="typing")

        # Background refresher keeps the indicator alive every 4 s for slow LLM calls
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(_keep_typing(ctx.bot, tg_chat_id, stop_typing))
        try:
            result = await self._orchestrator.dispatch(request)
            reply = result.content
        except Exception:
            logger.error("dispatch_failed", exc_info=True)
            reply = "⚠️ Something went wrong. Please try again."
        finally:
            stop_typing.set()
            typing_task.cancel()
            try:
                await asyncio.wait_for(typing_task, timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
                pass

        # Telegram has a 4096-char limit per message
        for chunk in _split(_sanitize_markdown(reply), 4000):
            try:
                await update.message.reply_text(chunk, parse_mode="Markdown")
            except Exception:
                try:
                    await update.message.reply_text(chunk)
                except Exception:
                    logger.error("reply_text_failed", exc_info=True)

        logger.info(
            "message_handled",
            intent=intent.value,
            confidence=round(confidence, 2),
            routing=routing.value,
            chars=len(reply),
        )


async def _keep_typing(bot: Any, chat_id: int, stop: asyncio.Event) -> None:  # noqa: ANN401
    """Refresh the typing indicator every 4 s until *stop* is set."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            logger.debug("typing_action_failed", exc_info=True)
        await asyncio.sleep(4)


def _sanitize_markdown(text: str) -> str:
    """Coerce LLM output to the Telegram Markdown subset.

    Telegram (MarkdownV1) only renders *bold*, _italic_, and [text](url).
    Models default to standard Markdown (**bold**, ## headers, ---) which
    renders as literal characters. Fix it here so prompt instructions alone
    don't need to be perfect.
    """
    import re

    # **bold** → *bold*  (non-greedy, single-line)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # ## Any header level → plain text (strip leading #+ and space)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # --- dividers → remove entirely
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.MULTILINE)
    # Strip trailing boilerplate summary sentences the model likes to append
    text = re.sub(
        r"\n*These (?:papers|results|findings|advancements|research).{0,300}$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Strip opening preamble lines ("Here are...", "The following...", etc.)
    text = re.sub(
        r"^(?:Here are|The following are|Below are|The latest)[^\n]*\n+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Strip mid-list transition sentences ("Additionally, X has published...")
    text = re.sub(
        r"\n+Additionally,[^\n]*\n+",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    # Convert bare URLs to [url](url) — bare URLs don't embed as links in
    # Telegram MarkdownV1. Skip URLs already inside []() link syntax.
    text = re.sub(r"(?<!\[)(?<!\()(https?://[^\s\)\]]+)", r"[\1](\1)", text)
    # Bold paper titles — lines with format "Title: description"
    # where the title isn't already wrapped in * or _
    text = re.sub(
        r"^(?![*_\[•\-])([^:\n]{10,80}):([ \t])",
        r"*\1*:\2",
        text,
        flags=re.MULTILINE,
    )
    return text.strip()


def _split(text: str, size: int) -> list[str]:
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
# Standalone entry point (runs bot + orchestrator together)               #
# ---------------------------------------------------------------------- #


async def _run() -> None:
    setup_logging(debug=get_settings().debug_mode)
    log_startup_warnings()

    from career.squad import CareerSquad
    from creative.squad import CreativeSquad
    from intelligence.squad import IntelligenceSquad
    from orchestrator.main import Orchestrator
    from spiritual.squad import SpiritualSquad

    orch = Orchestrator()
    for squad in (SpiritualSquad(), CareerSquad(), IntelligenceSquad(), CreativeSquad()):
        orch.register(squad)

    bot = SovereignEdgeBot(orch)

    # Wire morning brief delivery so the scheduler can push to Telegram
    orch.register_send_fn(bot.send_message)

    await asyncio.gather(orch.start(), bot.start())

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        await bot.stop()
        await orch.stop()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
