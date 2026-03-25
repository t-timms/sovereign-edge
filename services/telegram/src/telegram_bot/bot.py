"""
Sovereign Edge Telegram bot — primary user interface.

Accepts text messages, routes them through the orchestrator, and
returns responses.  Only the owner chat ID is authorised.

Commands:
  /start   — greeting
  /health  — squad health status
  /status  — current model + usage stats
  /squads  — list registered squads
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from core.config import get_settings
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
    "Use /health to check system status."
)


def _auth(func):
    """Decorator — reject requests from anyone but the owner."""
    async def wrapper(self: SovereignEdgeBot, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if update.effective_user is None or update.effective_chat is None:
            return
        if str(update.effective_chat.id) != str(self._settings.telegram_owner_chat_id):
            await update.message.reply_text("⛔ Unauthorised.")
            logger.warning(
                "unauthorised_request",
                chat_id=update.effective_chat.id,
                user=update.effective_user.username,
            )
            return
        return await func(self, update, ctx)
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

        self._app = (
            Application.builder()
            .token(token)
            .build()
        )

        self._app.add_handler(CommandHandler("start",   self._cmd_start))
        self._app.add_handler(CommandHandler("health",  self._cmd_health))
        self._app.add_handler(CommandHandler("status",  self._cmd_status))
        self._app.add_handler(CommandHandler("squads",  self._cmd_squads))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

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
        """Push a proactive message to the owner (used by morning digest)."""
        if self._app and self._settings.telegram_owner_chat_id:
            await self._app.bot.send_message(
                chat_id=self._settings.telegram_owner_chat_id,
                text=text,
                parse_mode="Markdown",
            )

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
        lines = [
            f"{'✅' if ok else '❌'} *{name}*"
            for name, ok in sorted(health.items())
        ]
        text = "🏥 *Squad Health*\n\n" + "\n".join(lines) if lines else "No squads registered."
        await update.message.reply_text(text, parse_mode="Markdown")

    @_auth
    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        registered = list(self._orchestrator._squads.keys())
        text = (
            f"⚡ *Sovereign Edge Status*\n\n"
            f"Squads: {len(registered)}\n"
            f"Registered: {', '.join(registered) or 'none'}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    @_auth
    async def _cmd_squads(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        names = sorted(self._orchestrator._squads.keys())
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

        await update.message.reply_text("⏳ Routing…")

        intent, confidence, routing = self._router.route(user_text)

        request = TaskRequest(
            content=user_text,
            intent=intent,
            priority=TaskPriority.HIGH,
            routing=routing,
        )

        try:
            result = await self._orchestrator.dispatch(request)
            reply = result.content
        except Exception:
            logger.error("dispatch_failed", exc_info=True)
            reply = "⚠️ Something went wrong. Please try again."

        # Telegram has a 4096-char limit per message
        for chunk in _split(reply, 4000):
            await update.message.reply_text(chunk, parse_mode="Markdown")

        logger.info(
            "message_handled",
            intent=intent.value,
            confidence=round(confidence, 2),
            routing=routing.value,
            chars=len(reply),
        )


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
        text = text[len(chunk):].strip()
    return chunks


# ---------------------------------------------------------------------- #
# Standalone entry point (runs bot + orchestrator together)               #
# ---------------------------------------------------------------------- #

async def _run() -> None:
    setup_logging(debug=get_settings().debug_mode)
    from career.squad import CareerSquad
    from creative.squad import CreativeSquad
    from intelligence.squad import IntelligenceSquad
    from orchestrator.main import Orchestrator
    from spiritual.squad import SpiritualSquad

    orch = Orchestrator()
    for squad in (SpiritualSquad(), CareerSquad(), IntelligenceSquad(), CreativeSquad()):
        orch.register(squad)

    bot = SovereignEdgeBot(orch)

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
