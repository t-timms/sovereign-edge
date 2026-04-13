"""
Sovereign Edge Discord bot — primary user interface.

Accepts text messages, routes them through the orchestrator, and
returns responses. Only the owner Discord user ID is authorised.

Commands (slash commands):
  /start   — greeting
  /health  — expert health status
  /stats   — today's usage and cost stats (TraceStore)
  /status  — current expert + routing info
  /experts  — list registered experts

# Requires in Settings:
#   discord_bot_token: str = ""
#   discord_owner_user_id: str = ""
# Add to core/config.py Settings class with env prefix SE_:
#   SE_DISCORD_BOT_TOKEN
#   SE_DISCORD_OWNER_USER_ID
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from core.config import get_settings, log_startup_warnings
from core.types import TaskPriority, TaskRequest
from observability.logging import get_logger, setup_logging
from router.classifier import LOW_CONFIDENCE_THRESHOLD, IntentRouter

import discord
from discord import app_commands

if TYPE_CHECKING:
    from orchestrator.main import Orchestrator

logger = get_logger(__name__, component="discord")

_WELCOME = (
    "👁️ **Sovereign Edge online.**\n\n"
    "Send any message and I'll route it to the right expert.\n"
    "Use /health to check system status.\n"
    "Use /stats to see today's usage."
)

# Maximum characters accepted from user input — prevents context-window flooding
_MAX_INPUT_CHARS = 2000

# Discord hard limit per message send
_DISCORD_MESSAGE_LIMIT = 2000

# Per-channel rate limiting: max 1 request per N seconds
_RATE_LIMIT_SECONDS = 2.0
# channel_id → monotonic timestamp of last allowed request
_last_request: dict[str, float] = {}


class SovereignEdgeDiscordBot(discord.Client):
    """Discord bot wired to the Sovereign Edge orchestrator."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self._settings = get_settings()
        self._orchestrator = orchestrator
        self._router = IntentRouter()
        self.tree = app_commands.CommandTree(self)

        # Register slash commands on the tree
        self._register_commands()

    # ------------------------------------------------------------------ #
    # Auth helper                                                          #
    # ------------------------------------------------------------------ #

    def _is_owner(self, user: discord.User | discord.Member) -> bool:
        """Return True only for the configured owner user ID."""
        try:
            return int(user.id) == int(self._settings.discord_owner_user_id)
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------ #
    # Bot lifecycle                                                        #
    # ------------------------------------------------------------------ #

    async def start(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        token = self._settings.discord_bot_token.get_secret_value()
        if not token:
            logger.warning("discord_token_missing — bot disabled")
            return
        logger.info("discord_bot_starting")
        await super().start(token, *args, **kwargs)

    async def stop(self) -> None:
        await self.close()
        logger.info("discord_bot_stopped")

    async def setup_hook(self) -> None:
        """Called by discord.py after login but before connecting to gateway.

        Syncs the slash-command tree with Discord's API.
        """
        await self.tree.sync()
        logger.info("discord_slash_commands_synced")

    async def on_ready(self) -> None:
        logger.info("discord_bot_ready user=%s", self.user)

    # ------------------------------------------------------------------ #
    # Proactive send (morning briefs)                                      #
    # ------------------------------------------------------------------ #

    async def send_message(self, text: str) -> None:
        """Push a proactive message to the owner (used by morning briefs).

        Mirrors the Telegram send_message() signature exactly so the
        orchestrator's register_send_fn() works with either bot.
        """
        owner_id = self._settings.discord_owner_user_id
        if not owner_id:
            return
        try:
            user = await self.fetch_user(int(owner_id))
        except (discord.NotFound, discord.HTTPException, ValueError):
            logger.error("discord_send_message_fetch_user_failed", exc_info=True)
            return

        for chunk in _split(text, _DISCORD_MESSAGE_LIMIT):
            try:
                await user.send(chunk)
            except discord.Forbidden:
                logger.error(
                    "discord_send_message_forbidden — owner DMs may be closed",
                    exc_info=True,
                )
                return
            except discord.HTTPException:
                logger.error("discord_send_message_failed", exc_info=True)
                return

    # ------------------------------------------------------------------ #
    # Message event handler                                                #
    # ------------------------------------------------------------------ #

    async def on_message(self, message: discord.Message) -> None:
        # Ignore bot's own messages
        if message.author.bot:
            return

        # Ignore DM messages — require guild context for channel-based rate limiting
        # (DM support can be added later; scope to guild messages for now)
        if message.guild is None:
            return

        # Auth check
        if not self._is_owner(message.author):
            await message.reply("⛔ Unauthorised.")
            logger.critical(
                "unauthorised_access_attempt",
                user_id=message.author.id,
                user=str(message.author),
                channel=message.channel.id,
            )
            return

        user_text = (message.content or "").strip()
        if not user_text:
            return

        channel_id = str(message.channel.id)

        # Per-channel rate limit
        now = time.monotonic()
        last = _last_request.get(channel_id, 0.0)
        if now - last < _RATE_LIMIT_SECONDS:
            await message.reply("⏳ Slow down — one message at a time.")
            return
        _last_request[channel_id] = now

        # Input length guard
        if len(user_text) > _MAX_INPUT_CHARS:
            logger.info("input_truncated channel_id=%s original_len=%d", channel_id, len(user_text))
            user_text = user_text[:_MAX_INPUT_CHARS]

        intent, confidence, routing = await self._router.aroute(user_text)

        # Low-confidence general query — nudge the user toward a specific expert.
        # Still dispatches so they get an answer; the hint improves future routing.
        if intent.value == "GENERAL" and confidence <= LOW_CONFIDENCE_THRESHOLD:
            await message.reply(
                "🤔 *Not sure which expert fits this — routing to intelligence.*\n"
                "For better results try: Bible/faith · job search · AI research · content creation"
            )

        request = TaskRequest(
            content=user_text,
            intent=intent,
            priority=TaskPriority.HIGH,
            routing=routing,
            context={"chat_id": channel_id},
        )

        # Send typing indicator immediately — visible even on cache hits
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(_keep_typing(message.channel, stop_typing))
        try:
            result = await self._orchestrator.dispatch(request)
            reply = result.content
        except Exception:
            logger.error("dispatch_failed", exc_info=True)
            reply = "⚠️ Something went wrong. Please try again."
        finally:
            stop_typing.set()
            if not typing_task.done():
                typing_task.cancel()
            try:
                await asyncio.wait_for(typing_task, timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
                pass

        # Discord has a 2000-char limit per message; no markdown sanitizer needed —
        # Discord renders standard Markdown natively (** bold, ## headers all work).
        first = True
        for chunk in _split(reply, _DISCORD_MESSAGE_LIMIT):
            try:
                if first:
                    await message.reply(chunk)
                    first = False
                else:
                    await message.channel.send(chunk)
            except discord.HTTPException:
                logger.error("discord_reply_failed", exc_info=True)

        logger.info(
            "message_handled",
            intent=intent.value,
            confidence=round(confidence, 2),
            routing=routing.value,
            chars=len(reply),
        )

    # ------------------------------------------------------------------ #
    # Slash commands                                                       #
    # ------------------------------------------------------------------ #

    def _register_commands(self) -> None:
        """Attach all slash command callbacks to self.tree."""

        @self.tree.command(name="start", description="Sovereign Edge greeting")
        async def cmd_start(interaction: discord.Interaction) -> None:
            if not self._is_owner(interaction.user):
                await interaction.response.send_message("⛔ Unauthorised.", ephemeral=True)
                return
            await interaction.response.send_message(_WELCOME)

        @self.tree.command(name="health", description="Check expert health status")
        async def cmd_health(interaction: discord.Interaction) -> None:
            if not self._is_owner(interaction.user):
                await interaction.response.send_message("⛔ Unauthorised.", ephemeral=True)
                return
            await interaction.response.defer()
            health = await self._orchestrator.health_check_all()
            lines = [f"{'✅' if ok else '❌'} **{name}**" for name, ok in sorted(health.items())]
            text = (
                "🏥 **Expert Health**\n\n" + "\n".join(lines) if lines else "No experts registered."
            )
            await interaction.followup.send(text)

        @self.tree.command(name="stats", description="Today's usage and cost stats")
        async def cmd_stats(interaction: discord.Interaction) -> None:
            if not self._is_owner(interaction.user):
                await interaction.response.send_message("⛔ Unauthorised.", ephemeral=True)
                return
            stats = self._orchestrator.get_daily_stats()
            if not stats:
                await interaction.response.send_message("No stats recorded today yet.")
                return
            avg_ms = stats.get("avg_latency_ms", 0.0)
            text = (
                "📊 **Today's Stats**\n\n"
                f"Requests: {stats.get('total_requests', 0)}\n"
                f"Cache hits: {stats.get('cache_hits', 0)}\n"
                f"Errors: {stats.get('errors', 0)}\n"
                f"Avg latency: {avg_ms:.0f}ms\n"
                f"Tokens in: {stats.get('total_tokens_in', 0):,}\n"
                f"Tokens out: {stats.get('total_tokens_out', 0):,}\n"
                f"Total cost: ${stats.get('total_cost_usd', 0.0):.4f}\n"
                f"Models: {stats.get('models_used') or 'none'}"
            )
            await interaction.response.send_message(text)

        @self.tree.command(name="status", description="Current expert and routing info")
        async def cmd_status(interaction: discord.Interaction) -> None:
            if not self._is_owner(interaction.user):
                await interaction.response.send_message("⛔ Unauthorised.", ephemeral=True)
                return
            names = self._orchestrator.expert_names
            text = (
                "⚡ **Sovereign Edge Status**\n\n"
                f"Experts: {len(names)}\n"
                f"Registered: {', '.join(names) or 'none'}"
            )
            await interaction.response.send_message(text)

        @self.tree.command(name="experts", description="List registered experts")
        async def cmd_experts(interaction: discord.Interaction) -> None:
            if not self._is_owner(interaction.user):
                await interaction.response.send_message("⛔ Unauthorised.", ephemeral=True)
                return
            names = self._orchestrator.expert_names
            text = "🤖 **Registered Experts**\n\n" + "\n".join(f"• {n}" for n in names)
            await interaction.response.send_message(text)


# ---------------------------------------------------------------------- #
# Typing indicator                                                        #
# ---------------------------------------------------------------------- #


async def _keep_typing(channel: discord.abc.Messageable, stop: asyncio.Event) -> None:
    """Keep the typing indicator alive every 8 s until *stop* is set.

    Discord's typing indicator lasts ~10 s, so refresh at 8 s to stay visible.
    """
    while not stop.is_set():
        try:
            async with channel.typing():
                await asyncio.sleep(8)
        except Exception:
            logger.debug("discord_typing_action_failed", exc_info=True)
            await asyncio.sleep(8)


# ---------------------------------------------------------------------- #
# Text helpers                                                            #
# ---------------------------------------------------------------------- #


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

    from career.expert import CareerExpert
    from creative.expert import CreativeExpert
    from goals.expert import GoalExpert
    from intelligence.expert import IntelligenceExpert
    from orchestrator.main import Orchestrator
    from spiritual.expert import SpiritualExpert

    orch = Orchestrator()
    for expert in (
        SpiritualExpert(),
        CareerExpert(),
        IntelligenceExpert(),
        CreativeExpert(),
        GoalExpert(),
    ):
        orch.register(expert)

    bot = SovereignEdgeDiscordBot(orch)

    # Wire morning brief delivery so the scheduler can push to Discord
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
