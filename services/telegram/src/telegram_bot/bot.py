"""
Sovereign Edge Telegram bot — primary user interface.

Accepts text messages, routes them through the orchestrator, and
returns responses. Only the owner chat ID is authorised.

Commands:
  /start   — greeting
  /health  — expert health status
  /stats   — today's usage and cost stats (TraceStore)
  /status  — current expert + routing info
  /experts  — list registered experts
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
from router.classifier import LOW_CONFIDENCE_THRESHOLD, IntentRouter
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
    "👁️ <b>Sovereign Edge online.</b>\n\n"
    "Send any message and I'll route it to the right expert.\n"
    "Use /health to check system status.\n"
    "Use /stats to see today's usage."
)

# Maximum characters accepted from user input — prevents context-window flooding
_MAX_INPUT_CHARS = 2000

# Maximum characters buffered from a single streaming response — prevents OOM on runaway LLMs
_MAX_BUFFER_CHARS = 16_000

# Maximum file size accepted for document upload — prevents large binary downloads
_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB

# Per-chat rate limiting: max 1 request per N seconds
_RATE_LIMIT_SECONDS = 2.0
# chat_id → monotonic timestamp of last allowed request
_last_request: dict[str, float] = {}

# Streaming: minimum seconds between Telegram message edits (rate limit is ~1/s per message)
_STREAM_EDIT_INTERVAL = 0.8


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
        token = self._settings.telegram_bot_token.get_secret_value()
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
        self._app.add_handler(CommandHandler("experts", self._cmd_experts))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))
        self._app.add_handler(MessageHandler(filters.Document.ALL, self._on_document))

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
                parse_mode="HTML",
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
        if update.message is None:
            return
        await update.message.reply_text(_WELCOME, parse_mode="HTML")

    @_auth
    async def _cmd_health(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return
        await update.message.reply_text("🔍 Checking experts…")
        health = await self._orchestrator.health_check_all()
        lines = [f"{'✅' if ok else '❌'} <b>{name}</b>" for name, ok in sorted(health.items())]
        text = "🏥 <b>Expert Health</b>\n\n" + "\n".join(lines) if lines else "No experts registered."
        await update.message.reply_text(text, parse_mode="HTML")

    @_auth
    async def _cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Today's LLM usage stats from TraceStore."""
        if update.message is None:
            return
        stats = self._orchestrator.get_daily_stats()
        if not stats:
            await update.message.reply_text("No stats recorded today yet.")
            return

        avg_ms = stats.get("avg_latency_ms", 0.0)
        text = (
            "📊 <b>Today's Stats</b>\n\n"
            f"Requests: {stats.get('total_requests', 0)}\n"
            f"Cache hits: {stats.get('cache_hits', 0)}\n"
            f"Errors: {stats.get('errors', 0)}\n"
            f"Avg latency: {avg_ms:.0f}ms\n"
            f"Tokens in: {stats.get('total_tokens_in', 0):,}\n"
            f"Tokens out: {stats.get('total_tokens_out', 0):,}\n"
            f"Total cost: ${stats.get('total_cost_usd', 0.0):.4f}\n"
            f"Models: {stats.get('models_used') or 'none'}"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    @_auth
    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return
        names = self._orchestrator.expert_names  # public property — no private access
        text = (
            "⚡ <b>Sovereign Edge Status</b>\n\n"
            f"Experts: {len(names)}\n"
            f"Registered: {', '.join(names) or 'none'}"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    @_auth
    async def _cmd_experts(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return
        names = self._orchestrator.expert_names  # public property — no private access
        text = "🤖 <b>Registered Experts</b>\n\n" + "\n".join(f"• {n}" for n in names)
        await update.message.reply_text(text, parse_mode="HTML")

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

        # Low-confidence general query — nudge the user toward a specific expert.
        # Still dispatches so they get an answer; the hint improves future routing.
        if intent.value == "GENERAL" and confidence <= LOW_CONFIDENCE_THRESHOLD:
            await update.message.reply_text(
                "🤔 _Not sure which expert fits this — routing to intelligence._\n"
                "For better results try: Bible/faith · job search · AI research · content creation",
                parse_mode="HTML",
            )

        request = TaskRequest(
            content=user_text,
            intent=intent,
            priority=TaskPriority.HIGH,
            routing=routing,
            context={"chat_id": chat_id},
        )

        # Send typing indicator while papers are being fetched
        await ctx.bot.send_chat_action(chat_id=tg_chat_id, action="typing")

        # Send a placeholder message — streaming edits this in place
        placeholder = await update.message.reply_text("…")

        buffer = ""
        last_edit = time.monotonic()
        try:
            async for chunk in self._orchestrator.stream_dispatch(request):
                buffer += chunk
                if len(buffer) >= _MAX_BUFFER_CHARS:
                    buffer = buffer[:_MAX_BUFFER_CHARS]
                    break
                # Edit at most every 800 ms to stay within Telegram's rate limit
                if time.monotonic() - last_edit >= _STREAM_EDIT_INTERVAL and buffer:
                    try:
                        await placeholder.edit_text(
                            _sanitize_markdown(buffer + " ▌"), parse_mode="HTML"
                        )
                        last_edit = time.monotonic()
                    except Exception:
                        logger.debug("stream_edit_failed", exc_info=True)
        except Exception:
            logger.error("stream_dispatch_failed", exc_info=True)
            buffer = buffer or "⚠️ Something went wrong. Please try again."

        # Final edit — clean text, no cursor; overflow into extra messages if needed
        chunks_out = _split(_sanitize_markdown(buffer), 4000)
        primary_sent = False
        try:
            await placeholder.edit_text(chunks_out[0], parse_mode="HTML")
            primary_sent = True
        except Exception:
            try:
                await placeholder.edit_text(chunks_out[0])
                primary_sent = True
            except Exception:
                logger.error("final_edit_failed — falling back to new message", exc_info=True)
        if not primary_sent:
            try:
                await update.message.reply_text(chunks_out[0], parse_mode="HTML")
            except Exception:
                await update.message.reply_text(chunks_out[0])
        for overflow in chunks_out[1:]:
            try:
                await update.message.reply_text(overflow, parse_mode="HTML")
            except Exception:
                await update.message.reply_text(overflow)

        logger.info(
            "message_handled",
            intent=intent.value,
            confidence=round(confidence, 2),
            routing=routing.value,
            chars=len(buffer),
        )

    # ------------------------------------------------------------------ #
    # Document / file handler                                             #
    # ------------------------------------------------------------------ #

    @_auth
    async def _on_document(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle file uploads — extract text and route to the appropriate expert.

        Supported: PDF (pdfplumber), plain text (.txt / .md), DOCX (python-docx).
        Files are downloaded to a temp path, extracted, then deleted — never stored.
        The file name and type inform routing:
          - resume.pdf / cv.pdf → CAREER expert
          - *.pdf with no signal   → INTELLIGENCE (treat as research paper)
          - *.txt / *.md          → route by content via IntentRouter
        """
        import tempfile
        from pathlib import Path

        doc = update.message.document
        if doc is None:
            return

        fname = (doc.file_name or "").lower()
        chat_id = str(update.effective_chat.id)

        if doc.file_size and doc.file_size > _MAX_FILE_BYTES:
            await update.message.reply_text(
                f"⚠️ File too large ({doc.file_size // (1024 * 1024)} MB). Maximum is 20 MB."
            )
            return

        await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        # Download to temp file
        try:
            tg_file = await ctx.bot.get_file(doc.file_id)
            suffix = Path(fname).suffix or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            await tg_file.download_to_drive(str(tmp_path))
        except Exception:
            logger.error("document_download_failed fname=%s", fname, exc_info=True)
            await update.message.reply_text("⚠️ Could not download the file.")
            return

        # Extract text
        text = ""
        try:
            text = _extract_file_text(tmp_path, fname)
        except Exception:
            logger.error("document_extract_failed fname=%s", fname, exc_info=True)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not text.strip():
            await update.message.reply_text(
                "⚠️ Could not extract text from this file. "
                "Supported formats: PDF, DOCX, TXT, MD."
            )
            return

        # Determine intent from filename signal first, then content
        if any(kw in fname for kw in ("resume", "cv", "cover")):
            from core.types import Intent, RoutingDecision
            intent, routing = Intent.CAREER, RoutingDecision.CLOUD
            confidence = 0.9
        else:
            intent, confidence, routing = await self._router.aroute(text[:500])

        content = (
            f"[File: {doc.file_name}]\n\n{text[:6000]}"
            if len(text) > 6000
            else f"[File: {doc.file_name}]\n\n{text}"
        )
        request = TaskRequest(
            content=content,
            intent=intent,
            priority=TaskPriority.HIGH,
            routing=routing,
            context={"chat_id": chat_id},
        )

        await update.message.reply_text(
            f"📄 Processing _{doc.file_name}_ → *{intent.value.lower()}* expert…",
            parse_mode="HTML",
        )

        placeholder = await update.message.reply_text("…")
        buffer = ""
        last_edit = time.monotonic()
        try:
            async for chunk in self._orchestrator.stream_dispatch(request):
                buffer += chunk
                if len(buffer) >= _MAX_BUFFER_CHARS:
                    buffer = buffer[:_MAX_BUFFER_CHARS]
                    break
                if time.monotonic() - last_edit >= _STREAM_EDIT_INTERVAL and buffer:
                    try:
                        await placeholder.edit_text(
                            _sanitize_markdown(buffer + " ▌"), parse_mode="HTML"
                        )
                        last_edit = time.monotonic()
                    except Exception:
                        logger.debug("doc_stream_edit_failed", exc_info=True)
        except Exception:
            logger.error("doc_stream_dispatch_failed", exc_info=True)
            buffer = buffer or "⚠️ Something went wrong processing the file."

        clean = _sanitize_markdown(buffer)
        chunks_out = _split(clean, 4000)
        doc_primary_sent = False
        try:
            await placeholder.edit_text(chunks_out[0], parse_mode="HTML")
            doc_primary_sent = True
        except Exception:
            try:
                await placeholder.edit_text(chunks_out[0])
                doc_primary_sent = True
            except Exception:
                logger.error("doc_final_edit_failed — falling back to new message", exc_info=True)
        if not doc_primary_sent:
            try:
                await update.message.reply_text(chunks_out[0], parse_mode="HTML")
            except Exception:
                await update.message.reply_text(chunks_out[0])
        for overflow in chunks_out[1:]:
            try:
                await update.message.reply_text(overflow, parse_mode="HTML")
            except Exception:
                await update.message.reply_text(overflow)

        logger.info(
            "document_handled fname=%s intent=%s confidence=%.2f chars=%d",
            fname, intent.value, confidence, len(buffer),
        )


def _extract_file_text(path: object, fname: str) -> str:
    """Extract plain text from a file. Supports PDF, DOCX, TXT, MD.

    Requires optional deps: pdfplumber (PDF), python-docx (DOCX).
    Falls back to raw UTF-8 read for text files. Returns empty string on failure.
    """
    from pathlib import Path

    p = Path(str(path))
    ext = p.suffix.lower()

    if ext == ".pdf":
        try:
            import pdfplumber  # type: ignore[import-untyped]

            with pdfplumber.open(str(p)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages[:30]]
            return "\n\n".join(pages).strip()
        except ImportError:
            logger.warning("pdfplumber not installed — PDF extraction unavailable")
            return ""

    if ext in (".docx",):
        try:
            import docx  # type: ignore[import-untyped]

            doc = docx.Document(str(p))
            return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
        except ImportError:
            logger.warning("python-docx not installed — DOCX extraction unavailable")
            return ""

    if ext in (".txt", ".md", ".rst", ".csv"):
        try:
            return p.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    logger.info("unsupported_file_type ext=%s fname=%s", ext, fname)
    return ""


async def _keep_typing(bot: Any, chat_id: int, stop: asyncio.Event) -> None:  # noqa: ANN401
    """Refresh the typing indicator every 4 s until *stop* is set."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            logger.debug("typing_action_failed", exc_info=True)
        await asyncio.sleep(4)


def _sanitize_markdown(text: str) -> str:
    """Convert LLM Markdown output to Telegram HTML.

    HTML parse mode is used instead of MarkdownV1 because unbalanced * or _
    in LLM output trigger Telegram's "Can't parse entities" error with Markdown,
    causing the entire message to fall back to plain text. HTML is forgiving of
    unclosed tags and gives predictable link rendering.
    """
    import html
    import re

    # Strip trailing boilerplate
    text = re.sub(
        r"\n*These (?:papers|results|findings|advancements|research).{0,300}$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Strip opening preamble lines
    text = re.sub(
        r"^(?:Here are|The following are|Below are|The latest)[^\n]*\n+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Strip mid-list transition sentences
    text = re.sub(r"\n+Additionally,[^\n]*\n+", "\n", text, flags=re.IGNORECASE)
    # Strip --- dividers
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.MULTILINE)
    # Strip ## headers (keep text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Extract Markdown links [text](url) before HTML-escaping so we can
    # reconstruct them as <a href="url">text</a> afterward.
    _LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)")
    link_placeholder: dict[str, tuple[str, str]] = {}

    def _stash_link(m: re.Match) -> str:  # type: ignore[type-arg]
        key = f"\x00LINK{len(link_placeholder)}\x00"
        link_placeholder[key] = (m.group(1), m.group(2))
        return key

    text = _LINK_RE.sub(_stash_link, text)

    # HTML-escape the remaining text (prevents injection, fixes < > & in titles)
    text = html.escape(text)

    # Restore links as <a href="url">text</a>
    for key, (link_text, url) in link_placeholder.items():
        safe_text = html.escape(link_text)
        safe_url = html.escape(url)
        text = text.replace(key, f'<a href="{safe_url}">{safe_text}</a>')

    # Convert bare URLs that are NOT already inside an <a> tag
    text = re.sub(
        r"(?<!href=\")(https?://[^\s<>&\"]+)",
        lambda m: f'<a href="{m.group(1)}">{m.group(1)}</a>',
        text,
    )

    # **bold** or *bold* → <b>bold</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"\*([^*\n]+?)\*", r"<b>\1</b>", text)

    # _italic_ → <i>italic</i>
    text = re.sub(r"_([^_\n]+?)_", r"<i>\1</i>", text)

    # Bold "Title: description" lines
    text = re.sub(
        r"^(?![<•\-])([^:\n]{10,80}):([ \t])",
        r"<b>\1</b>:\2",
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

    from career.expert import CareerExpert
    from creative.expert import CreativeExpert
    from intelligence.expert import IntelligenceExpert
    from orchestrator.main import Orchestrator
    from spiritual.expert import SpiritualExpert

    orch = Orchestrator(use_director=True)
    for expert in (SpiritualExpert(), CareerExpert(), IntelligenceExpert(), CreativeExpert()):
        orch.register(expert)

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
