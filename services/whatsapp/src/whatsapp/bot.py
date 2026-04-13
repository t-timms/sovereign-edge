"""
Sovereign Edge WhatsApp bot — Twilio webhook receiver.

Accepts inbound WhatsApp messages from the configured owner number,
routes them through the orchestrator, and replies via Twilio's API.

Endpoint:
  POST /webhook  — Twilio sends inbound message payloads here

Security:
  - Twilio signature validation on every request (RequestValidator)
  - Only the owner's WhatsApp number is authorised

Proactive messages:
  send_message(text) — mirrors Telegram bot signature so the orchestrator's
  register_send_fn() works with any bot type.

Run:
  uvicorn whatsapp.bot:app --host 0.0.0.0 --port 8081
  # Expose via ngrok for dev: ngrok http 8081
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any

import uvicorn
from core.config import get_settings
from core.types import TaskPriority, TaskRequest
from fastapi import FastAPI, Form, Header, HTTPException, Request, status
from observability.logging import get_logger

logger = get_logger(__name__, component="whatsapp")

# WhatsApp hard limit per message
_WHATSAPP_MESSAGE_LIMIT = 4096

app = FastAPI(
    title="Sovereign Edge WhatsApp",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
)

# ── Lazy orchestrator singleton ────────────────────────────────────────────────
# Initialised on first request to avoid blocking import at module load.

_orchestrator: Any = None
_router: Any = None


def _get_orchestrator() -> object:
    global _orchestrator, _router
    if _orchestrator is None:
        from career.expert import CareerExpert
        from creative.expert import CreativeExpert
        from goals.expert import GoalExpert
        from intelligence.expert import IntelligenceExpert
        from orchestrator.main import Orchestrator
        from router.classifier import IntentRouter
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
        _orchestrator = orch
        _router = IntentRouter()
    return _orchestrator


# ── Twilio helpers ─────────────────────────────────────────────────────────────


def _validate_twilio_signature(request_url: str, params: dict[str, str], signature: str) -> bool:
    """Return True if Twilio signature is valid."""
    try:
        from twilio.request_validator import RequestValidator

        settings = get_settings()
        auth_token = settings.twilio_auth_token.get_secret_value()
        if not auth_token:
            logger.warning("whatsapp_twilio_auth_token_missing")
            return False
        validator = RequestValidator(auth_token)
        return validator.validate(request_url, params, signature)
    except Exception:
        logger.error("whatsapp_signature_validation_failed", exc_info=True)
        return False


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


async def _send_whatsapp(to: str, body: str) -> None:
    """Send a WhatsApp message via Twilio (runs in thread — Twilio SDK is sync)."""
    settings = get_settings()
    account_sid = settings.twilio_account_sid.get_secret_value()
    auth_token = settings.twilio_auth_token.get_secret_value()
    from_number = settings.twilio_whatsapp_from

    if not all([account_sid, auth_token, from_number]):
        logger.warning("whatsapp_send_skipped — Twilio credentials not configured")
        return

    # Normalize: strip any existing whatsapp: prefix before re-adding so the
    # setting can store either "+1..." or "whatsapp:+1..." without double-prefixing.
    from_bare = from_number.removeprefix("whatsapp:")
    to_bare = to.removeprefix("whatsapp:")

    def _sync_send(chunk: str) -> None:
        from twilio.rest import Client

        client = Client(account_sid, auth_token)
        client.messages.create(
            from_=f"whatsapp:{from_bare}",
            to=f"whatsapp:{to_bare}",
            body=chunk,
        )

    for chunk in _split(body, _WHATSAPP_MESSAGE_LIMIT):
        try:
            await asyncio.to_thread(_sync_send, chunk)
        except Exception:
            logger.error("whatsapp_send_failed", exc_info=True)
            return


async def send_message(text: str) -> None:
    """Proactive send — push text to the owner (morning briefs).

    Mirrors Telegram/Discord send_message() so orchestrator's
    register_send_fn() works with any bot type.
    """
    settings = get_settings()
    owner = settings.whatsapp_owner_number
    if not owner:
        return
    await _send_whatsapp(owner, text)


# ── Webhook endpoint ───────────────────────────────────────────────────────────


@app.post("/webhook")
async def webhook(
    request: Request,
    From: Annotated[str, Form()] = "",
    Body: Annotated[str, Form()] = "",
    x_twilio_signature: Annotated[str | None, Header()] = None,
) -> dict[str, str]:
    """Receive and process inbound WhatsApp messages."""
    settings = get_settings()

    # ── 0. Content-Type guard ────────────────────────────────────────────────
    content_type = request.headers.get("content-type", "")
    _is_form = (
        "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type
    )
    if not _is_form:
        logger.warning("whatsapp_bad_content_type content_type=%r", content_type)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Expected application/x-www-form-urlencoded",
        )

    # ── 1. Signature validation ──────────────────────────────────────────────
    form_data = await request.form()
    params = dict(form_data)
    request_url = str(request.url)
    signature = x_twilio_signature or ""

    if not _validate_twilio_signature(request_url, params, signature):
        logger.critical(
            "whatsapp_invalid_signature",
            url=request_url,
            from_number=From,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Twilio signature",
        )

    # ── 2. Owner check ──────────────────────────────────────────────────────
    owner_raw = settings.whatsapp_owner_number
    # Twilio sends From as "whatsapp:+1234567890"
    sender = From.replace("whatsapp:", "")
    owner = owner_raw.replace("whatsapp:", "")

    if sender != owner:
        logger.critical(
            "whatsapp_unauthorised_sender",
            sender=sender,
            expected=owner,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorised sender",
        )

    # ── 3. Empty body + length guard ─────────────────────────────────────────
    _MAX_INPUT_CHARS = 2000
    text = Body.strip()
    if not text:
        return {"status": "ok"}
    if len(text) > _MAX_INPUT_CHARS:
        logger.info("input_truncated sender=%s original_len=%d", sender, len(text))
        text = text[:_MAX_INPUT_CHARS]

    # ── 4. Dispatch through orchestrator ────────────────────────────────────
    try:
        orch = _get_orchestrator()
        intent, _confidence, routing = await _router.aroute(text)
        task = TaskRequest(
            content=text,
            intent=intent,
            priority=TaskPriority.HIGH,
            routing=routing,
            context={"chat_id": sender},
        )

        t0 = time.monotonic()
        result = await orch.dispatch(task)
        latency_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "whatsapp_message_handled",
            intent=intent.value,
            latency_ms=round(latency_ms, 1),
        )

        await _send_whatsapp(From.replace("whatsapp:", ""), result.content)

    except Exception:
        logger.error("whatsapp_dispatch_failed", exc_info=True)
        await _send_whatsapp(
            From.replace("whatsapp:", ""),
            "Something went wrong. Please try again.",
        )

    return {"status": "ok"}


# ── Standalone entry point ─────────────────────────────────────────────────────


def main() -> None:
    from observability.logging import setup_logging

    setup_logging(debug=False)
    uvicorn.run(
        "whatsapp.bot:app",
        host="0.0.0.0",  # noqa: S104 — container-internal only; expose via reverse proxy
        port=8081,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
