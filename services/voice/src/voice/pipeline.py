"""Voice pipeline — wake word → STT → orchestrator → TTS.

Full loop:
  1. WakeWordDetector.wait_for_wake()   — blocks until "hey sovereign"
  2. SpeechRecognizer.record_and_transcribe()  — capture + transcribe user speech
  3. orchestrator.run_turn(text)        — LangGraph agents process the query
  4. TextToSpeech.speak(response)       — play synthesized response aloud
  5. Repeat

Run as a foreground process on the Jetson:
    uv run python -m voice
    uv run python -m voice --no-wake  # skip wake word (always listening)
"""

from __future__ import annotations

import asyncio

import structlog
from core.config import Settings, get_settings

from voice.stt import SpeechRecognizer
from voice.tts import TextToSpeech
from voice.wake import WakeWordDetector

logger = structlog.get_logger(__name__)

_WAKE_TIMEOUT = 0.0  # 0 = wait forever
_RECORD_SECONDS = 12.0
_FALLBACK_ERROR = "Sorry, I encountered an issue. Please try again."


class VoicePipeline:
    """Orchestrates the full wake→listen→respond→speak loop.

    Parameters
    ----------
    settings:
        Application settings. Defaults to get_settings().
    wake_model_path:
        Path to custom OpenWakeWord .onnx model (None = bundled default).
    tts_voice:
        Piper voice name (e.g. "en_US-lessac-medium").
    whisper_model:
        faster-whisper model size ("tiny.en", "base.en", "small.en").
        Defaults to settings.stt_model (base.en unless overridden in .env).
    thread_id:
        LangGraph conversation thread ID (persistent memory per session).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        wake_model_path: str | None = None,
        tts_voice: str = "en_US-lessac-medium",
        whisper_model: str | None = None,
        thread_id: str = "voice_default",
    ) -> None:
        self._settings = settings or get_settings()
        self._wake = WakeWordDetector(model_path=wake_model_path)
        self._stt = SpeechRecognizer(model_size=whisper_model or self._settings.stt_model)
        self._tts = TextToSpeech(voice=tts_voice)
        self._thread_id = thread_id
        self._running = False

    def check_deps(self) -> dict[str, bool]:
        """Return availability of each optional dependency."""
        return {
            "wake_word": self._wake.is_available(),
            "stt": self._stt.is_available(),
            "tts": self._tts.is_available(),
        }

    async def _run_turn(self, text: str) -> str:
        """Send transcribed text through the LangGraph orchestrator."""
        from orchestrator.graph import run_turn  # lazy import to avoid circular init

        try:
            response = await run_turn(text, thread_id=self._thread_id)
            return response or _FALLBACK_ERROR
        except Exception:
            logger.error("voice.pipeline.orchestrator_failed", exc_info=True)
            return _FALLBACK_ERROR

    def _speak_thinking(self) -> None:
        """Play a short acknowledgement while the LLM processes."""
        try:
            self._tts.speak("One moment.")
        except Exception:
            pass  # Non-critical — silence is fine

    async def run_single_turn(self, *, use_wake_word: bool = True) -> str | None:
        """Execute one full interaction cycle.

        Returns the assistant's response text, or None if no speech was detected.
        """
        if use_wake_word:
            logger.info("voice.pipeline.waiting_for_wake")
            detected = self._wake.wait_for_wake(timeout_sec=_WAKE_TIMEOUT)
            if not detected:
                return None

        logger.info("voice.pipeline.listening")
        try:
            text = self._stt.record_and_transcribe(max_seconds=_RECORD_SECONDS)
        except Exception:
            logger.error("voice.pipeline.stt_failed", exc_info=True)
            return None

        if not text.strip():
            logger.info("voice.pipeline.empty_transcription")
            self._tts.speak("I didn't catch that. Please try again.")
            return None

        logger.info("voice.pipeline.transcribed", text=text[:80])
        self._speak_thinking()

        response = await self._run_turn(text)
        logger.info("voice.pipeline.responding", length=len(response))

        try:
            self._tts.speak(response)
        except Exception:
            logger.error("voice.pipeline.tts_failed", exc_info=True)

        return response

    async def run_forever(self, *, use_wake_word: bool = True) -> None:
        """Main loop — run indefinitely until interrupted."""
        self._running = True
        logger.info(
            "voice.pipeline.started",
            wake_word=use_wake_word,
            thread_id=self._thread_id,
        )

        # Warm up STT model on first call
        try:
            import numpy as np

            self._stt.transcribe(np.zeros(1280, dtype=np.int16))
            logger.info("voice.pipeline.stt_warmed_up")
        except Exception:
            logger.warning("voice.pipeline.warmup_failed", exc_info=True)

        while self._running:
            try:
                await self.run_single_turn(use_wake_word=use_wake_word)
            except KeyboardInterrupt:
                break
            except Exception:
                logger.error("voice.pipeline.loop_error", exc_info=True)
                await asyncio.sleep(1)  # Brief backoff on unexpected errors

        logger.info("voice.pipeline.stopped")

    def stop(self) -> None:
        """Signal the pipeline to stop after the current turn."""
        self._running = False
