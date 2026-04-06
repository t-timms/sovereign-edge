"""Text-to-speech with Kokoro-82M as primary engine, Piper TTS as fallback.

Kokoro-82M (Apache 2.0):
    82M parameter model, 96x real-time on CPU, ~50 voices.
    Install: pip install kokoro soundfile

    pip install kokoro sounddevice

Piper TTS fallback (MIT):
    ~50ms latency on CPU, designed for embedded devices.
    Install: pip install piper-tts

    Voice models (~50MB each):
        python -c "from piper import download_voice; download_voice('en_US-lessac-medium')"

Usage:
    speaker = TextToSpeech()
    speaker.speak("Hello, I am Sovereign Edge.")
"""

from __future__ import annotations

import io
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_KOKORO_VOICE = "af_heart"  # American English, female — highest-rated Kokoro voice
_DEFAULT_PIPER_VOICE = "en_US-lessac-medium"
_KOKORO_SAMPLE_RATE = 24000
_PIPER_SAMPLE_RATE = 22050


class TextToSpeech:
    """Neural TTS — Kokoro-82M primary, Piper TTS fallback.

    Kokoro-82M delivers substantially higher audio quality (MOS 4.1 vs 3.5)
    at 96x real-time. Falls back to Piper automatically if Kokoro is not
    installed.

    Parameters
    ----------
    voice:
        Kokoro voice name (e.g. "af_heart", "am_michael") or Piper voice
        name (e.g. "en_US-lessac-medium") when using the Piper fallback.
    data_dir:
        Directory to cache downloaded Piper voice models.
    speed:
        Speaking rate multiplier (1.0 = normal, 0.8 = slower, 1.2 = faster).
    """

    def __init__(
        self,
        voice: str = _DEFAULT_KOKORO_VOICE,
        data_dir: Path | None = None,
        speed: float = 1.0,
    ) -> None:
        self._voice = voice
        self._data_dir = data_dir or (Path.home() / ".local" / "share" / "piper-voices")
        self._speed = speed
        self._kokoro_pipeline: object | None = None
        self._piper_model: object | None = None
        self._engine: str = "unknown"

    # ── Kokoro-82M ────────────────────────────────────────────────────────────

    def _get_kokoro(self) -> object:
        if self._kokoro_pipeline is None:
            from kokoro import KPipeline  # type: ignore

            lang_code = "a"  # American English; change to 'b' for British
            self._kokoro_pipeline = KPipeline(lang_code=lang_code)
            self._engine = "kokoro"
            logger.info("voice.tts.kokoro_ready", voice=self._voice)
        return self._kokoro_pipeline

    def _synthesize_kokoro(self, text: str) -> tuple[bytes, int]:
        """Synthesize via Kokoro. Returns (pcm_bytes, sample_rate)."""
        import numpy as np

        pipeline = self._get_kokoro()
        pcm_chunks: list[bytes] = []
        for samples, _sample_rate, _ in pipeline(  # type: ignore[call-arg]
            text, voice=self._voice, speed=self._speed
        ):
            arr = np.array(samples)
            # Kokoro returns float32 [-1, 1] — convert to int16
            pcm = (arr * 32767).astype(np.int16).tobytes()
            pcm_chunks.append(pcm)
        return b"".join(pcm_chunks), _KOKORO_SAMPLE_RATE

    # ── Piper TTS fallback ────────────────────────────────────────────────────

    def _get_piper(self) -> object:
        if self._piper_model is None:
            from piper import PiperVoice  # type: ignore

            voice = self._voice if "_" in self._voice else _DEFAULT_PIPER_VOICE
            model_path = self._data_dir / f"{voice}.onnx"
            config_path = self._data_dir / f"{voice}.onnx.json"

            if not model_path.exists():
                logger.info("voice.tts.downloading_piper_model", voice=voice)
                self._download_piper_voice(voice, self._data_dir)

            logger.info("voice.tts.loading_piper_model", voice=voice, path=str(model_path))
            self._piper_model = PiperVoice.load(str(model_path), config_path=str(config_path))
            self._engine = "piper"
            logger.info("voice.tts.piper_ready", voice=voice)
        return self._piper_model

    def _download_piper_voice(self, voice: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        import httpx

        base_url = "https://github.com/rhasspy/piper/releases/download/v0.0.2"
        for ext in (".onnx", ".onnx.json"):
            url = f"{base_url}/{voice}{ext}"
            out_path = dest / f"{voice}{ext}"
            if out_path.exists():
                continue
            logger.info("voice.tts.downloading", url=url)
            resp = httpx.get(url, follow_redirects=True, timeout=120.0)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            logger.info("voice.tts.downloaded", path=str(out_path), size_mb=len(resp.content) / 1e6)

    def _synthesize_piper(self, text: str) -> tuple[bytes, int]:
        """Synthesize via Piper. Returns (pcm_bytes, sample_rate)."""
        model = self._get_piper()
        buf = io.BytesIO()
        with model.to_audio(  # type: ignore[union-attr]
            text,
            length_scale=1.0 / self._speed,
        ) as audio_stream:
            for chunk in audio_stream:
                buf.write(chunk)
        return buf.getvalue(), _PIPER_SAMPLE_RATE

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Synthesize text to raw PCM int16 bytes.

        Returns
        -------
        (pcm_bytes, sample_rate) — pass both to sounddevice.play().
        """
        if not text or not text.strip():
            return b"", _KOKORO_SAMPLE_RATE

        logger.info("voice.tts.synthesizing", text_length=len(text))
        try:
            pcm, sr = self._synthesize_kokoro(text)
            logger.info("voice.tts.synthesized", engine="kokoro", bytes=len(pcm))
            return pcm, sr
        except ImportError:
            logger.warning("voice.tts.kokoro_unavailable_falling_back_to_piper")
        except Exception:
            logger.error("voice.tts.kokoro_synthesis_failed", exc_info=True)

        # Piper fallback
        pcm, sr = self._synthesize_piper(text)
        logger.info("voice.tts.synthesized", engine="piper", bytes=len(pcm))
        return pcm, sr

    def speak(self, text: str) -> None:
        """Synthesize and play audio through the default audio output device."""
        pcm, sample_rate = self.synthesize(text)
        if not pcm:
            return

        try:
            import numpy as np
            import sounddevice as sd  # type: ignore

            audio_array = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            logger.info("voice.tts.playing", duration_s=round(len(audio_array) / sample_rate, 2))
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()
            logger.info("voice.tts.done", engine=self._engine)
        except ImportError:
            logger.error("voice.tts.sounddevice_not_installed")
            raise
        except Exception:
            logger.error("voice.tts.playback_failed", exc_info=True)
            raise

    def speak_to_file(self, text: str, output_path: Path) -> None:
        """Synthesize text and write WAV file instead of playing."""
        pcm, sample_rate = self.synthesize(text)
        if not pcm:
            return

        import numpy as np
        import scipy.io.wavfile as wav  # type: ignore

        audio_array = np.frombuffer(pcm, dtype=np.int16)
        wav.write(str(output_path), sample_rate, audio_array)
        logger.info("voice.tts.saved", path=str(output_path), engine=self._engine)

    def is_available(self) -> bool:
        """Return True if at least one TTS engine and sounddevice are importable."""
        try:
            import sounddevice  # type: ignore  # noqa: F401
        except ImportError:
            return False

        try:
            import kokoro  # type: ignore  # noqa: F401

            return True
        except ImportError:
            pass

        try:
            import piper  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False
