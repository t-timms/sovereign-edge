"""Speech-to-text via faster-whisper.

faster-whisper runs the Whisper model in CTranslate2 format — significantly
faster than the original HuggingFace Whisper on CPU/GPU.

Install (not in workspace deps — install in mlenv):
    pip install faster-whisper sounddevice numpy

On Jetson (ARM64):
    pip install faster-whisper --extra-index-url https://pypi.org/simple/

Usage:
    recognizer = SpeechRecognizer(model_size="base.en")
    text = recognizer.record_and_transcribe(max_seconds=10.0)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_SAMPLE_RATE = 16_000


class SpeechRecognizer:
    """Records audio from microphone and transcribes via faster-whisper.

    Parameters
    ----------
    model_size:
        faster-whisper model name — defaults to Settings.stt_model (base.en).
        Override in .env: STT_MODEL=small.en for higher accuracy.
    device:
        "cpu" or "cuda" — auto-selected based on availability
    compute_type:
        "int8" for CPU (fastest), "float16" for CUDA
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str = "auto",
        compute_type: str | None = None,
    ) -> None:
        if model_size is None:
            from core.config import get_settings

            model_size = get_settings().stt_model
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore

                device = self._device
                if device == "auto":
                    try:
                        import torch

                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"

                compute_type = self._compute_type
                if compute_type is None:
                    compute_type = "int8" if device == "cpu" else "float16"

                logger.info(
                    "voice.stt.loading_model",
                    model=self._model_size,
                    device=device,
                    compute_type=compute_type,
                )
                self._model = WhisperModel(
                    self._model_size,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info("voice.stt.model_ready", model=self._model_size)
            except ImportError:
                logger.error("voice.stt.faster_whisper_not_installed")
                raise
        return self._model

    def record(self, max_seconds: float = 10.0, silence_threshold: float = 0.01) -> np.ndarray:
        """Record audio from the default microphone until silence or timeout.

        Parameters
        ----------
        max_seconds:
            Maximum recording duration in seconds.
        silence_threshold:
            RMS amplitude below which audio is considered silence.

        Returns
        -------
        numpy array of int16 samples at 16kHz.
        """
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.error("voice.stt.sounddevice_not_installed")
            raise

        logger.info("voice.stt.recording", max_seconds=max_seconds)
        samples = sd.rec(
            int(max_seconds * _SAMPLE_RATE),
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()  # Wait for recording to finish
        audio = samples.flatten()
        logger.info("voice.stt.recorded", samples=len(audio), duration_s=len(audio) / _SAMPLE_RATE)
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio samples to text.

        Parameters
        ----------
        audio:
            int16 audio samples at 16kHz.

        Returns
        -------
        Transcribed text string (empty string if nothing detected).
        """
        model = self._get_model()

        # faster-whisper requires float32 normalized to [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        # Write to a temporary WAV file — faster-whisper reads from file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            import scipy.io.wavfile as wav  # type: ignore

            wav.write(str(tmp_path), _SAMPLE_RATE, (audio_f32 * 32768).astype(np.int16))
            segments, info = model.transcribe(  # type: ignore[union-attr]
                str(tmp_path),
                beam_size=5,
                language="en",
                vad_filter=True,  # Voice Activity Detection — skip silence
                vad_parameters={"min_silence_duration_ms": 500},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logger.info(
                "voice.stt.transcribed",
                text_length=len(text),
                language=info.language,
                probability=round(info.language_probability, 3),
            )
            return text
        finally:
            tmp_path.unlink(missing_ok=True)

    def record_and_transcribe(self, max_seconds: float = 10.0) -> str:
        """Convenience: record then transcribe in one call."""
        audio = self.record(max_seconds=max_seconds)
        return self.transcribe(audio)

    def is_available(self) -> bool:
        """Return True if faster-whisper and sounddevice are importable."""
        try:
            import faster_whisper  # type: ignore  # noqa: F401
            import sounddevice  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False
