"""Wake word detection via OpenWakeWord.

OpenWakeWord listens on the default microphone for a trigger phrase
("hey sovereign", "ok sovereign", or any custom model).

Install (not in workspace deps — install in mlenv):
    pip install openwakeword sounddevice numpy

Usage:
    detector = WakeWordDetector()
    # Blocking call — returns when wake word is detected
    detector.wait_for_wake()
"""

from __future__ import annotations

import time

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_CHUNK_SIZE = 1280  # 80ms at 16kHz — OpenWakeWord expects 1280 samples
_SAMPLE_RATE = 16_000
_THRESHOLD = 0.5  # Detection confidence threshold


class WakeWordDetector:
    """Listens on the microphone for the configured wake word model.

    Supports any OpenWakeWord model. Defaults to 'alexa' which ships
    bundled with OpenWakeWord; swap for a custom "hey_sovereign" model
    by passing the model path.

    Parameters
    ----------
    model_path:
        Path to a custom .onnx OpenWakeWord model, or None to use the
        bundled 'alexa' model (useful for dev/testing).
    threshold:
        Confidence threshold for detection (0–1, default 0.5).
    """

    def __init__(self, model_path: str | None = None, threshold: float = _THRESHOLD) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._oww: object | None = None

    def _get_oww(self) -> object:
        if self._oww is None:
            try:
                from openwakeword.model import Model  # type: ignore

                if self._model_path:
                    self._oww = Model(
                        wakeword_models=[self._model_path], inference_framework="onnx"
                    )
                else:
                    # Use the built-in 'alexa' model for development
                    self._oww = Model(inference_framework="onnx")
                logger.info("voice.wake.model_loaded", model=self._model_path or "default")
            except ImportError:
                logger.error("voice.wake.openwakeword_not_installed")
                raise
        return self._oww

    def wait_for_wake(self, timeout_sec: float = 0.0) -> bool:
        """Block until wake word is detected or timeout expires.

        Parameters
        ----------
        timeout_sec:
            Maximum seconds to wait (0 = infinite).

        Returns
        -------
        True if wake word detected, False if timed out.
        """
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.error("voice.wake.sounddevice_not_installed")
            raise

        oww = self._get_oww()
        logger.info("voice.wake.listening", threshold=self._threshold)

        start = time.monotonic()
        with sd.InputStream(
            samplerate=_SAMPLE_RATE, channels=1, dtype="int16", blocksize=_CHUNK_SIZE
        ) as stream:
            while True:
                if timeout_sec > 0 and (time.monotonic() - start) > timeout_sec:
                    logger.info("voice.wake.timeout")
                    return False

                chunk, _ = stream.read(_CHUNK_SIZE)
                audio = chunk.flatten().astype(np.int16)
                predictions = oww.predict(audio)  # type: ignore[union-attr]

                for model_name, score in predictions.items():
                    if score >= self._threshold:
                        logger.info("voice.wake.detected", model=model_name, score=float(score))
                        return True

    def is_available(self) -> bool:
        """Return True if OpenWakeWord and sounddevice are importable."""
        try:
            import openwakeword  # type: ignore  # noqa: F401
            import sounddevice  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False
