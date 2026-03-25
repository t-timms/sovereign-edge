"""
ONNX DistilBERT intent classifier.

Two modes:
  1. ONNX mode (production): Loads fine-tuned model from data/models/router.onnx
  2. Keyword mode (bootstrap): Simple keyword matching until ONNX model is trained

The ONNX model should be trained using scripts/train_router.py with synthetic data.
Until then, keyword mode provides ~85% accuracy for development.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from core.types import Intent, RoutingDecision

from router.pii import PIIDetector

logger = logging.getLogger(__name__)


# Keyword fallback for bootstrapping before ONNX model is trained
_KEYWORD_MAP: dict[str, Intent] = {
    # Spiritual
    "bible": Intent.SPIRITUAL, "verse": Intent.SPIRITUAL, "pray": Intent.SPIRITUAL,
    "devotion": Intent.SPIRITUAL, "scripture": Intent.SPIRITUAL, "god": Intent.SPIRITUAL,
    "jesus": Intent.SPIRITUAL, "yeshua": Intent.SPIRITUAL, "hebrew": Intent.SPIRITUAL,
    "greek": Intent.SPIRITUAL, "psalm": Intent.SPIRITUAL, "gospel": Intent.SPIRITUAL,
    "faith": Intent.SPIRITUAL, "church": Intent.SPIRITUAL, "worship": Intent.SPIRITUAL,
    # Career
    "job": Intent.CAREER, "resume": Intent.CAREER, "interview": Intent.CAREER,
    "salary": Intent.CAREER, "hiring": Intent.CAREER, "application": Intent.CAREER,
    "linkedin": Intent.CAREER, "recruiter": Intent.CAREER, "career": Intent.CAREER,
    "apply": Intent.CAREER, "position": Intent.CAREER, "employer": Intent.CAREER,
    # Intelligence
    "market": Intent.INTELLIGENCE, "stock": Intent.INTELLIGENCE, "arxiv": Intent.INTELLIGENCE,
    "paper": Intent.INTELLIGENCE, "research": Intent.INTELLIGENCE, "news": Intent.INTELLIGENCE,
    "trend": Intent.INTELLIGENCE, "analysis": Intent.INTELLIGENCE, "price": Intent.INTELLIGENCE,
    # Creative
    "video": Intent.CREATIVE, "content": Intent.CREATIVE, "post": Intent.CREATIVE,
    "linkedin post": Intent.CREATIVE, "thumbnail": Intent.CREATIVE, "script": Intent.CREATIVE,
    "manim": Intent.CREATIVE, "animation": Intent.CREATIVE, "shorts": Intent.CREATIVE,
}


class IntentRouter:
    """Routes user input to the appropriate squad."""

    def __init__(self, model_path: str | None = None) -> None:
        self.pii_detector = PIIDetector()
        self._onnx_session = None
        self._tokenizer = None
        self._use_onnx = False

        if model_path and Path(model_path).exists():
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer

                self._onnx_session = ort.InferenceSession(
                    model_path,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self._use_onnx = True
                logger.info("ONNX intent classifier loaded from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load ONNX model, falling back to keywords: %s", e)
        else:
            logger.info("No ONNX model found — using keyword classifier (bootstrap mode)")

    def classify(self, text: str) -> tuple[Intent, float]:
        """Classify user intent. Returns (intent, confidence)."""
        if self._use_onnx and self._onnx_session and self._tokenizer:
            return self._classify_onnx(text)
        return self._classify_keywords(text)

    def route(self, text: str) -> tuple[Intent, float, RoutingDecision]:
        """Full routing: classify intent + determine processing location."""
        intent, confidence = self.classify(text)

        if self.pii_detector.contains_pii(text):
            return intent, confidence, RoutingDecision.LOCAL

        # TODO: Add semantic cache check here once LanceDB is wired up
        # if self._check_cache(text):
        #     return intent, confidence, RoutingDecision.CACHE

        return intent, confidence, RoutingDecision.CLOUD

    def _classify_onnx(self, text: str) -> tuple[Intent, float]:
        """Classify using fine-tuned DistilBERT ONNX model."""
        assert self._tokenizer is not None
        assert self._onnx_session is not None

        inputs = self._tokenizer(
            text, return_tensors="np", truncation=True, max_length=128, padding="max_length"
        )
        input_dict = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        outputs = self._onnx_session.run(None, input_dict)
        probs = self._softmax(outputs[0][0])
        idx = int(np.argmax(probs))
        labels = list(Intent)
        return labels[idx], float(probs[idx])

    def _classify_keywords(self, text: str) -> tuple[Intent, float]:
        """Simple keyword matching for bootstrap mode."""
        text_lower = text.lower()
        intent_scores: dict[Intent, int] = {i: 0 for i in Intent}

        for keyword, intent in _KEYWORD_MAP.items():
            if keyword in text_lower:
                intent_scores[intent] += 1

        best_intent = max(intent_scores, key=intent_scores.get)  # type: ignore[arg-type]
        best_score = intent_scores[best_intent]

        if best_score == 0:
            return Intent.GENERAL, 0.5

        # Normalize confidence (more keyword hits = higher confidence)
        confidence = min(0.95, 0.6 + (best_score * 0.1))
        return best_intent, confidence

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
