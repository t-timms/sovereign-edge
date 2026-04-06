"""
ONNX DistilBERT intent classifier.

Two modes:
  1. ONNX mode (production): Loads fine-tuned model from data/models/router.onnx
  2. Keyword mode (bootstrap): Simple keyword matching until ONNX model is trained

The ONNX model should be trained using scripts/train_router.py with synthetic data.
Until then, keyword mode provides ~85% accuracy for development.

Low-confidence threshold: if score <= LOW_CONFIDENCE_THRESHOLD and intent is GENERAL,
the caller should consider prompting the user for clarification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from core.config import Settings, get_settings
from core.exceptions import RouterError
from core.types import Intent, IntentClass, RouterResult, RoutingDecision
from observability.logging import get_logger

from router.pii import PIIDetector

logger = get_logger(__name__, component="router")

# If confidence is at or below this value AND intent is GENERAL, the classification
# is unreliable. The bot uses this to offer a gentle clarification prompt.
LOW_CONFIDENCE_THRESHOLD = 0.55


# Keyword fallback for bootstrapping before ONNX model is trained
# Keys are lowercase substrings — multi-word keys are checked first (specificity ordering)
_KEYWORD_MAP: dict[str, Intent] = {
    # Multi-word (checked first for specificity)
    "linkedin post": Intent.CREATIVE,
    "linkedin article": Intent.CREATIVE,
    "cover letter": Intent.CAREER,
    "job description": Intent.CAREER,
    "machine learning": Intent.INTELLIGENCE,
    "deep learning": Intent.INTELLIGENCE,
    "neural network": Intent.INTELLIGENCE,
    "language model": Intent.INTELLIGENCE,
    "fine tun": Intent.INTELLIGENCE,
    "fine-tun": Intent.INTELLIGENCE,
    # Spiritual
    "bible": Intent.SPIRITUAL,
    "verse": Intent.SPIRITUAL,
    "pray": Intent.SPIRITUAL,
    "devotion": Intent.SPIRITUAL,
    "scripture": Intent.SPIRITUAL,
    "god": Intent.SPIRITUAL,
    "jesus": Intent.SPIRITUAL,
    "yeshua": Intent.SPIRITUAL,
    "hebrew": Intent.SPIRITUAL,
    "greek": Intent.SPIRITUAL,
    "psalm": Intent.SPIRITUAL,
    "gospel": Intent.SPIRITUAL,
    "faith": Intent.SPIRITUAL,
    "church": Intent.SPIRITUAL,
    "worship": Intent.SPIRITUAL,
    "sermon": Intent.SPIRITUAL,
    "theology": Intent.SPIRITUAL,
    "proverb": Intent.SPIRITUAL,
    # Career
    "job": Intent.CAREER,
    "resume": Intent.CAREER,
    "interview": Intent.CAREER,
    "salary": Intent.CAREER,
    "hiring": Intent.CAREER,
    "application": Intent.CAREER,
    "linkedin": Intent.CAREER,
    "recruiter": Intent.CAREER,
    "career": Intent.CAREER,
    "apply": Intent.CAREER,
    "position": Intent.CAREER,
    "employer": Intent.CAREER,
    "offer letter": Intent.CAREER,
    "compensation": Intent.CAREER,
    # Intelligence
    "arxiv": Intent.INTELLIGENCE,
    "paper": Intent.INTELLIGENCE,
    "research": Intent.INTELLIGENCE,
    "news": Intent.INTELLIGENCE,
    "trend": Intent.INTELLIGENCE,
    "analysis": Intent.INTELLIGENCE,
    "model": Intent.INTELLIGENCE,
    "benchmark": Intent.INTELLIGENCE,
    "inference": Intent.INTELLIGENCE,
    "training": Intent.INTELLIGENCE,
    "dataset": Intent.INTELLIGENCE,
    "huggingface": Intent.INTELLIGENCE,
    "llama": Intent.INTELLIGENCE,
    # Goals
    "add goal": Intent.GOALS,
    "my goals": Intent.GOALS,
    "list goals": Intent.GOALS,
    "show goals": Intent.GOALS,
    "goal": Intent.GOALS,
    "habit": Intent.GOALS,
    "objective": Intent.GOALS,
    "milestone": Intent.GOALS,
    "progress": Intent.GOALS,
    "complete goal": Intent.GOALS,
    # Creative
    "video": Intent.CREATIVE,
    "content": Intent.CREATIVE,
    "post": Intent.CREATIVE,
    "thumbnail": Intent.CREATIVE,
    "script": Intent.CREATIVE,
    "manim": Intent.CREATIVE,
    "animation": Intent.CREATIVE,
    "shorts": Intent.CREATIVE,
    "write": Intent.CREATIVE,
    "story": Intent.CREATIVE,
    "caption": Intent.CREATIVE,
    "blog": Intent.CREATIVE,
    "tweet": Intent.CREATIVE,
    "thread": Intent.CREATIVE,
}

# Sort by length descending so multi-word keys are checked first
_SORTED_KEYWORDS = sorted(_KEYWORD_MAP.items(), key=lambda x: len(x[0]), reverse=True)

# Minimum cosine similarity to trust embedding classification over keyword fallback
_EMBEDDING_CONFIDENCE_THRESHOLD = 0.45

# Representative prototype sentences per intent — used for embedding similarity routing
_INTENT_PROTOTYPES: dict[Intent, list[str]] = {
    Intent.SPIRITUAL: [
        "What does the Bible say about forgiveness?",
        "Write me a prayer for strength and guidance",
        "Explain the meaning of Psalm 23",
        "I need a morning devotional for today",
        "What does this scripture verse mean in context?",
        "Help me understand the Sermon on the Mount",
    ],
    Intent.CAREER: [
        "Find me ML engineer jobs in Dallas Fort Worth",
        "Help me improve my resume for AI engineering roles",
        "How should I prepare for a machine learning interview?",
        "What is the salary range for LLM engineers in DFW?",
        "Write a cover letter for this job description",
        "Which companies in Dallas are hiring AI engineers?",
    ],
    Intent.INTELLIGENCE: [
        "What are the latest breakthroughs in large language models?",
        "Summarize recent arXiv papers on transformer architecture",
        "What is the current state of fine-tuning techniques for LLMs?",
        "Explain the GRPO training method used in DeepSeek-R1",
        "What inference optimization works best on consumer GPUs?",
        "What papers came out this week on AI agents?",
    ],
    Intent.CREATIVE: [
        "Write a LinkedIn post about my AI project",
        "Help me create a content strategy for technical creators",
        "Generate a short story about technology and humanity",
        "Write a YouTube script about machine learning",
        "Create a Twitter thread about LLM trends",
        "Draft a blog post about my experience with fine-tuning",
    ],
    Intent.GOALS: [
        "Add a goal to land an ML engineer job by June",
        "Show me my active goals and progress",
        "Update goal #1 to 75% complete",
        "Mark goal #2 as done",
        "What are my current objectives?",
        "I want to track a new habit",
    ],
    Intent.GENERAL: [
        "What time is it?",
        "Help me think through this general problem",
        "I have a random question for you",
        "Can you help me with something?",
    ],
}

# Map internal Intent (uppercase) → IntentClass (lowercase)
_INTENT_TO_CLASS: dict[Intent, IntentClass] = {
    Intent.SPIRITUAL: IntentClass.SPIRITUAL,
    Intent.CAREER: IntentClass.CAREER,
    Intent.INTELLIGENCE: IntentClass.INTELLIGENCE,
    Intent.CREATIVE: IntentClass.CREATIVE,
    Intent.GOALS: IntentClass.GOALS,
    Intent.GENERAL: IntentClass.GENERAL,
}


def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax with numerical stability."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


class IntentRouter:
    """Routes user input to the appropriate expert."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_path: str | None = None,
    ) -> None:
        s = settings or get_settings()
        # Resolve model path: explicit arg takes precedence, then settings
        resolved_path = model_path or (str(s.router_model_path) if s.router_model_path else None)

        self.pii_detector = PIIDetector()
        self._onnx_session = None
        self._tokenizer = None
        self._use_onnx = False

        if resolved_path and Path(resolved_path).exists():
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer

                self._onnx_session = ort.InferenceSession(
                    resolved_path,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                tokenizer_dir = Path(resolved_path).parent / "tokenizer"
                tokenizer_source = str(tokenizer_dir) if tokenizer_dir.exists() else "distilbert-base-uncased"
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
                self._use_onnx = True
                logger.info("onnx_classifier_loaded path=%s", resolved_path)
            except Exception:
                logger.warning(
                    "onnx_load_failed — falling back to keyword classifier",
                    exc_info=True,
                )
        else:
            logger.info("onnx_model_not_found — using keyword classifier (bootstrap mode)")

    def classify(self, text: str) -> RouterResult:
        """Classify user intent. Returns RouterResult with intent and confidence.

        Raises RouterError for empty or whitespace-only input.
        Unknown input defaults to IntentClass.GENERAL.
        """
        if not text or not text.strip():
            raise RouterError("Input text must not be empty")

        if self._use_onnx and self._onnx_session and self._tokenizer:
            intent, confidence = self._classify_onnx(text)
        else:
            intent, confidence = self._classify_keywords(text)

        return RouterResult(
            intent=_INTENT_TO_CLASS[intent],
            confidence=confidence,
        )

    def route(self, text: str) -> tuple[Intent, float, RoutingDecision]:
        """Full routing: classify intent + determine processing location."""
        result = self.classify(text)
        # Convert IntentClass back to Intent for backward-compat callers
        intent = Intent(result.intent.upper())
        confidence = result.confidence

        if self.pii_detector.contains_pii(text):
            logger.info("pii_detected routing=LOCAL intent=%s", intent.value)
            return intent, confidence, RoutingDecision.LOCAL

        if intent == Intent.GENERAL and confidence <= LOW_CONFIDENCE_THRESHOLD:
            logger.debug(
                "low_confidence_classification intent=%s confidence=%.2f",
                intent.value,
                confidence,
            )

        return intent, confidence, RoutingDecision.CLOUD

    async def aroute(self, text: str) -> tuple[Intent, float, RoutingDecision]:
        """Async routing — embedding similarity tier before keyword fallback.

        Classification priority:
          1. PII check → force LOCAL
          2. Embedding cosine similarity (Ollama, graceful degradation)
          3. ONNX model (if loaded)
          4. Keyword fallback
        """
        if self.pii_detector.contains_pii(text):
            result = self.classify(text)
            intent = Intent(result.intent.upper())
            logger.info("pii_detected routing=LOCAL intent=%s", intent.value)
            return intent, result.confidence, RoutingDecision.LOCAL

        # Attempt embedding classification
        e_intent, e_confidence = await self._classify_embeddings(text)
        if e_intent is not None and e_confidence is not None:
            logger.debug("embedding_route intent=%s confidence=%.2f", e_intent.value, e_confidence)
            return e_intent, e_confidence, RoutingDecision.CLOUD

        # Fall back to sync classifier (ONNX → keyword)
        result = self.classify(text)
        intent = Intent(result.intent.upper())
        return intent, result.confidence, RoutingDecision.CLOUD

    async def _classify_embeddings(self, text: str) -> tuple[Intent, float] | tuple[None, None]:
        """Embed query and score against per-intent prototype sentences.

        Returns (None, None) on any failure so callers can degrade gracefully.
        Prototype embeddings are computed via asyncio.to_thread to avoid blocking
        the event loop on the first call (subsequent calls are served from LRU cache).
        """
        try:
            import asyncio

            from memory.embeddings import aembed, cosine_similarity, embed_cached

            query_vec = await aembed(text)
            if query_vec is None:
                return None, None

            best_intent: Intent = Intent.GENERAL
            best_score: float = 0.0

            for intent, prototypes in _INTENT_PROTOTYPES.items():
                for proto in prototypes:
                    # Run in thread to avoid blocking the event loop on first (uncached) call
                    cached = await asyncio.to_thread(embed_cached, proto)
                    if cached is None:
                        continue
                    proto_vec = np.array(cached, dtype=np.float32)
                    score = cosine_similarity(query_vec, proto_vec)
                    if score > best_score:
                        best_score = score
                        best_intent = intent

            if best_score < _EMBEDDING_CONFIDENCE_THRESHOLD:
                return None, None  # uncertain — let keyword fallback decide

            confidence = float(min(0.98, best_score))
            return best_intent, confidence

        except Exception:
            logger.debug("_classify_embeddings_failed", exc_info=True)
            return None, None

    def _classify_onnx(self, text: str) -> tuple[Intent, float]:
        """Classify using fine-tuned DistilBERT ONNX model."""
        if self._tokenizer is None or self._onnx_session is None:
            logger.warning("onnx_classifier_unavailable — falling back to keyword classifier")
            return self._classify_keywords(text)

        inputs = self._tokenizer(
            text, return_tensors="np", truncation=True, max_length=128, padding="max_length"
        )
        input_dict = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        outputs = self._onnx_session.run(None, input_dict)
        probs = _softmax(outputs[0][0])
        idx = int(np.argmax(probs))
        labels = list(Intent)
        if idx >= len(labels):
            logger.warning(
                "onnx_idx_out_of_range idx=%d n_labels=%d — falling back to keyword classifier",
                idx,
                len(labels),
            )
            return self._classify_keywords(text)
        return labels[idx], float(probs[idx])

    def _classify_keywords(self, text: str) -> tuple[Intent, float]:
        """Keyword matching with multi-word priority (longer keys checked first)."""
        text_lower = text.lower()
        intent_scores: dict[Intent, int] = {i: 0 for i in Intent}

        for keyword, intent in _SORTED_KEYWORDS:
            if keyword in text_lower:
                # Weight multi-word matches more heavily
                weight = len(keyword.split())
                intent_scores[intent] += weight

        best_intent = max(intent_scores, key=intent_scores.get)  # type: ignore[arg-type]
        best_score = intent_scores[best_intent]

        if best_score == 0:
            # No keywords matched — fall back to GENERAL
            return Intent.GENERAL, 0.5

        # Normalize confidence (more/heavier keyword hits = higher confidence)
        confidence = min(0.95, 0.6 + (best_score * 0.08))
        return best_intent, confidence
