"""Generate synthetic training data for the DistilBERT intent router.

Writes JSONL to data/router_train.jsonl — format: {"text": "...", "label": 0-3}

Label mapping:
  0 = spiritual
  1 = career
  2 = intelligence
  3 = creative

Run:
    uv run python scripts/generate-synth-data.py
    uv run python scripts/generate-synth-data.py --output data/router_train.jsonl --samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Label index (must match IntentClass enum order) ───────────────────────────

LABEL_MAP: dict[str, int] = {
    "spiritual": 0,
    "career": 1,
    "intelligence": 2,
    "creative": 3,
}

# ── Seed examples per class ───────────────────────────────────────────────────

_SPIRITUAL: list[str] = [
    "What does the Bible say about forgiveness?",
    "Give me a verse about strength and courage",
    "Explain the meaning of John 3:16",
    "What are the fruits of the Spirit in Galatians?",
    "Write me a morning devotional for today",
    "Who wrote the book of Psalms?",
    "What does Proverbs say about wisdom?",
    "Find scripture about trusting God in hard times",
    "Explain the Sermon on the Mount",
    "What is the significance of the Lord's Prayer?",
    "Help me understand the book of Revelation",
    "What does Romans 8:28 mean?",
    "Give me a verse about God's grace",
    "What does the Old Testament say about tithing?",
    "Explain the parable of the prodigal son",
    "What is the Great Commission?",
    "Look up Isaiah 40:31 for me",
    "How should a Christian handle anxiety according to scripture?",
    "What does Hebrews 11 teach about faith?",
    "Give me a prayer for difficult times",
    "What does Jesus say about loving your enemies?",
    "Explain the Trinity in Christianity",
    "What is the significance of baptism in the New Testament?",
    "Find me verses about peace and contentment",
    "What does the gospel of Mark emphasize?",
    "Tell me about King David's prayer life",
    "What does the Bible say about the Sabbath?",
    "Explain the armor of God in Ephesians 6",
    "What does Colossians 3 say about the new self?",
    "Give me a devotional about serving others",
    "What are the Ten Commandments?",
    "Explain the significance of Pentecost",
    "What does Matthew 5 teach about the beatitudes?",
    "Find scripture about God's faithfulness",
    "What does the Bible say about money and wealth?",
    "Explain the meaning of grace vs works in Ephesians 2",
    "What does 1 Corinthians 13 say about love?",
    "Help me understand the book of Job",
    "What does the Bible say about marriage?",
    "Explain predestination vs free will in scripture",
]

_CAREER: list[str] = [
    "Find me ML Engineer jobs in Dallas",
    "Help me rewrite my resume for a machine learning role",
    "What should I say in a cover letter for an AI engineering position?",
    "Find software engineering openings in DFW",
    "How do I prepare for a system design interview?",
    "What's a good salary range for a senior ML engineer?",
    "Find AI engineer jobs at Capital One",
    "Review my resume for a data scientist role",
    "What questions do FAANG companies ask in ML interviews?",
    "Help me tailor my resume for a LangGraph job posting",
    "What are the best job boards for AI/ML roles?",
    "Find remote LLM engineer positions",
    "How do I negotiate a higher salary offer?",
    "What should I include in a LinkedIn profile for ML engineers?",
    "Find jobs at AT&T in Dallas for software engineers",
    "Help me write a cover letter for an AI Solutions Engineer position",
    "What certifications help for cloud ML engineering jobs?",
    "Find junior ML engineer positions near Plano TX",
    "How do I get referrals at big tech companies?",
    "What is a typical ML engineer interview process at Google?",
    "Find AI product manager roles in Fort Worth",
    "How do I present my LangGraph project in an interview?",
    "What does a typical ML engineer day look like?",
    "Search for NLP engineer openings at Microsoft",
    "Help me create a portfolio project for a data science job application",
    "What Python skills are most in demand for ML engineering?",
    "Find openings for AI engineers at American Airlines",
    "How do I write a strong technical resume with limited experience?",
    "What are the top companies hiring for LLMOps roles?",
    "Explain how to ace a take-home ML coding assignment",
]

_INTELLIGENCE: list[str] = [
    "What is NVDA stock price today?",
    "Show me the latest AI research papers",
    "What happened to Microsoft stock this week?",
    "Summarize recent LLM papers from arXiv",
    "What are the top AI news stories today?",
    "How is the S&P 500 performing?",
    "What new models did Anthropic release recently?",
    "Show me the Hugging Face papers daily digest",
    "What is Meta's market cap right now?",
    "Any big moves in my stock watchlist today?",
    "Summarize the latest reasoning model papers",
    "What happened in AI this week?",
    "What is the current price of AAPL?",
    "Tell me about the latest GRPO research",
    "Are there any new transformer architecture papers?",
    "What is the NASDAQ trend this month?",
    "Summarize the top ML engineering papers from this week",
    "What is Nvidia's latest GPU announcement?",
    "Any significant earnings reports today?",
    "What new AI tools were released this week?",
    "Tell me about the latest RAG technique papers",
    "What is Google's stock doing today?",
    "Summarize recent multimodal AI research",
    "Any AI startup funding news today?",
    "What papers from DeepSeek should I read?",
    "What is the VIX at right now?",
    "Tell me about the latest fine-tuning research",
    "Is MSFT up or down today?",
    "What were the top AI papers published last week?",
    "Give me a market summary for my watchlist",
]

_CREATIVE: list[str] = [
    "Write a YouTube script about LangGraph multi-agent systems",
    "Create a D2 diagram for a RAG pipeline architecture",
    "Draft a LinkedIn post about my new GRPO fine-tuning project",
    "Write a video script about fine-tuning with Unsloth",
    "Generate a Manim animation script for explaining attention mechanisms",
    "Create a diagram for the Sovereign Edge architecture",
    "Draft a Twitter thread about running LLMs on Jetson",
    "Write a tutorial script on using ONNX for fast inference",
    "Help me write a YouTube intro for my ML channel",
    "Design a D2 diagram for a LangGraph StateGraph",
    "Write a short LinkedIn article about local AI on edge devices",
    "Create a video script explaining LoRA fine-tuning in 5 minutes",
    "Draft a social media post about my RTX 5070 Ti benchmarks",
    "Write an intro hook for a video about TensorRT-LLM",
    "Create a D2 architecture diagram for a FastAPI microservice",
    "Write a LinkedIn post summarizing my AI assistant project",
    "Draft a tweet about the Blackwell GPU architecture",
    "Help me write a narration for my Manim CUDA animation",
    "Write a YouTube script about building RAG with LlamaIndex",
    "Create a diagram showing the Sovereign Edge agent flow",
    "Write a 60-second video script for a TikTok about AI",
    "Draft a LinkedIn carousel post about vector databases",
    "Help me write the hook for a video about GRPO reasoning",
    "Create a D2 flowchart for a CI/CD pipeline",
    "Write a video script comparing vLLM vs llama.cpp",
    "Draft a social post about passing a big ML interview",
    "Generate a diagram for a Telegram bot architecture",
    "Write a YouTube script about quantization techniques",
    "Create a LinkedIn post celebrating a new open-source release",
    "Draft a short Twitter thread about agentic AI patterns",
]

# Augmentation templates — slot {seed} into variations
_AUGMENT_TEMPLATES: dict[str, list[str]] = {
    "spiritual": [
        "{seed}",
        "I want to know: {seed}",
        "Can you help me understand: {seed}",
        "Scripture question: {seed}",
        "For my Bible study — {seed}",
    ],
    "career": [
        "{seed}",
        "I need help with: {seed}",
        "Career question: {seed}",
        "Job search: {seed}",
        "Professional advice needed — {seed}",
    ],
    "intelligence": [
        "{seed}",
        "Market update: {seed}",
        "Research query: {seed}",
        "Quick check — {seed}",
        "Intelligence feed: {seed}",
    ],
    "creative": [
        "{seed}",
        "Content task: {seed}",
        "Creative request: {seed}",
        "Please help me — {seed}",
        "New content idea: {seed}",
    ],
}

_SEEDS: dict[str, list[str]] = {
    "spiritual": _SPIRITUAL,
    "career": _CAREER,
    "intelligence": _INTELLIGENCE,
    "creative": _CREATIVE,
}


def generate_samples(n_per_class: int, seed: int = 42) -> list[dict[str, str | int]]:
    """Generate n_per_class samples per label via seed + augmentation."""
    rng = random.Random(seed)  # noqa: S311 — seeded augmentation for ML training data, not crypto
    samples: list[dict[str, str | int]] = []

    for label_name, label_idx in LABEL_MAP.items():
        seeds = _SEEDS[label_name]
        templates = _AUGMENT_TEMPLATES[label_name]
        count = 0

        # Exhaust seeds first, then augment with templates
        shuffled = seeds.copy()
        rng.shuffle(shuffled)

        while count < n_per_class:
            seed_text = shuffled[count % len(shuffled)]
            template = templates[count % len(templates)]
            text = template.replace("{seed}", seed_text)
            samples.append({"text": text, "label": label_idx, "intent": label_name})
            count += 1

    rng.shuffle(samples)
    return samples


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic router training data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/router_train.jsonl"),
        help="Output JSONL path (default: data/router_train.jsonl)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Samples per class (default: 200, total = samples x 4)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    samples = generate_samples(n_per_class=args.samples, seed=args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    total = len(samples)
    per_class = total // len(LABEL_MAP)
    logger.info(
        "Generated %d samples (%d per class) → %s",
        total,
        per_class,
        args.output,
    )

    # Print class distribution
    from collections import Counter

    counts = Counter(s["intent"] for s in samples)
    for intent, count in sorted(counts.items()):
        logger.info("  %-15s %d", intent, count)


if __name__ == "__main__":
    main()
