# Experts

Each expert is a specialized AI agent with its own system prompt, live data sources, and response format. All experts share the same LLM gateway and write to the same trace store.

---

## Intelligence Expert

**Intent:** `INTELLIGENCE`

Synthesizes AI/ML research into actionable briefs. Fetches live papers from arXiv and HuggingFace Daily Papers so every response reflects what published this week, not what was in the training data.

### Live Data Sources

| Source | Endpoint | What it fetches |
|---|---|---|
| arXiv | Atom API (free, no auth) | Latest papers from cs.LG, cs.AI, cs.CL, cs.CV |
| HuggingFace Daily Papers | HF Hub API (free) | Trending community-curated papers |

Papers are de-duplicated across calls via a daily novelty filter — the same paper won't appear in consecutive requests on the same day.

### Capabilities

- Research synthesis: summarize and connect ideas across multiple papers
- Trend spotting: identify emerging methods and architectural patterns
- Concept explanation: break down novel techniques with concrete examples
- Literature context: situate a paper within the broader research landscape

### Morning Brief

Runs at 05:30 Central. Fetches the latest papers from both sources in parallel, builds a digest of the top findings, and pushes it to Telegram. Scoped to 400 tokens to keep it scannable.

### Response Format

Each paper entry:

```
*Paper Title* — _venue/source_
One-sentence summary of the contribution.
Why it matters for practitioners.
[Read more](https://link)
```

---

## Career Expert

**Intent:** `CAREER`

A DFW-focused ML/AI career strategist with live access to job listings. Every response is grounded with real-time search results from Jina so job data, salary ranges, and company intel are current.

### Live Data Sources

| Source | What it fetches |
|---|---|
| Jina Search | LinkedIn and Indeed job listings for ML/AI roles in DFW |
| Jina Search | Company news, salary data, market context |

### Capabilities

- Job search: find and summarize open ML/AI roles in the DFW metro
- Resume coaching: tailor language and keywords to a specific JD
- Interview prep: behavioral and technical question practice
- Market intelligence: DFW AI/ML hiring trends and salary ranges
- Offer evaluation: total compensation analysis

### Target Profile

The expert is calibrated for the user's differentiators:

- GRPO fine-tuning (DeepSeek-R1 method)
- LangGraph multi-agent systems
- MCP server development
- vLLM / TensorRT-LLM production serving
- Structured outputs and LLMOps
- Blackwell GPU (RTX 5070 Ti) hands-on experience

Top DFW target companies: Capital One (Plano), AT&T (Dallas), Sabre (Southlake), American Airlines (Fort Worth), JPMorgan Chase (Plano).

### Morning Brief

Runs at 06:00 and 18:00 Central. Searches for new ML/AI openings in DFW, extracts the highest-value listing, and provides one concrete action to take that day. Scoped to 250 tokens.

---

## Creative Expert

**Intent:** `CREATIVE`

A content strategist and creative director with live trend awareness. Grounds creative output in what is actually working right now in the creator and content economy, not generic advice.

### Live Data Sources

| Source | What it fetches |
|---|---|
| Jina Search | Current content trends, platform conventions, creator economy news |
| Jina Search | Examples of high-performing content in the requested format |

### Capabilities

- Long-form writing: blog posts, essays, articles with a defined voice
- Social media: LinkedIn, Twitter/X, and short-form content tailored to format
- Content strategy: editorial calendar, pillar content, repurposing frameworks
- Brand voice development: articulate and document tone, vocabulary, persona
- Storytelling: narrative structure, hooks, and audience-specific framing

### Morning Brief

Runs at 07:00 Central. Searches for AI content creation and LinkedIn strategy trends, then generates one micro-challenge: a specific, actionable creative exercise completable in 15–20 minutes. Scoped to 150 tokens.

---

## Spiritual Expert

**Intent:** `SPIRITUAL`

A contemplative guide rooted in Christian faith. Every response involving scripture is grounded with live Bible verse retrieval so quotations are always accurate and properly cited.

### Live Data Sources

| Source | Endpoint | What it fetches |
|---|---|---|
| bible-api.com | Free, no auth required | KJV verse by reference or random |

Supports KJV, WEB, YLT, DARBY, ASV, and BBE translations. KJV is the default.

### Capabilities

- Scripture study: exegesis, cross-references, historical context
- Prayer composition: guided or impromptu prayer for specific situations
- Devotionals: short daily reflections anchored to scripture
- Theological questions: Christian doctrine, church history, apologetics
- Faith application: connecting scripture to daily life and decision-making

### Scripture Handling

When the user's message contains a scripture reference (e.g., "John 3:16" or "Psalm 23"), the expert extracts the reference via regex and fetches the exact verse before responding. If no reference is given, it fetches a random verse to anchor the response.

Scripture is always quoted in italics with full citation:
`_"For God so loved the world..."_ — John 3:16 KJV`

### Morning Brief

Runs at 05:15 Central. Fetches a random verse, writes a brief morning devotional (verse + 2–3 sentences of reflection + one-sentence prayer), and pushes it to Telegram. Under 120 words.

---

## PII Routing

All experts respect the routing decision set by the PII detector. If the user's message contains SSN, credit card, email, phone, or IP address patterns, routing is forced to `LOCAL` and no external data sources (Jina, arXiv, HuggingFace, Bible API) are called. The response comes from the local Ollama model only.
