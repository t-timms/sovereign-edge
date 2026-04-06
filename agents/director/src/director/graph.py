"""
Director agent — LangGraph-powered multi-expert orchestrator.

Replaces single-shot intent routing with a plan-and-execute graph that can
chain experts when a query spans multiple domains.

Each expert is now itself a LangGraph subgraph.  The director invokes them
via node-wrapper functions (Pattern B / isolated state) so each expert's
internal state remains fully encapsulated.

Examples of multi-expert queries:
  "Research the latest GRPO papers and write a LinkedIn post about them"
  → intelligence (fetch + synthesise) → creative (draft the post)

  "Find ML engineer jobs at companies working on inference optimisation"
  → intelligence (which companies?) → career (job search those companies)

Single-expert queries pass through with zero overhead — the director plan
resolves to one node and exits immediately.

Graph nodes:
  plan      LLM decides the expert chain for this query
  execute   runs the next expert subgraph in the plan
  merge     combines multi-expert outputs into a final response (optional)

Usage:
    from director.graph import DirectorGraph

    graph = DirectorGraph(experts={"intelligence": expert, "creative": expert})
    result = await graph.run(task_request)
"""

from __future__ import annotations

import json
import logging
import re
from typing import (
    Any,
    TypedDict,  # stdlib — always available on Python 3.11+
)

from core.expert import BaseExpert
from core.types import ExpertName, Intent, RoutingDecision, TaskRequest, TaskResult
from llm.gateway import get_gateway
from pydantic import BaseModel, ValidationError, field_validator

try:
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore[assignment,misc]
    END = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── Expert names the director can route to ─────────────────────────────────────
_ROUTABLE_EXPERTS: list[str] = [
    ExpertName.SPIRITUAL,
    ExpertName.CAREER,
    ExpertName.INTELLIGENCE,
    ExpertName.CREATIVE,
]

_DIRECTOR_SYSTEM = """\
You are the Director of Sovereign Edge — a multi-expert personal AI system.
Your job is to analyse the user's request and produce a routing plan.

Available experts:
  spiritual    — Bible, faith, prayer, devotionals, theology
  career       — job search, resume, interviews, salary, LinkedIn
  intelligence — AI/ML research, arXiv papers, tech news, trends
  creative     — content writing, social media posts, scripts, blogs

RULES:
1. Return a JSON object ONLY — no prose, no markdown fences.
2. The "experts" list contains 1-3 expert names in execution order.
3. The "rationale" is one sentence explaining why.
4. Use multiple experts ONLY when the query clearly needs it.
   Single-expert queries are the common case — do not over-engineer.

OUTPUT FORMAT (strict JSON):
{
  "experts": ["<expert1>", "<expert2>"],
  "rationale": "<one sentence>",
  "context_pass": true
}

"context_pass": true means pass the first expert's output as context to the next.
Set to false if the experts are independent (parallel is not yet implemented).

EXAMPLES:
  Input: "What does Psalm 23 mean?"
  Output: {"experts": ["spiritual"], "rationale": "Pure scripture question.", "context_pass": false}

  Input: "Research the latest GRPO papers and write a LinkedIn post about them"
  Output: {"experts": ["intelligence", "creative"],
           "rationale": "Research first, then draft the post using those findings.",
           "context_pass": true}

  Input: "Find ML engineer jobs at companies building inference chips"
  Output: {"experts": ["intelligence", "career"],
           "rationale": "Intelligence identifies target companies, career searches those.",
           "context_pass": true}
"""


# ── Director plan model ───────────────────────────────────────────────────────


class DirectorPlan(BaseModel):
    """Validated routing plan returned by the director LLM.

    The field validator filters out any expert names not in _ROUTABLE_EXPERTS,
    preventing the director from hallucinating expert names that don't exist.
    """

    experts: list[str] = []
    rationale: str = ""
    context_pass: bool = False

    @field_validator("experts")
    @classmethod
    def _filter_routable(cls, v: list[str]) -> list[str]:
        return [s for s in v if s in _ROUTABLE_EXPERTS]


# ── LangGraph state ───────────────────────────────────────────────────────────


class DirectorState(TypedDict):
    request: TaskRequest
    plan: list[str]  # ordered expert names
    context_pass: bool  # carry output between experts
    results: list[str]  # accumulated expert outputs
    final_output: str
    error: str


# ── Director graph ────────────────────────────────────────────────────────────


class DirectorGraph:
    """LangGraph-powered director that plans and executes multi-expert chains.

    Falls back to single-expert dispatch when LangGraph is unavailable or
    when the director LLM call fails — preserving backward compatibility
    with the existing Orchestrator.
    """

    def __init__(self, experts: dict[str, BaseExpert]) -> None:
        self._experts = experts
        self._graph = self._build_graph() if _LANGGRAPH_AVAILABLE else None
        if not _LANGGRAPH_AVAILABLE:
            logger.warning(
                "director_langgraph_unavailable — install langgraph>=1.0 for multi-expert chains"
            )

    def _build_graph(self) -> Any:  # noqa: ANN401
        """Construct the StateGraph. Called once at init."""
        graph = StateGraph(DirectorState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("merge", self._merge_node)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges(
            "execute",
            self._should_continue,
            {"continue": "execute", "merge": "merge", END: END},
        )
        graph.add_edge("merge", END)

        return graph.compile()

    # ── Node implementations ──────────────────────────────────────────────────

    async def _plan_node(self, state: DirectorState) -> dict[str, Any]:
        """Ask the LLM to produce an expert execution plan for this query."""
        request = state["request"]
        gateway = get_gateway()

        try:
            result = await gateway.complete(
                messages=[
                    {"role": "system", "content": _DIRECTOR_SYSTEM},
                    {"role": "user", "content": request.content},
                ],
                max_tokens=200,
                temperature=0.1,
                routing=RoutingDecision.CLOUD,
                expert="director",
            )
            plan_json = _extract_json(result["content"])
            try:
                plan = DirectorPlan.model_validate(plan_json)
                experts = plan.experts
                context_pass = plan.context_pass
            except ValidationError:
                logger.warning("director_plan_validation_failed plan=%r", plan_json)
                experts = []
                context_pass = False

            if not experts:
                experts = [_intent_to_expert(request.intent)]

            logger.info(
                "director_plan experts=%s context_pass=%s query_len=%d",
                experts,
                context_pass,
                len(request.content),
            )
            return {"plan": experts, "context_pass": context_pass, "results": [], "error": ""}

        except Exception:
            logger.warning("director_plan_failed — falling back to intent routing", exc_info=True)
            fallback = _intent_to_expert(request.intent)
            return {"plan": [fallback], "context_pass": False, "results": [], "error": ""}

    async def _execute_node(self, state: DirectorState) -> dict[str, Any]:
        """Execute the next expert in the plan via its subgraph or BaseExpert.process()."""
        plan = list(state["plan"])
        results = list(state.get("results", []))

        if not plan:
            return {"plan": plan, "results": results}

        expert_name = plan.pop(0)
        expert = self._experts.get(expert_name)
        if expert is None:
            logger.warning("director_expert_missing expert=%s", expert_name)
            return {"plan": plan, "results": results}

        # Inject prior expert output as context when context_pass=True
        request = state["request"]
        if state.get("context_pass") and results:
            prior_context = "\n\n---\nPrior expert output:\n" + results[-1]
            enriched_content = request.content + prior_context
            request = request.model_copy(update={"content": enriched_content})

        # Try to invoke the expert's subgraph directly for richer observability
        subgraph = _get_expert_subgraph(expert_name)
        if subgraph is not None:
            try:
                import json as _json

                history: list[dict[str, str]] = []
                if history_json := request.context.get("history"):
                    try:
                        history = _json.loads(history_json)
                    except (ValueError, TypeError):
                        pass

                expert_result = await subgraph.ainvoke(
                    _build_expert_state(expert_name, request, history)
                )
                content = expert_result.get("response", "")
                results.append(content)
                logger.info(
                    "director_execute expert=%s chars=%d via=subgraph",
                    expert_name,
                    len(content),
                )
                return {"plan": plan, "results": results}
            except Exception:
                logger.warning(
                    "director_subgraph_invoke_failed expert=%s — falling back to expert.process()",
                    expert_name,
                    exc_info=True,
                )

        # Fallback: call expert.process() (e.g. LangGraph unavailable in expert package)
        try:
            result = await expert.process(request)
            results.append(result.content)
            logger.info(
                "director_execute expert=%s chars=%d via=process()",
                expert_name,
                len(result.content),
            )
        except Exception:
            logger.error("director_execute_failed expert=%s", expert_name, exc_info=True)
            results.append(f"[{expert_name} unavailable]")

        return {"plan": plan, "results": results}

    async def _merge_node(self, state: DirectorState) -> dict[str, Any]:
        """Merge multi-expert outputs into a coherent final response."""
        results = state.get("results", [])
        if not results:
            return {"final_output": "No results from any expert."}

        if len(results) == 1:
            return {"final_output": results[0]}

        gateway = get_gateway()
        merge_prompt = (
            "You have received outputs from multiple AI experts for a single user request. "
            "Weave them into ONE coherent, well-structured response. "
            "Do not repeat yourself. Keep the Telegram Markdown format (*bold*, _italic_, links).\n\n"  # noqa: E501
            + "\n\n---\n".join(f"Expert output {i + 1}:\n{r}" for i, r in enumerate(results))
        )
        try:
            merged = await gateway.complete(
                messages=[{"role": "user", "content": merge_prompt}],
                max_tokens=2048,
                routing=state["request"].routing,
                expert="director-merge",
            )
            return {"final_output": merged["content"]}
        except Exception:
            logger.warning("director_merge_failed — concatenating outputs", exc_info=True)
            return {"final_output": "\n\n---\n\n".join(results)}

    # ── Routing condition ─────────────────────────────────────────────────────

    @staticmethod
    def _should_continue(state: DirectorState) -> str:
        """Route: continue executing experts, merge when done, or end on error."""
        if state.get("error"):
            return END
        if state.get("plan"):
            return "continue"
        results = state.get("results", [])
        if len(results) > 1:
            return "merge"
        return END

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, request: TaskRequest) -> TaskResult:
        """Execute the director graph and return a TaskResult."""
        if self._graph is None:
            expert_name = _intent_to_expert(request.intent)
            expert = self._experts.get(expert_name) or next(iter(self._experts.values()), None)
            if expert is None:
                return TaskResult(
                    task_id=request.task_id,
                    expert=ExpertName.GENERAL,
                    content="No experts registered.",
                    model_used="none",
                    routing=request.routing,
                )
            return await expert.process(request)

        initial_state: DirectorState = {
            "request": request,
            "plan": [],
            "context_pass": False,
            "results": [],
            "final_output": "",
            "error": "",
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)
        except Exception:
            logger.error("director_graph_failed", exc_info=True)
            expert_name = _intent_to_expert(request.intent)
            expert = self._experts.get(expert_name) or next(iter(self._experts.values()), None)
            if expert:
                return await expert.process(request)
            return TaskResult(
                task_id=request.task_id,
                expert=ExpertName.GENERAL,
                content="Director graph failed and no fallback expert available.",
                model_used="none",
                routing=request.routing,
            )

        content = final_state.get("final_output") or (final_state.get("results", [""])[-1])
        experts_used = ",".join(
            s for s in _ROUTABLE_EXPERTS if any(s in r for r in final_state.get("results", []))
        ) or _intent_to_expert(request.intent)

        return TaskResult(
            task_id=request.task_id,
            expert=ExpertName(experts_used.split(",")[0]) if experts_used else ExpertName.GENERAL,
            content=content,
            model_used="director",
            routing=request.routing,
            metadata={"experts_used": experts_used},
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _intent_to_expert(intent: Intent) -> str:
    """Map Intent enum to expert name string."""
    return {
        Intent.SPIRITUAL: ExpertName.SPIRITUAL,
        Intent.CAREER: ExpertName.CAREER,
        Intent.INTELLIGENCE: ExpertName.INTELLIGENCE,
        Intent.CREATIVE: ExpertName.CREATIVE,
    }.get(intent, ExpertName.INTELLIGENCE)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response string.

    Strips markdown code fences (```json ... ```) that some models emit
    despite being instructed not to, before searching for the JSON object.
    """
    # Strip markdown code fences — handles ```json, ```, and trailing ```
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def _get_expert_subgraph(expert_name: str) -> Any | None:  # noqa: ANN401
    """Return the compiled subgraph for a given expert name, or None.

    Catches all exceptions (not just ImportError) so a NameError or other
    module-level failure in a subgraph doesn't propagate into the director
    execute node — it degrades gracefully to expert.process() instead.
    """
    try:
        if expert_name == ExpertName.INTELLIGENCE:
            from intelligence.subgraph import intelligence_subgraph

            return intelligence_subgraph
        if expert_name == ExpertName.CAREER:
            from career.subgraph import career_subgraph

            return career_subgraph
        if expert_name == ExpertName.SPIRITUAL:
            from spiritual.subgraph import spiritual_subgraph

            return spiritual_subgraph
        if expert_name == ExpertName.CREATIVE:
            from creative.subgraph import creative_subgraph

            return creative_subgraph
    except Exception:
        logger.warning("subgraph_import_failed expert=%s", expert_name, exc_info=True)
    return None


def _build_expert_state(
    expert_name: str,
    request: TaskRequest,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    """Build the initial state dict for an expert subgraph invocation."""
    base: dict[str, Any] = {
        "query": request.content,
        "routing": request.routing,
        "history": history,
        "is_morning_brief": False,
        "response": "",
        "model_used": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
    }
    # Expert-specific intermediate fields
    if expert_name == ExpertName.INTELLIGENCE:
        base.update({"raw_papers": [], "ranked_papers": [], "repo_relevant_papers": []})
    elif expert_name == ExpertName.CAREER:
        base["search_results"] = ""
    elif expert_name == ExpertName.SPIRITUAL:
        base["scripture"] = ""
    elif expert_name == ExpertName.CREATIVE:
        base["trend_context"] = ""
    else:
        logger.warning(
            "director_unknown_expert_state expert=%s — subgraph may KeyError", expert_name
        )
    return base
