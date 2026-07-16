"""
workflow.py — LangGraph StateGraph orchestration for CampusAI Lite

Graph topology:
    START → planner_node → information_node → validation_node
                                  ↑                  |
                                  └── retry (≤2) ────┘  (if invalid)
    validation_node → END  (on valid, or retries exhausted → fallback message)
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START

from src.config import VERBOSE
from src.schemas import PlannerOutput, ValidationVerdict
from src.agents.planner_agent import run_planner
from src.agents.information_agent import run_information_agent
from src.agents.validation_agent import run_validation_agent, FALLBACK_MESSAGE, MAX_RETRIES


# ── Typed state schema for LangGraph ──────────────────────────────────────
class State(TypedDict, total=False):
    query: str
    intent: Optional[str]
    plan: Optional[dict]
    draft_answer: Optional[str]
    sources: list
    is_valid: Optional[bool]
    validation_verdict: Optional[dict]
    final_answer: Optional[str]
    retry_count: int
    error: Optional[str]


# ── Node implementations ────────────────────────────────────────────────────

def planner_node(state: State) -> dict:
    """Planner node: classifies intent and produces a structured plan."""
    query: str = state.get("query", "")
    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"[LangGraph] PLANNER NODE — query: {query}")

    try:
        plan: PlannerOutput = run_planner(query)
        return {
            "intent": plan.intent,
            "plan": plan.model_dump(),
        }
    except Exception as exc:
        return {
            "intent": "general",
            "plan": None,
            "error": f"Planner error: {exc}",
        }


def information_node(state: State) -> dict:
    """Information node: invokes the tool and drafts an answer."""
    query: str = state.get("query", "")
    plan_dict: Optional[dict] = state.get("plan")

    if VERBOSE:
        print(f"\n[LangGraph] INFORMATION NODE — intent={state.get('intent')}, retry={state.get('retry_count', 0)}")

    # Reconstruct PlannerOutput from dict
    if plan_dict:
        try:
            plan = PlannerOutput(**plan_dict)
        except Exception:
            plan = PlannerOutput(intent=state.get("intent", "general"), sub_query=query)  # type: ignore
    else:
        plan = PlannerOutput(intent=state.get("intent", "general"), sub_query=query)  # type: ignore

    try:
        draft_answer, sources = run_information_agent(
            original_query=query,
            plan=plan,
        )
        if VERBOSE:
            print(f"[LangGraph] Draft (first 150): {draft_answer[:150]}")
        return {
            "draft_answer": draft_answer,
            "sources": sources,
        }
    except Exception as exc:
        return {
            "draft_answer": FALLBACK_MESSAGE,
            "sources": [],
            "error": f"Information agent error: {exc}",
        }


def validation_node(state: State) -> dict:
    """Validation node: validates the draft and decides to accept or retry."""
    query: str = state.get("query", "")
    draft: str = state.get("draft_answer", "")
    sources: list = state.get("sources", [])

    if VERBOSE:
        print(f"\n[LangGraph] VALIDATION NODE — retry_count={state.get('retry_count', 0)}")

    try:
        verdict: ValidationVerdict = run_validation_agent(
            original_query=query,
            draft_answer=draft,
            sources=sources,
        )
        if VERBOSE:
            print(f"[LangGraph] Validation: is_valid={verdict.is_valid}")
        return {
            "is_valid": verdict.is_valid,
            "validation_verdict": verdict.model_dump(),
            "final_answer": verdict.final_answer if verdict.is_valid else None,
        }
    except Exception as exc:
        return {
            "is_valid": False,
            "validation_verdict": None,
            "error": f"Validation error: {exc}",
        }


# ── Conditional edge: retry or end ─────────────────────────────────────────
def should_retry(state: State) -> str:
    """
    Returns "retry" → back to information_node  (if invalid AND retries remain)
    Returns "end"   → END                        (if valid OR retries exhausted)
    """
    is_valid: bool = state.get("is_valid", False)
    retry_count: int = state.get("retry_count", 0)

    if is_valid:
        if VERBOSE:
            print("[LangGraph] → VALID → END")
        return "end"

    if retry_count < MAX_RETRIES:
        if VERBOSE:
            print(f"[LangGraph] → INVALID, retry {retry_count + 1}/{MAX_RETRIES}")
        return "retry"

    if VERBOSE:
        print("[LangGraph] → retries exhausted → END with fallback")
    return "end"


# ── Retry counter increment node ───────────────────────────────────────────
def increment_retry(state: State) -> dict:
    """Increment retry counter before going back to information_node."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ── Graph assembly ──────────────────────────────────────────────────────────
def build_graph():
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(State)

    builder.add_node("planner_node", planner_node)
    builder.add_node("information_node", information_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("increment_retry", increment_retry)

    builder.add_edge(START, "planner_node")
    builder.add_edge("planner_node", "information_node")
    builder.add_edge("information_node", "validation_node")
    builder.add_edge("increment_retry", "information_node")

    builder.add_conditional_edges(
        "validation_node",
        should_retry,
        {
            "retry": "increment_retry",
            "end": END,
        }
    )

    graph = builder.compile()
    if VERBOSE:
        print("[LangGraph] Graph compiled successfully.")
    return graph


# ── Module-level cached graph ──────────────────────────────────────────────
_COMPILED_GRAPH = None


def get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


# ── Public entry point ─────────────────────────────────────────────────────
def run_workflow(query: str, verbose: bool = False) -> dict:
    """
    Run the full CampusAI Lite workflow for a given user query.
    Returns final state dict with: final_answer, intent, sources, retry_count, is_valid.
    """
    import os
    if verbose:
        os.environ["VERBOSE"] = "true"

    graph = get_graph()

    initial_state: State = {
        "query": query,
        "retry_count": 0,
        "sources": [],
    }

    if VERBOSE or verbose:
        print(f"\n{'#'*60}\n# Query: {query}\n{'#'*60}")

    final_state = graph.invoke(initial_state)

    # Ensure final_answer is always set
    if not final_state.get("final_answer"):
        verdict_dict = final_state.get("validation_verdict")
        if verdict_dict and isinstance(verdict_dict, dict):
            final_state["final_answer"] = verdict_dict.get("final_answer", FALLBACK_MESSAGE)
        else:
            final_state["final_answer"] = FALLBACK_MESSAGE

    if VERBOSE or verbose:
        print(f"\n{'='*60}\nFINAL ANSWER: {final_state['final_answer']}\n{'='*60}\n")

    return final_state


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the tuition fee for B.Tech Computer Science?"
    result = run_workflow(q, verbose=True)
    print(f"\nAnswer: {result.get('final_answer')}")
    print(f"Intent: {result.get('intent')}")
    print(f"Retries: {result.get('retry_count', 0)}")
