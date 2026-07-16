"""
beeai_poc.py — Part B: BeeAI proof-of-concept for CampusAI Lite

A minimal two-agent BeeAI workflow that demonstrates the framework's design
pattern for the Planner → Responder pipeline, reusing the same tool and dataset.

Scope note: The build spec explicitly calls this a "small proof of concept."
Full three-agent parity with CrewAI is not required. This file demonstrates:
  - BeeAI Agent definition with tools
  - Two-agent pipeline (PlannerAgent + ResponderAgent)
  - Same campus_data_lookup tool reused
  - Direct comparison to CrewAI / AG2 approaches

BeeAI framework (beeai-framework>=0.1.0) uses an async-first design.
This file provides both async and sync-compatible entry points.

Usage:
    python part_b_beeai/beeai_poc.py
    python part_b_beeai/beeai_poc.py --query "What time does the library close?"
    python part_b_beeai/beeai_poc.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tools.campus_data_lookup import campus_data_lookup_fn
from src.schemas import PlannerOutput, ValidationVerdict
from src.agents.validation_agent import FALLBACK_MESSAGE

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


# ── Try importing BeeAI; fall back to direct LLM calls if unavailable ──────
try:
    from beeai_framework.agents.react import ReActAgent
    from beeai_framework.backend.chat import ChatModel
    from beeai_framework.tools.base import Tool, StringToolOutput
    from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
    BEEAI_AVAILABLE = True
except ImportError:
    BEEAI_AVAILABLE = False

# ── Fallback LLM helper (used when BeeAI native calls aren't needed) ────────
def _llm_call(system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
    """Direct LangChain LLM call — used as fallback and for validation."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import SystemMessage, HumanMessage

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])
    return response.content if hasattr(response, "content") else str(response)


def _retry(fn, max_attempts=3, delay=5.0):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if any(k in str(exc).lower() for k in ("429", "rate", "quota", "resource exhausted")):
                time.sleep(delay * attempt)
            else:
                raise
    raise last_exc


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — Native BeeAI ReActAgent (when beeai-framework is available)
# ══════════════════════════════════════════════════════════════════════════════

if BEEAI_AVAILABLE:

    class CampusLookupTool(Tool):
        """BeeAI Tool wrapping campus_data_lookup for use in ReActAgent."""
        name = "campus_data_lookup"
        description = (
            "Look up CampusAI University information. "
            "Inputs: category (fees|timetable|exam|faculty|library|hostel|canteen|general), "
            "query (natural language question)."
        )

        async def _run(self, input: dict, options=None) -> StringToolOutput:
            category = input.get("category", "general")
            query = input.get("query", "")
            records = campus_data_lookup_fn(category=category, query=query)
            result = json.dumps(
                {"found": bool(records), "count": len(records), "records": records},
                indent=2
            )
            return StringToolOutput(result)

    async def run_beeai_native(query: str, verbose: bool = False) -> dict:
        """
        Native BeeAI ReActAgent workflow.
        A single ReActAgent (combining planner + responder reasoning) with
        the campus_data_lookup tool.
        """
        if verbose:
            print(f"\n[beeai-native] Query: {query}")

        model = ChatModel.from_name(
            f"google_genai:{LLM_MODEL}",
            settings={"api_key": GOOGLE_API_KEY, "temperature": 0.2},
        )

        agent = ReActAgent(
            llm=model,
            tools=[CampusLookupTool()],
            memory=UnconstrainedMemory(),
        )

        system_context = (
            "You are CampusAI Assistant for CampusAI University. "
            "Use the campus_data_lookup tool to retrieve accurate information before answering. "
            "Always ground your answer in the tool results. "
            "Classify the query into one of: fees, timetable, exam, faculty, library, hostel, canteen, general. "
            "Then call the tool with that category and the user query."
        )

        from beeai_framework.agents.react.types import ReActAgentInput
        from beeai_framework.context import RunContext

        response = await agent.run(
            ReActAgentInput(prompt=f"{system_context}\n\nStudent query: {query}"),
            run=RunContext(),
        )

        answer = response.result.text if hasattr(response, "result") else str(response)

        if verbose:
            print(f"[beeai-native] Answer: {answer}")

        return {
            "final_answer": answer,
            "intent": "unknown",  # ReActAgent determines this internally
            "sources": [],
            "retry_count": 0,
            "is_valid": True,
            "framework": "BeeAI (native ReActAgent)",
        }


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — BeeAI-pattern workflow (pure Python, framework-pattern demo)
# This implements the same architectural pattern BeeAI promotes:
#   - Agents as typed, composable units
#   - Tool use separated from reasoning
#   - Async-first design
# This works regardless of beeai-framework install status.
# ══════════════════════════════════════════════════════════════════════════════

BEEAI_PLANNER_PROMPT = """You are the PlannerAgent in a BeeAI-style multi-agent system for CampusAI University.

Classify the student query and respond with ONLY a JSON object:
{
  "intent": "<fees|timetable|exam|faculty|library|hostel|canteen|general>",
  "sub_query": "<refined query for tool lookup>",
  "keywords": ["<kw1>", "<kw2>"]
}"""

BEEAI_RESPONDER_PROMPT = """You are the ResponderAgent in a BeeAI-style multi-agent system for CampusAI University.

Draft a clear, helpful answer using ONLY the provided database records.
If records are empty, say you don't have that information.
Be concise but complete."""


class BeeAIStylePlannerAgent:
    """
    BeeAI-pattern Planner Agent.
    Typed inputs/outputs, single responsibility, composable.
    """
    name = "PlannerAgent"
    description = "Classifies student query intent and produces a lookup plan."

    def run(self, query: str) -> PlannerOutput:
        raw = _retry(lambda: _llm_call(BEEAI_PLANNER_PROMPT, f"Student query: {query}", temperature=0.1))
        try:
            data = _extract_json(raw)
            return PlannerOutput(**data)
        except Exception as e:
            # Keyword fallback
            q = query.lower()
            intent = "general"
            for kw, cat in [("fee","fees"),("timetable","timetable"),("exam","exam"),
                            ("faculty","faculty"),("library","library"),("hostel","hostel"),
                            ("canteen","canteen")]:
                if kw in q:
                    intent = cat
                    break
            return PlannerOutput(intent=intent, sub_query=query, reasoning=f"fallback: {e}")  # type: ignore


class BeeAIStyleResponderAgent:
    """
    BeeAI-pattern Responder Agent.
    Takes plan + tool results → drafts answer.
    """
    name = "ResponderAgent"
    description = "Retrieves data via tool and drafts a grounded answer."

    def run(self, query: str, plan: PlannerOutput) -> tuple[str, list[dict]]:
        # Tool call (explicit, not LLM-mediated — BeeAI style)
        sources = campus_data_lookup_fn(category=plan.intent, query=plan.sub_query)
        records_str = json.dumps(sources, indent=2) if sources else "No records found."

        user_msg = (
            f"Student query: {query}\n\n"
            f"Intent: {plan.intent}\n\n"
            f"Database records:\n{records_str}\n\n"
            f"Draft your answer:"
        )
        draft = _retry(lambda: _llm_call(BEEAI_RESPONDER_PROMPT, user_msg, temperature=0.3))
        return draft, sources


class BeeAIPocWorkflow:
    """
    BeeAI proof-of-concept: two-agent pipeline (Planner + Responder).
    Demonstrates BeeAI's architectural pattern:
      - Agents as composable typed units
      - Explicit tool calls (not hidden inside agent reasoning)
      - Async-compatible (run_async method available)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose or VERBOSE
        self.planner = BeeAIStylePlannerAgent()
        self.responder = BeeAIStyleResponderAgent()
        if self.verbose:
            print(f"[beeai] Using {'native BeeAI ReActAgent' if BEEAI_AVAILABLE else 'BeeAI-pattern workflow'}")

    def run(self, query: str) -> dict:
        """Synchronous entry point."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[beeai] Query: {query}")

        # ── PlannerAgent ──────────────────────────────────────────────────
        plan = self.planner.run(query)
        if self.verbose:
            print(f"[beeai][Planner] intent={plan.intent}, sub_query={plan.sub_query}")

        # ── ResponderAgent ────────────────────────────────────────────────
        draft, sources = self.responder.run(query, plan)
        if self.verbose:
            print(f"[beeai][Responder] Draft: {draft[:150]}...")
            print(f"[beeai][Responder] Sources: {len(sources)} record(s)")

        # Note: BeeAI PoC scope is two agents (Planner + Responder).
        # Validation is intentionally omitted per spec ("small proof of concept").
        # The comparison report notes this design choice explicitly.

        return {
            "final_answer": draft,
            "intent": plan.intent,
            "sources": sources,
            "retry_count": 0,
            "is_valid": True,  # No separate validation agent in PoC
            "framework": "BeeAI (pattern PoC)",
            "beeai_native": BEEAI_AVAILABLE,
        }

    async def run_async(self, query: str) -> dict:
        """Async entry point — uses native BeeAI if available, else pattern PoC."""
        if BEEAI_AVAILABLE:
            try:
                return await run_beeai_native(query, verbose=self.verbose)
            except Exception as exc:
                if self.verbose:
                    print(f"[beeai] Native ReActAgent failed ({exc}), falling back to pattern PoC")
        # Run sync version in executor to keep async interface
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, query)


# ── Convenience wrapper ────────────────────────────────────────────────────
_beeai_instance: BeeAIPocWorkflow | None = None


def run_beeai_workflow(query: str, verbose: bool = False) -> dict:
    """Run the BeeAI PoC workflow. Reuses a single instance."""
    global _beeai_instance
    if _beeai_instance is None:
        _beeai_instance = BeeAIPocWorkflow(verbose=verbose)
    return _beeai_instance.run(query)


# ── CLI entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CampusAI Lite — BeeAI PoC")
    parser.add_argument("--query", type=str,
                        default="What are the library borrowing rules?",
                        help="Query to process")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--async-mode", action="store_true",
                        help="Run in async mode (uses native BeeAI if available)")
    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "true"

    print(f"BeeAI native available: {BEEAI_AVAILABLE}")

    if args.async_mode:
        result = asyncio.run(BeeAIPocWorkflow(verbose=args.verbose).run_async(args.query))
    else:
        result = run_beeai_workflow(args.query, verbose=args.verbose)

    print(f"\nAnswer  : {result['final_answer']}")
    print(f"Intent  : {result['intent']}")
    print(f"Sources : {len(result.get('sources', []))} record(s)")
    print(f"Framework: {result['framework']}")
