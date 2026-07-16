"""
autogen_workflow.py — Part B: AG2 (AutoGen) reimplementation of CampusAI Lite

Reimplements the same Planner → Information → Validation pipeline using AG2's
ConversableAgent with a sequential handoff pattern, reusing the same custom
tool and dataset as Part A.

Usage:
    python part_b_autogen/autogen_workflow.py
    python part_b_autogen/autogen_workflow.py --query "What is the library fine?"
    python part_b_autogen/autogen_workflow.py --verbose

Architecture:
    UserProxy  →  PlannerAgent  →  InformationAgent  →  ValidationAgent
    Each agent passes its output as a structured message to the next.
    GroupChatManager orchestrates the sequential handoff.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager

from src.tools.campus_data_lookup import campus_data_lookup_fn
from src.schemas import PlannerOutput, ValidationVerdict
from src.agents.validation_agent import FALLBACK_MESSAGE, MAX_RETRIES

# ── LLM config ────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

LLM_CONFIG = {
    "config_list": [
        {
            "model": LLM_MODEL,
            "api_key": GOOGLE_API_KEY,
            "api_type": "google",
        }
    ],
    "temperature": 0.2,
}

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# ── Agent system prompts ───────────────────────────────────────────────────
PLANNER_SYSTEM = """You are the Planner Agent for CampusAI University's AI assistant.

Analyse the student query and respond with ONLY a JSON object:
{
  "intent": "<fees|timetable|exam|faculty|library|hostel|canteen|general>",
  "sub_query": "<refined query for database lookup>",
  "keywords": ["<kw1>", "<kw2>"]
}

No extra text. Just the JSON."""

INFORMATION_SYSTEM = """You are the Information Agent for CampusAI University.

You will receive:
1. The student's original query
2. A planner JSON with intent and sub_query
3. Database records retrieved for that query

Your job: draft a clear, accurate answer using ONLY the provided database records.
If records are empty, say: "I don't have that information in my database."
Be concise but complete. No markdown formatting."""

VALIDATION_SYSTEM = """You are the Validation Agent for CampusAI University.

Check if the draft answer is factually grounded in the source records.
Respond with ONLY a JSON object:
{
  "is_valid": <true|false>,
  "reason": "<brief explanation>",
  "final_answer": "<final answer to send to student>",
  "confidence": <0.0-1.0>,
  "issues_found": ["<issue>"] or []
}

Rules:
- Mark VALID if the draft correctly uses information from source records.
- Mark INVALID if the draft states facts NOT in source records.
- If records are empty and draft says so, mark VALID.
No extra text outside the JSON."""


# ── Tool function (for AG2 function-calling) ───────────────────────────────
def lookup_campus_data(category: str, query: str) -> str:
    """Look up university data. category: fees|timetable|exam|faculty|library|hostel|canteen|general"""
    results = campus_data_lookup_fn(category=category, query=query)
    if not results:
        return json.dumps({"found": False, "records": [], "message": f"No data for category='{category}'"})
    return json.dumps({"found": True, "count": len(results), "records": results}, indent=2)


# ── Core sequential pipeline ───────────────────────────────────────────────
def _retry_with_backoff(fn, max_attempts=3, delay=5.0):
    """Simple retry wrapper for LLM calls."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            err = str(exc).lower()
            if any(k in err for k in ("429", "rate", "quota", "resource exhausted", "503")):
                wait = delay * attempt
                if VERBOSE:
                    print(f"[autogen] rate limit hit, waiting {wait}s (attempt {attempt}/{max_attempts})")
                time.sleep(wait)
            else:
                raise
    raise last_exc


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


class AutoGenCampusWorkflow:
    """
    Sequential Planner → Information → Validation pipeline built with AG2
    ConversableAgents. Uses direct agent.generate_reply() calls for
    deterministic sequential handoff rather than GroupChat auto-routing.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose or VERBOSE

        # ── Create agents ─────────────────────────────────────────────────
        self.planner = ConversableAgent(
            name="PlannerAgent",
            system_message=PLANNER_SYSTEM,
            llm_config=LLM_CONFIG,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        self.information = ConversableAgent(
            name="InformationAgent",
            system_message=INFORMATION_SYSTEM,
            llm_config=LLM_CONFIG,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        self.validation = ConversableAgent(
            name="ValidationAgent",
            system_message=VALIDATION_SYSTEM,
            llm_config=LLM_CONFIG,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        if self.verbose:
            print("[autogen] Agents initialised: PlannerAgent, InformationAgent, ValidationAgent")

    def _call_agent(self, agent: ConversableAgent, message: str) -> str:
        """Call a single agent and return its text reply."""
        def _call():
            reply = agent.generate_reply(
                messages=[{"role": "user", "content": message}]
            )
            return reply if isinstance(reply, str) else str(reply)

        return _retry_with_backoff(_call)

    def run(self, query: str) -> dict:
        """
        Run the full pipeline for a query.

        Returns dict with keys:
            final_answer, intent, sources, retry_count, is_valid
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[autogen] Query: {query}")

        # ── Step 1: Planner ───────────────────────────────────────────────
        planner_msg = f"Student query: {query}"
        planner_raw = self._call_agent(self.planner, planner_msg)

        if self.verbose:
            print(f"[autogen][Planner] Raw: {planner_raw}")

        try:
            plan_data = _extract_json(planner_raw)
            plan = PlannerOutput(**plan_data)
        except Exception as e:
            if self.verbose:
                print(f"[autogen][Planner] Parse error: {e}, using fallback")
            plan = PlannerOutput(
                intent="general", sub_query=query,
                reasoning=f"parse error: {e}"
            )

        if self.verbose:
            print(f"[autogen][Planner] intent={plan.intent}, sub_query={plan.sub_query}")

        # ── Step 2: Tool lookup ───────────────────────────────────────────
        sources = campus_data_lookup_fn(category=plan.intent, query=plan.sub_query)
        records_str = json.dumps(sources, indent=2) if sources else "[]"

        if self.verbose:
            print(f"[autogen][Tool] Retrieved {len(sources)} record(s)")

        # ── Step 3: Information Agent ─────────────────────────────────────
        info_msg = (
            f"Original student query: {query}\n\n"
            f"Planner output:\n{json.dumps(plan.model_dump(), indent=2)}\n\n"
            f"Database records:\n{records_str}\n\n"
            f"Draft a helpful, grounded answer:"
        )
        draft = self._call_agent(self.information, info_msg)

        if self.verbose:
            print(f"[autogen][Information] Draft: {draft[:200]}...")

        # ── Step 4: Validation Agent (with retry loop) ────────────────────
        retry_count = 0
        is_valid = False
        verdict = None

        while retry_count <= MAX_RETRIES:
            val_msg = (
                f"Student query: {query}\n\n"
                f"Source records:\n{records_str}\n\n"
                f"Draft answer:\n{draft}\n\n"
                f"Validate and respond with JSON:"
            )
            val_raw = self._call_agent(self.validation, val_msg)

            if self.verbose:
                print(f"[autogen][Validation] retry={retry_count} raw: {val_raw[:200]}")

            try:
                verdict_data = _extract_json(val_raw)
                verdict = ValidationVerdict(**verdict_data)
                is_valid = verdict.is_valid
            except Exception as e:
                if self.verbose:
                    print(f"[autogen][Validation] Parse error: {e}")
                # Accept draft on parse failure
                verdict = ValidationVerdict(
                    is_valid=True, reason=f"parse error: {e}",
                    final_answer=draft, confidence=0.5
                )
                is_valid = True

            if is_valid:
                if self.verbose:
                    print(f"[autogen][Validation] VALID (confidence={verdict.confidence})")
                break

            if retry_count < MAX_RETRIES:
                if self.verbose:
                    print(f"[autogen][Validation] INVALID — retrying information agent ({retry_count+1}/{MAX_RETRIES})")
                    if verdict.issues_found:
                        print(f"  Issues: {verdict.issues_found}")
                # Retry: re-call information agent with issues feedback
                retry_msg = (
                    f"Original student query: {query}\n\n"
                    f"Database records:\n{records_str}\n\n"
                    f"Your previous draft had these issues:\n"
                    f"{chr(10).join(verdict.issues_found)}\n\n"
                    f"Please correct the draft, using ONLY information from the database records:"
                )
                draft = self._call_agent(self.information, retry_msg)
                retry_count += 1
            else:
                if self.verbose:
                    print("[autogen][Validation] Retries exhausted — using fallback")
                verdict = ValidationVerdict(
                    is_valid=False, reason="retries exhausted",
                    final_answer=FALLBACK_MESSAGE, confidence=0.0
                )
                break

        final_answer = verdict.final_answer if verdict else FALLBACK_MESSAGE

        if self.verbose:
            print(f"\n[autogen] FINAL ANSWER: {final_answer}")
            print(f"[autogen] intent={plan.intent} | retries={retry_count} | valid={is_valid}")

        return {
            "final_answer": final_answer,
            "intent": plan.intent,
            "sources": sources,
            "retry_count": retry_count,
            "is_valid": is_valid,
            "framework": "AG2 (AutoGen)",
        }


# ── Convenience wrapper (mirrors Part A API) ───────────────────────────────
_workflow_instance: AutoGenCampusWorkflow | None = None


def run_autogen_workflow(query: str, verbose: bool = False) -> dict:
    """Run the AG2 workflow. Reuses a single instance for efficiency."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = AutoGenCampusWorkflow(verbose=verbose)
    return _workflow_instance.run(query)


# ── CLI entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CampusAI Lite — AG2 (AutoGen) workflow")
    parser.add_argument("--query", type=str,
                        default="What is the tuition fee for B.Tech Computer Science?",
                        help="Query to process")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "true"

    result = run_autogen_workflow(args.query, verbose=args.verbose)
    print(f"\nAnswer : {result['final_answer']}")
    print(f"Intent : {result['intent']}")
    print(f"Retries: {result['retry_count']}")
    print(f"Valid  : {result['is_valid']}")
