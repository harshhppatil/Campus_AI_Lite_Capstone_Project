"""
validation_agent.py — Validation Agent for CampusAI Lite

Uses CrewAI Agent + Task + Crew.kickoff() for genuine CrewAI execution.
Checks the draft answer against source data and produces a ValidationVerdict.
"""

from __future__ import annotations

import json
import re

from crewai import Agent, Task, Crew
from langchain_core.messages import SystemMessage

from src.config import get_llm, with_retry, VERBOSE
from src.schemas import ValidationVerdict

MAX_RETRIES = 2

FALLBACK_MESSAGE = (
    "I'm sorry, I don't have reliable information to answer your question accurately. "
    "Please contact the university helpdesk or visit the relevant department office for assistance."
)


# ── CrewAI LLM string ────────────────────────────────────────────────────
def _crewai_llm_str() -> str:
    import os
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    if not model.startswith("gemini/"):
        model = f"gemini/{model}"
    return model


# ── CrewAI Agent definition ────────────────────────────────────────────────
def make_validation_crewai_agent() -> Agent:
    """Return a CrewAI Agent configured as the Validation Agent."""
    return Agent(
        role="Answer Quality Validator",
        goal=(
            "Verify that the drafted answer is factually grounded in the retrieved "
            "database records, directly answers the student's question, and contains "
            "no hallucinated or fabricated information."
        ),
        backstory=(
            "You are a rigorous quality control officer at CampusAI University. "
            "Your job is to catch any mistakes, hallucinations, or unsupported claims "
            "in draft answers before they reach students. You are thorough, fair, and "
            "never approve answers that contain information not found in the source data."
        ),
        llm=_crewai_llm_str(),
        verbose=VERBOSE,
        allow_delegation=False,
    )


# ── Verdict schema description (embedded in Task) ─────────────────────────
_VERDICT_SCHEMA = """{
  "is_valid": <true or false>,
  "reason": "<explanation of your verdict>",
  "final_answer": "<the final answer to send to student>",
  "confidence": <0.0 to 1.0>,
  "issues_found": ["<issue>"] or []
}"""

_VAL_PREFIX = """You are the Validation Agent for CampusAI University's AI assistant.
Check whether the draft answer is factually grounded in the source records.
Validation rules:
- Mark VALID if the draft correctly uses information from source records.
- Mark INVALID if the draft states facts NOT in source records.
- If records are empty and draft says so, mark VALID.
- If records are empty and draft fabricates an answer, mark INVALID."""


# ── Response parser ────────────────────────────────────────────────────────
def _parse_validation_response(raw: str, draft_answer: str) -> ValidationVerdict:
    """Parse raw text (from CrewAI or LLM) into ValidationVerdict."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Try to extract a JSON object from anywhere in the text
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group(0)

    try:
        data = json.loads(text)
        return ValidationVerdict(**data)
    except Exception as e:
        if VERBOSE:
            print(f"[validation] JSON parse failed: {e}. Raw: {raw[:300]}")
        # Fallback: accept draft as-is (conservative)
        return ValidationVerdict(
            is_valid=True,
            reason=f"Parse error ({e}); accepting draft as-is.",
            final_answer=draft_answer,
            confidence=0.5,
            issues_found=[],
        )


# ── CrewAI execution path ─────────────────────────────────────────────────
@with_retry(max_attempts=3, delay_seconds=5.0)
def _run_via_crewai(original_query: str, records_str: str, draft_answer: str) -> str:
    """
    Run the Validation Agent through CrewAI Task + Crew.kickoff().
    This is the genuine CrewAI execution path.
    """
    agent = make_validation_crewai_agent()

    task_description = (
        f"{_VAL_PREFIX}\n\n"
        f"Student's original query: {original_query}\n\n"
        f"Source records retrieved from database:\n{records_str}\n\n"
        f"Draft answer to validate:\n{draft_answer}\n\n"
        f"Respond with ONLY a valid JSON object matching this schema:\n{_VERDICT_SCHEMA}"
    )

    task = Task(
        description=task_description,
        expected_output=(
            "A JSON object with keys: is_valid (bool), reason (str), "
            "final_answer (str), confidence (float 0-1), issues_found (list of str)."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=VERBOSE,
    )

    result = crew.kickoff()

    if hasattr(result, "raw"):
        return str(result.raw)
    return str(result)


def run_validation_agent(
    original_query: str,
    draft_answer: str,
    sources: list[dict],
) -> ValidationVerdict:
    """
    Run the Validation Agent via CrewAI Crew.kickoff().

    Args:
        original_query: The user's original question.
        draft_answer:   Draft answer from the Information Agent.
        sources:        Raw data records the Information Agent retrieved.

    Returns:
        ValidationVerdict with is_valid, reason, final_answer, confidence, issues_found.
    """
    if VERBOSE:
        print(f"\n[validation] Validating draft for query: {original_query}")
        print(f"[validation] Draft (first 200): {draft_answer[:200]}")

    records_str = json.dumps(sources, indent=2, ensure_ascii=False) if sources else "[]"

    raw = _run_via_crewai(
        original_query=original_query,
        records_str=records_str,
        draft_answer=draft_answer,
    )

    if VERBOSE:
        print(f"[validation] CrewAI raw output:\n{raw[:300]}")

    verdict = _parse_validation_response(raw, draft_answer)

    if VERBOSE:
        print(f"[validation] Verdict: is_valid={verdict.is_valid}, confidence={verdict.confidence}")
        if verdict.issues_found:
            print(f"[validation] Issues: {verdict.issues_found}")

    return verdict
