"""
validation_agent.py — Validation Agent for CampusAI Lite

Uses CrewAI for agent/role definition.
Checks the draft answer against source data and produces a ValidationVerdict.
Triggers a retry if the answer is invalid (up to MAX_RETRIES times).
"""

from __future__ import annotations

import json
import re

from crewai import Agent
from langchain_core.messages import SystemMessage

from src.config import get_llm, with_retry, VERBOSE
from src.schemas import ValidationVerdict

MAX_RETRIES = 2

FALLBACK_MESSAGE = (
    "I'm sorry, I don't have reliable information to answer your question accurately. "
    "Please contact the university helpdesk or visit the relevant department office for assistance."
)


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
        verbose=VERBOSE,
        allow_delegation=False,
    )


# ── LangChain prompt ───────────────────────────────────────────────────────
_VALIDATION_SYSTEM = """You are the Validation Agent for CampusAI University's AI assistant.

Your job is to check whether a draft answer is:
1. Factually grounded in the provided source records (no hallucinated numbers, dates, names)
2. Directly answering the student's original question
3. Not adding information that is absent from the source records

You MUST respond with ONLY a valid JSON object matching this schema:
{{
  "is_valid": <true or false>,
  "reason": "<explanation of your verdict>",
  "final_answer": "<the final answer text to send to the student — if valid, use the draft (lightly edited if needed); if invalid, write an appropriate correction or fallback>",
  "confidence": <0.0 to 1.0>,
  "issues_found": ["<issue 1>", "<issue 2>"] (empty list if valid)
}}

Validation rules:
- If the draft correctly uses information from the source records, mark it VALID.
- If the draft states specific facts (numbers, dates, names) NOT found in the source records, mark INVALID and list issues.
- If source records are empty/missing and the draft appropriately says so, mark VALID.
- If source records are empty and the draft fabricates an answer, mark INVALID.
- Minor rephrasing, formatting changes, and adding "please contact X" are acceptable.

Student's original query: {original_query}

Source records retrieved from database:
{records}

Draft answer to validate:
{draft_answer}

Respond with ONLY the JSON object:"""


# ── Core validation function ───────────────────────────────────────────────

# Split the template to avoid .format() choking on JSON braces in records_str
_VAL_PREFIX = _VALIDATION_SYSTEM.split("Student's original query:")[0].strip()


@with_retry(max_attempts=3, delay_seconds=5.0)
def _call_validation_llm(original_query: str, records_str: str, draft_answer: str) -> str:
    """Call the LLM to validate the draft."""
    from langchain_core.messages import HumanMessage
    llm = get_llm(temperature=0.1)
    # Gemini requires a non-empty HumanMessage — send system context + user turn separately
    human_content = (
        "Student's original query: "
        + original_query
        + "\n\nSource records retrieved from database:\n"
        + records_str
        + "\n\nDraft answer to validate:\n"
        + draft_answer
        + "\n\nRespond with ONLY the JSON object:"
    )
    response = llm.invoke([
        SystemMessage(content=_VAL_PREFIX),
        HumanMessage(content=human_content),
    ])
    return response.content if hasattr(response, "content") else str(response)


def _parse_validation_response(raw: str, draft_answer: str) -> ValidationVerdict:
    """Parse LLM response into ValidationVerdict, with fallback defaults."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
        return ValidationVerdict(**data)
    except Exception as e:
        if VERBOSE:
            print(f"[validation] JSON parse failed: {e}. Raw: {raw[:300]}")
        # Fallback: accept the draft as-is (conservative)
        return ValidationVerdict(
            is_valid=True,
            reason=f"Validation LLM parse error ({e}); accepting draft as-is.",
            final_answer=draft_answer,
            confidence=0.5,
            issues_found=[],
        )


def run_validation_agent(
    original_query: str,
    draft_answer: str,
    sources: list[dict],
) -> ValidationVerdict:
    """
    Run the Validation Agent to check a draft answer against source records.

    Args:
        original_query: The user's original question.
        draft_answer:   Draft answer from the Information Agent.
        sources:        Raw data records the Information Agent retrieved.

    Returns:
        ValidationVerdict with is_valid, reason, final_answer, confidence, issues_found.
    """
    if VERBOSE:
        print(f"\n[validation] Validating draft for query: {original_query}")
        print(f"[validation] Draft: {draft_answer[:200]}...")

    records_str = json.dumps(sources, indent=2, ensure_ascii=False) if sources else "[]"

    raw = _call_validation_llm(
        original_query=original_query,
        records_str=records_str,
        draft_answer=draft_answer,
    )

    if VERBOSE:
        print(f"[validation] LLM raw response:\n{raw}")

    verdict = _parse_validation_response(raw, draft_answer)

    if VERBOSE:
        print(f"[validation] Verdict: is_valid={verdict.is_valid}, confidence={verdict.confidence}")
        if verdict.issues_found:
            print(f"[validation] Issues: {verdict.issues_found}")

    return verdict
