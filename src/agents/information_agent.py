"""
information_agent.py — Information Agent for CampusAI Lite

Uses CrewAI Agent + Task + Crew.kickoff() for genuine CrewAI execution.
Invokes the campus_data_lookup tool and drafts a natural-language answer.
"""

from __future__ import annotations

import json

from crewai import Agent, Task, Crew

from src.config import get_llm, with_retry, VERBOSE
from src.schemas import PlannerOutput
from src.tools.campus_data_lookup import campus_data_lookup_fn


# ── CrewAI LLM string ────────────────────────────────────────────────────
def _crewai_llm_str() -> str:
    import os
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    if not model.startswith("gemini/"):
        model = f"gemini/{model}"
    return model


# ── CrewAI Agent definition ────────────────────────────────────────────────
def make_information_crewai_agent() -> Agent:
    """Return a CrewAI Agent configured as the Information Agent."""
    return Agent(
        role="University Information Specialist",
        goal=(
            "Use the campus database to retrieve accurate information and "
            "draft a clear, helpful, and grounded answer to the student's query."
        ),
        backstory=(
            "You are the information desk officer at CampusAI University. "
            "You have access to the complete university database and always "
            "provide accurate, specific answers grounded in official data. "
            "You never make up information — if it's not in the database, you say so."
        ),
        llm=_crewai_llm_str(),
        verbose=VERBOSE,
        allow_delegation=False,
    )


# ── System instruction (kept for reference — embedded in Task description) ─
_INFO_SYSTEM_RULES = """Rules for drafting your answer:
1. Base your answer STRICTLY on the provided source records. Do not add information not in the records.
2. If records contain the answer, give a clear, specific response with relevant details (numbers, dates, names).
3. If records do NOT contain enough information, clearly state: "I don't have specific information about that in my database."
4. Format your answer in plain text — no markdown headers, no bullet points unless listing multiple items.
5. Be concise but complete.
6. Always mention the source (e.g., "According to the university database...")."""

_INFO_PREFIX = _INFO_SYSTEM_RULES  # kept for compatibility with existing tests


# ── CrewAI execution path ─────────────────────────────────────────────────
@with_retry(max_attempts=3, delay_seconds=5.0)
def _run_via_crewai(original_query: str, intent: str, records_str: str) -> str:
    """
    Run the Information Agent through CrewAI Task + Crew.kickoff().
    This is the genuine CrewAI execution path.
    """
    agent = make_information_crewai_agent()

    task_description = (
        f"A student has asked: '{original_query}'\n\n"
        f"Intent category: {intent}\n\n"
        f"Retrieved database records:\n{records_str}\n\n"
        f"{_INFO_SYSTEM_RULES}\n\n"
        f"Draft a helpful, grounded answer to the student's query using only the records above."
    )

    task = Task(
        description=task_description,
        expected_output=(
            "A clear, accurate, plain-text answer to the student's query, "
            "grounded strictly in the provided database records."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=VERBOSE,
    )

    result = crew.kickoff()

    # CrewAI kickoff returns a CrewOutput object; extract string
    if hasattr(result, "raw"):
        return str(result.raw)
    return str(result)


def run_information_agent(
    original_query: str,
    plan: PlannerOutput,
) -> tuple[str, list[dict]]:
    """
    Run the Information Agent via CrewAI Crew.kickoff().

    Args:
        original_query: The user's original question.
        plan: PlannerOutput from the Planner Agent.

    Returns:
        Tuple of (draft_answer: str, sources: list[dict])
    """
    if VERBOSE:
        print(f"\n[information] Query: {original_query}")
        print(f"[information] Plan: intent={plan.intent}, sub_query={plan.sub_query}")

    # ── Step 1: Invoke the tool (direct — tool call is not via CrewAI) ────
    sources = []
    if plan.needs_tool:
        sources = campus_data_lookup_fn(
            category=plan.intent,
            query=plan.sub_query,
        )
        if VERBOSE:
            print(f"[information] Tool returned {len(sources)} record(s)")

    # ── Step 2: Build records string ──────────────────────────────────────
    records_str = (
        json.dumps(sources, indent=2, ensure_ascii=False)
        if sources
        else "No records found in the database for this query."
    )

    # ── Step 3: Draft via CrewAI ──────────────────────────────────────────
    draft = _run_via_crewai(
        original_query=original_query,
        intent=plan.intent,
        records_str=records_str,
    )

    if VERBOSE:
        print(f"[information] Draft answer:\n{draft[:200]}")

    return draft, sources
