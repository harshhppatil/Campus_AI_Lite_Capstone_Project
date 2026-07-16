"""
information_agent.py — Information Agent for CampusAI Lite

Uses CrewAI for agent/role definition.
Invokes the campus_data_lookup tool and drafts a natural-language answer.
"""

from __future__ import annotations

import json

from crewai import Agent

from src.config import get_llm, with_retry, VERBOSE
from src.schemas import PlannerOutput
from src.tools.campus_data_lookup import campus_data_lookup_fn


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
        verbose=VERBOSE,
        allow_delegation=False,
    )


# ── LangChain prompt ───────────────────────────────────────────────────────
_INFO_SYSTEM = """You are the Information Agent for CampusAI University's AI assistant.

Your job is to draft a helpful, accurate answer to a student's query using ONLY the provided database records.

Rules:
1. Base your answer STRICTLY on the provided source records. Do not add information that is not in the records.
2. If the records contain the answer, give a clear, specific response with relevant details (numbers, dates, names).
3. If the records do NOT contain enough information to answer, clearly state: "I don't have specific information about that in my database."
4. Format your answer in plain text — no markdown headers, no bullet points unless listing multiple items.
5. Be concise but complete. A student reading this should get all the information they need.
6. Always mention the source (e.g., "According to the university fee structure..." or "Based on the campus database...").

Original student query: {original_query}
Intent category: {intent}

Retrieved database records:
{records}

Draft a helpful, grounded answer:"""

# ── Core information retrieval function ───────────────────────────────────

_INFO_PREFIX = _INFO_SYSTEM.split("Original student query:")[0].strip()


@with_retry(max_attempts=3, delay_seconds=5.0)
def _call_information_llm(original_query: str, intent: str, records_str: str) -> str:
    """Call the LLM to generate a draft answer."""
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = get_llm(temperature=0.3)
    # Gemini requires a non-empty HumanMessage — send system context + user turn separately
    human_content = (
        "Original student query: "
        + original_query
        + "\nIntent category: "
        + intent
        + "\n\nRetrieved database records:\n"
        + records_str
        + "\n\nDraft a helpful, grounded answer:"
    )
    response = llm.invoke([
        SystemMessage(content=_INFO_PREFIX),
        HumanMessage(content=human_content),
    ])
    return response.content if hasattr(response, "content") else str(response)


def run_information_agent(
    original_query: str,
    plan: PlannerOutput,
) -> tuple[str, list[dict]]:
    """
    Run the Information Agent.

    Args:
        original_query: The user's original question.
        plan: PlannerOutput from the Planner Agent.

    Returns:
        Tuple of (draft_answer: str, sources: list[dict])
    """
    if VERBOSE:
        print(f"\n[information] Query: {original_query}")
        print(f"[information] Plan: intent={plan.intent}, sub_query={plan.sub_query}")

    # ── Step 1: Invoke the tool ───────────────────────────────────────────
    sources = []
    if plan.needs_tool:
        sources = campus_data_lookup_fn(
            category=plan.intent,
            query=plan.sub_query,
        )
        if VERBOSE:
            print(f"[information] Tool returned {len(sources)} record(s)")

    # ── Step 2: Draft the answer ──────────────────────────────────────────
    if sources:
        records_str = json.dumps(sources, indent=2, ensure_ascii=False)
    else:
        records_str = "No records found in the database for this query."

    draft = _call_information_llm(
        original_query=original_query,
        intent=plan.intent,
        records_str=records_str,
    )

    if VERBOSE:
        print(f"[information] Draft answer:\n{draft}")

    return draft, sources
