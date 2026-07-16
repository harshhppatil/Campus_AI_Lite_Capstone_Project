"""
planner_agent.py — Planner Agent for CampusAI Lite

Uses CrewAI for agent/role definition and LangChain for LLM calls.
Produces a PydanticAI-typed PlannerOutput.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from crewai import Agent
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_llm, with_retry, VERBOSE
from src.schemas import IntentType, PlannerOutput

# ── CrewAI Agent definition ────────────────────────────────────────────────
def make_planner_crewai_agent() -> Agent:
    """Return a CrewAI Agent configured as the Planner."""
    return Agent(
        role="University Query Planner",
        goal=(
            "Analyse the student's question, classify it into the correct intent category "
            "(fees, timetable, exam, faculty, library, hostel, canteen, or general), "
            "and produce a structured plan for the Information Agent to execute."
        ),
        backstory=(
            "You are an expert academic coordinator at CampusAI University. "
            "You have deep knowledge of university processes and can instantly "
            "recognise what kind of information a student needs from their question."
        ),
        verbose=VERBOSE,
        allow_delegation=False,
    )


# ── LangChain prompt ───────────────────────────────────────────────────────
_PLANNER_SYSTEM = """You are the Planner Agent for CampusAI University's AI assistant.

Your job is to analyse the student's query and produce a structured JSON plan.

Intent categories and what they cover:
- fees: tuition fees, hostel fees, exam fees, payment deadlines, fee structure
- timetable: class schedules, room numbers, day-wise schedule, lecture timings
- exam: exam dates, mid-sem/end-sem schedules, result dates, supplementary exams
- faculty: faculty names, contact info, office hours, subjects, designations
- library: library hours, borrowing rules, e-resources, fines, digital library
- hostel: hostel facilities, room rent, mess timings, hostel rules, Wi-Fi
- canteen: canteen timings, food items, smart card, cafeteria info
- general: attendance policy, academic calendar, placement, IT services, anti-ragging, bonafide

You MUST respond with ONLY a valid JSON object matching this schema:
{{
  "intent": "<one of: fees | timetable | exam | faculty | library | hostel | canteen | general>",
  "sub_query": "<refined, specific version of the user query for database lookup>",
  "needs_tool": true,
  "tool_name": "campus_data_lookup",
  "reasoning": "<brief explanation of why you chose this intent>",
  "keywords": ["<keyword1>", "<keyword2>", ...]
}}

No markdown, no preamble, no explanation outside the JSON object."""

_PLANNER_HUMAN = "Student query: {query}"

_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _PLANNER_SYSTEM),
    ("human", _PLANNER_HUMAN),
])


# ── Core planning function ─────────────────────────────────────────────────
@with_retry(max_attempts=3, delay_seconds=5.0)
def _call_planner_llm(query: str) -> str:
    """Call the LLM and return raw text response."""
    llm = get_llm(temperature=0.1)
    chain = _PLANNER_PROMPT | llm
    response = chain.invoke({"query": query})
    return response.content if hasattr(response, "content") else str(response)


def _parse_planner_response(raw: str, query: str) -> PlannerOutput:
    """Parse LLM response into PlannerOutput, with fallback defaults."""
    # Strip markdown code fences if present
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
        return PlannerOutput(**data)
    except Exception as e:
        if VERBOSE:
            print(f"[planner] JSON parse failed: {e}. Raw: {raw[:200]}")
        # Fallback: infer intent from keywords
        query_lower = query.lower()
        intent: IntentType = "general"
        for kw, cat in [
            ("fee", "fees"), ("tuition", "fees"), ("payment", "fees"),
            ("timetable", "timetable"), ("schedule", "timetable"), ("class", "timetable"),
            ("exam", "exam"), ("result", "exam"), ("test", "exam"),
            ("faculty", "faculty"), ("professor", "faculty"), ("teacher", "faculty"),
            ("library", "library"), ("book", "library"),
            ("hostel", "hostel"), ("dorm", "hostel"), ("mess", "hostel"),
            ("canteen", "canteen"), ("food", "canteen"), ("cafeteria", "canteen"),
        ]:
            if kw in query_lower:
                intent = cat  # type: ignore
                break
        return PlannerOutput(
            intent=intent,
            sub_query=query,
            needs_tool=True,
            tool_name="campus_data_lookup",
            reasoning=f"Fallback classification (LLM parse error): {e}",
            keywords=[w for w in query.lower().split() if len(w) > 3],
        )


def run_planner(query: str) -> PlannerOutput:
    """
    Run the Planner Agent on a user query.

    Args:
        query: Raw user question.

    Returns:
        PlannerOutput with intent, sub_query, keywords, etc.
    """
    if VERBOSE:
        print(f"\n[planner] Input query: {query}")

    raw = _call_planner_llm(query)

    if VERBOSE:
        print(f"[planner] LLM raw response:\n{raw}")

    result = _parse_planner_response(raw, query)

    if VERBOSE:
        print(f"[planner] PlannerOutput: {result.model_dump()}")

    return result
