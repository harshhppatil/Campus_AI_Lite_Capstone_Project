"""
planner_agent.py — Planner Agent for CampusAI Lite

Uses CrewAI Agent + Task + Crew.kickoff() for genuine CrewAI execution.
Uses pydantic_ai.Agent for schema enforcement on the output — PydanticAI
validates the CrewAI output into a typed PlannerOutput object.

Integration pattern:
    1. CrewAI Crew.kickoff() drives execution and returns raw text.
    2. PydanticAI Agent parses + validates that text into PlannerOutput.
    3. Keyword fallback if both fail.
"""

from __future__ import annotations

import os
from typing import Optional

from crewai import Agent, Task, Crew

from src.config import with_retry, VERBOSE, LLM_MODEL
from src.schemas import IntentType, PlannerOutput


# ── CrewAI LLM string (model name in LiteLLM format for CrewAI) ───────────
def _crewai_llm_str() -> str:
    """
    Return the model name string for CrewAI agents.
    CrewAI 1.x expects llm as a string (LiteLLM format) or dict, not a BaseLLM instance.
    """
    import os
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    # LiteLLM / CrewAI format for Gemini: "gemini/<model_name>"
    if not model.startswith("gemini/"):
        model = f"gemini/{model}"
    return model


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
        llm=_crewai_llm_str(),
        verbose=VERBOSE,
        allow_delegation=False,
    )


# ── Task description for CrewAI ────────────────────────────────────────────
_PLANNER_TASK_TEMPLATE = """Analyse this student query and produce a classification plan.

Student query: {query}

Intent categories:
- fees: tuition fees, hostel fees, exam fees, payment deadlines, fee structure
- timetable: class schedules, room numbers, day-wise schedule, lecture timings
- exam: exam dates, mid-sem/end-sem schedules, result dates, supplementary exams
- faculty: faculty names, contact info, office hours, subjects, designations
- library: library hours, borrowing rules, e-resources, fines, digital library
- hostel: hostel facilities, room rent, mess timings, hostel rules, Wi-Fi
- canteen: canteen timings, food items, smart card, cafeteria info
- general: attendance policy, academic calendar, placement, IT services, anti-ragging

Respond with ONLY a valid JSON object:
{{
  "intent": "<one of the categories above>",
  "sub_query": "<refined, specific version of the query for database lookup>",
  "needs_tool": true,
  "tool_name": "campus_data_lookup",
  "reasoning": "<brief explanation>",
  "keywords": ["<keyword1>", "<keyword2>"]
}}"""

# ── PydanticAI system prompt ────────────────────────────────────────────────
_PYDANTIC_SYSTEM = """You are a query classifier for CampusAI University.
Parse the provided JSON classification output into a structured plan.
Extract intent, sub_query, needs_tool, tool_name, reasoning, and keywords."""


# ── PydanticAI Agent (module-level, initialised lazily) ────────────────────
_pydantic_agent = None


def _get_pydantic_agent():
    """Lazily initialise and cache the PydanticAI Agent."""
    global _pydantic_agent
    if _pydantic_agent is not None:
        return _pydantic_agent

    import pydantic_ai
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(LLM_MODEL, provider=provider)

    _pydantic_agent = pydantic_ai.Agent(
        model=model,
        output_type=PlannerOutput,
        system_prompt=_PYDANTIC_SYSTEM,
        model_settings={"temperature": 0.1},
    )

    if VERBOSE:
        print(f"[planner] PydanticAI Agent initialised with model={LLM_MODEL}")

    return _pydantic_agent


# ── Keyword fallback ────────────────────────────────────────────────────────
def _keyword_fallback(query: str, reason: str) -> PlannerOutput:
    """Infer intent from query keywords when all LLM calls fail."""
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
        reasoning=f"Keyword fallback ({reason})",
        keywords=[w for w in query.lower().split() if len(w) > 3],
    )


# ── Step 1: CrewAI execution ───────────────────────────────────────────────
@with_retry(max_attempts=3, delay_seconds=5.0)
def _run_via_crewai(query: str) -> str:
    """
    Run the Planner Agent through CrewAI Task + Crew.kickoff().
    Returns the raw text output from CrewAI.
    """
    agent = make_planner_crewai_agent()

    task = Task(
        description=_PLANNER_TASK_TEMPLATE.format(query=query),
        expected_output=(
            "A JSON object with intent, sub_query, needs_tool, tool_name, "
            "reasoning, and keywords fields."
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


# ── Step 2: PydanticAI validation of CrewAI output ────────────────────────
def _validate_with_pydantic(raw_text: str) -> PlannerOutput:
    """
    Use PydanticAI Agent to parse and validate the CrewAI output into PlannerOutput.
    PydanticAI enforces the schema at the model-call boundary.
    """
    import asyncio

    agent = _get_pydantic_agent()

    # Feed the raw CrewAI output to PydanticAI for structured extraction
    prompt = f"Parse this classification output into a PlannerOutput:\n\n{raw_text}"

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, agent.run(prompt))
                result = future.result(timeout=60)
        else:
            result = loop.run_until_complete(agent.run(prompt))
    except RuntimeError:
        result = asyncio.run(agent.run(prompt))

    return result.output


def run_planner(query: str) -> PlannerOutput:
    """
    Run the Planner Agent on a user query.

    Pipeline:
        1. CrewAI Crew.kickoff() generates raw classification text.
        2. PydanticAI Agent parses + validates it into PlannerOutput.
        3. Keyword fallback if either step fails.

    Args:
        query: Raw user question.

    Returns:
        PlannerOutput with intent, sub_query, keywords, etc.
    """
    if VERBOSE:
        print(f"\n[planner] Input query: {query}")

    # Step 1: CrewAI execution
    raw_text = None
    try:
        raw_text = _run_via_crewai(query)
        if VERBOSE:
            print(f"[planner] CrewAI raw output: {raw_text[:200]}")
    except Exception as exc:
        if VERBOSE:
            print(f"[planner] CrewAI call failed ({exc}), skipping to PydanticAI direct")

    # Step 2: PydanticAI validation
    # If CrewAI produced output, validate it. Otherwise run PydanticAI directly on query.
    try:
        input_text = raw_text if raw_text else query
        result = _validate_with_pydantic(input_text)
        if VERBOSE:
            print(f"[planner] PydanticAI output: intent={result.intent}")
        return result
    except Exception as exc:
        if VERBOSE:
            print(f"[planner] PydanticAI validation failed ({exc}), using keyword fallback")
        return _keyword_fallback(query, str(exc))
