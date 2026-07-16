"""
schemas.py — PydanticAI typed models for CampusAI Lite

All structured I/O between agents is governed by these models.
PydanticAI enforces the schema at the model-call boundary.

Models:
    PlannerOutput       — Planner Agent's structured analysis of a user query
    ValidationVerdict   — Validation Agent's verdict on a draft answer
    GraphState          — LangGraph shared state (TypedDict + Pydantic-backed)
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

# ── Intent categories (mirrors campus_data.json top-level keys) ────────────
IntentType = Literal["fees", "timetable", "exam", "faculty", "library", "hostel", "canteen", "general"]


# ── Planner Agent output ───────────────────────────────────────────────────
class PlannerOutput(BaseModel):
    """
    Structured plan produced by the Planner Agent.
    The Information Agent consumes this to know what to look up.
    """
    intent: IntentType = Field(
        description="Classified intent category of the user query."
    )
    sub_query: str = Field(
        description="A refined, specific version of the user query for the tool lookup."
    )
    needs_tool: bool = Field(
        default=True,
        description="Whether the campus_data_lookup tool should be invoked."
    )
    tool_name: str = Field(
        default="campus_data_lookup",
        description="Name of the tool to invoke (currently always 'campus_data_lookup')."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this intent was chosen."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms extracted from the query useful for data lookup."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intent": "fees",
                "sub_query": "What is the tuition fee for B.Tech Computer Science semester 1?",
                "needs_tool": True,
                "tool_name": "campus_data_lookup",
                "reasoning": "User is asking about fee structure for a specific program.",
                "keywords": ["B.Tech", "Computer Science", "fee", "semester 1"],
            }
        }
    )


# ── Validation Agent output ───────────────────────────────────────────────
class ValidationVerdict(BaseModel):
    """
    Verdict from the Validation Agent on a draft answer.
    Controls whether the answer is accepted, retried, or fallen back.
    """
    is_valid: bool = Field(
        description="True if the draft answer is factually grounded in the retrieved source data."
    )
    reason: str = Field(
        description="Explanation of why the draft is valid or invalid."
    )
    final_answer: str = Field(
        description="The final answer to present to the user. If valid, this is the (possibly lightly edited) draft. If invalid and retries exhausted, this is a graceful fallback message."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0–1) that the answer is correct and complete."
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="List of specific issues found (hallucinations, missing info, etc.). Empty if valid."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_valid": True,
                "reason": "The answer correctly states the fee amount from the retrieved record.",
                "final_answer": "The tuition fee for B.Tech Computer Science Semester 1 is INR 85,000.",
                "confidence": 0.95,
                "issues_found": [],
            }
        }
    )


# ── LangGraph shared graph state ───────────────────────────────────────────
class GraphState(BaseModel):
    """
    Shared state passed between all LangGraph nodes.
    Each node reads what it needs and writes its outputs back.
    """
    # Input
    query: str = Field(description="Original user query.")

    # Planner output
    intent: Optional[str] = Field(default=None, description="Classified intent.")
    plan: Optional[PlannerOutput] = Field(default=None, description="Full planner output.")

    # Information Agent output
    draft_answer: Optional[str] = Field(default=None, description="Draft answer from Information Agent.")
    sources: list[Any] = Field(default_factory=list, description="Raw data records used to generate the draft.")

    # Validation Agent output
    is_valid: Optional[bool] = Field(default=None, description="Validation verdict.")
    validation_verdict: Optional[ValidationVerdict] = Field(default=None, description="Full validation verdict.")
    final_answer: Optional[str] = Field(default=None, description="Final answer to return to the user.")

    # Control flow
    retry_count: int = Field(default=0, description="Number of retries attempted so far (max 2).")
    error: Optional[str] = Field(default=None, description="Error message if something went wrong.")

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ── Convenience: dict-compatible TypedDict alias for LangGraph ─────────────
from typing import TypedDict, NotRequired


class GraphStateDict(TypedDict):
    """TypedDict version of GraphState for LangGraph StateGraph compatibility."""
    query: str
    intent: NotRequired[Optional[str]]
    plan: NotRequired[Optional[dict]]
    draft_answer: NotRequired[Optional[str]]
    sources: NotRequired[list]
    is_valid: NotRequired[Optional[bool]]
    validation_verdict: NotRequired[Optional[dict]]
    final_answer: NotRequired[Optional[str]]
    retry_count: NotRequired[int]
    error: NotRequired[Optional[str]]
