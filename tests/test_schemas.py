"""
test_schemas.py — Tests for PydanticAI schemas (PlannerOutput, ValidationVerdict, GraphState)

Run: python -m pytest tests/test_schemas.py -v
"""

import pytest
from pydantic import ValidationError

from src.schemas import PlannerOutput, ValidationVerdict, GraphStateDict


class TestPlannerOutput:

    def test_valid_all_fields(self):
        p = PlannerOutput(
            intent="fees",
            sub_query="B.Tech CS tuition fee?",
            needs_tool=True,
            tool_name="campus_data_lookup",
            reasoning="User asked about fees",
            keywords=["B.Tech", "CS", "fee"],
        )
        assert p.intent == "fees"
        assert p.needs_tool is True
        assert p.tool_name == "campus_data_lookup"
        assert "B.Tech" in p.keywords

    def test_defaults_applied(self):
        p = PlannerOutput(intent="exam", sub_query="exam dates?")
        assert p.needs_tool is True
        assert p.tool_name == "campus_data_lookup"
        assert p.keywords == []
        assert p.reasoning == ""

    def test_all_valid_intents(self):
        valid_intents = ["fees", "timetable", "exam", "faculty",
                         "library", "hostel", "canteen", "general"]
        for intent in valid_intents:
            p = PlannerOutput(intent=intent, sub_query="test")
            assert p.intent == intent

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValidationError):
            PlannerOutput(intent="unknown_category", sub_query="test")

    def test_serialisation(self):
        p = PlannerOutput(intent="library", sub_query="library hours?", keywords=["library"])
        d = p.model_dump()
        assert d["intent"] == "library"
        assert isinstance(d["keywords"], list)

    def test_roundtrip_from_dict(self):
        data = {"intent": "hostel", "sub_query": "mess timings", "keywords": ["mess"]}
        p = PlannerOutput(**data)
        d = p.model_dump()
        p2 = PlannerOutput(**d)
        assert p2.intent == p.intent


class TestValidationVerdict:

    def test_valid_verdict(self):
        v = ValidationVerdict(
            is_valid=True,
            reason="Correctly states fee",
            final_answer="The fee is INR 85,000.",
            confidence=0.95,
            issues_found=[],
        )
        assert v.is_valid is True
        assert v.confidence == 0.95
        assert v.issues_found == []

    def test_invalid_verdict_with_issues(self):
        v = ValidationVerdict(
            is_valid=False,
            reason="Hallucinated fee amount",
            final_answer="I don't have that information.",
            confidence=0.1,
            issues_found=["Fee stated as INR 1,20,000 but source shows INR 85,000"],
        )
        assert v.is_valid is False
        assert len(v.issues_found) == 1

    def test_defaults(self):
        v = ValidationVerdict(is_valid=True, reason="ok", final_answer="answer")
        assert v.confidence == 1.0
        assert v.issues_found == []

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ValidationVerdict(is_valid=True, reason="ok", final_answer="ok", confidence=1.5)
        with pytest.raises(ValidationError):
            ValidationVerdict(is_valid=True, reason="ok", final_answer="ok", confidence=-0.1)

    def test_serialisation(self):
        v = ValidationVerdict(is_valid=True, reason="ok", final_answer="The fee is 85k.")
        d = v.model_dump()
        assert "is_valid" in d
        assert "final_answer" in d
        assert isinstance(d["issues_found"], list)


class TestGraphStateDict:

    def test_minimal_state(self):
        state: GraphStateDict = {"query": "What is the fee?"}
        assert state["query"] == "What is the fee?"

    def test_full_state(self):
        state: GraphStateDict = {
            "query": "What is the fee?",
            "intent": "fees",
            "plan": {"intent": "fees", "sub_query": "fee?"},
            "draft_answer": "The fee is INR 85,000.",
            "sources": [{"id": "fee-001"}],
            "is_valid": True,
            "validation_verdict": {"is_valid": True, "reason": "ok", "final_answer": "INR 85k"},
            "final_answer": "The fee is INR 85,000.",
            "retry_count": 0,
            "error": None,
        }
        assert state["intent"] == "fees"
        assert state["retry_count"] == 0
        assert state["final_answer"] == "The fee is INR 85,000."
