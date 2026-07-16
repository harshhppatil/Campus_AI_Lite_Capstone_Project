"""
test_workflow.py — Tests for the LangGraph workflow (tool + graph logic)

These tests cover:
  - The campus_data_lookup tool (no API key needed)
  - Graph state transitions (mocked LLM calls)
  - Retry logic
  - Fallback message

Run: python -m pytest tests/test_workflow.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas import PlannerOutput, ValidationVerdict, GraphStateDict
from src.tools.campus_data_lookup import campus_data_lookup_fn, campus_data_lookup
from src.agents.validation_agent import FALLBACK_MESSAGE, MAX_RETRIES


# ══════════════════════════════════════════════════════════════════════════════
# Tool tests (no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestCampusDataLookup:

    def test_fees_query_returns_records(self):
        results = campus_data_lookup_fn("fees", "B.Tech tuition fee")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_all_categories_return_data(self):
        categories = ["fees", "timetable", "exam", "faculty",
                      "library", "hostel", "canteen", "general"]
        for cat in categories:
            results = campus_data_lookup_fn(cat, cat)
            assert len(results) > 0, f"No records for category: {cat}"

    def test_faculty_query_returns_names(self):
        results = campus_data_lookup_fn("faculty", "Dr. Priya Sharma")
        assert len(results) > 0
        names = [r.get("name", "") for r in results]
        assert any("Priya" in n for n in names)

    def test_keyword_scoring_prioritises_relevant(self):
        results = campus_data_lookup_fn("fees", "MBA fee structure")
        # MBA fee record should rank above B.Tech records
        if len(results) > 0:
            top = results[0]
            assert "fees" in top.get("category", "").lower() or top.get("id", "").startswith("fee")

    def test_top_n_limit_respected(self):
        results = campus_data_lookup_fn("fees", "fee", top_n=2)
        assert len(results) <= 2

    def test_empty_query_returns_records(self):
        # Empty query tokens → return first top_n records
        results = campus_data_lookup_fn("library", "")
        assert len(results) > 0

    def test_unknown_category_falls_back_to_all(self):
        results = campus_data_lookup_fn("nonexistent_category", "placement")
        # Should search across all categories
        assert len(results) > 0

    def test_alias_categories(self):
        # "book" should map to library
        results_book = campus_data_lookup_fn("book", "borrow")
        results_lib = campus_data_lookup_fn("library", "borrow")
        assert len(results_book) > 0
        assert len(results_lib) > 0

    def test_langchain_tool_returns_json_string(self):
        result_str = campus_data_lookup.invoke({"category": "fees", "query": "tuition"})
        assert isinstance(result_str, str)
        data = json.loads(result_str)
        assert "found" in data
        assert "records" in data

    def test_langchain_tool_no_results_message(self):
        # Very specific query that won't match anything
        result_str = campus_data_lookup.invoke({
            "category": "fees",
            "query": "zzzznonexistentzzzz"
        })
        data = json.loads(result_str)
        # May or may not find records, but should always be valid JSON
        assert "found" in data


# ══════════════════════════════════════════════════════════════════════════════
# Graph node tests (mocked LLM)
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphNodes:

    def _make_planner_output(self, intent="fees"):
        return PlannerOutput(
            intent=intent,
            sub_query="B.Tech CS tuition fee?",
            keywords=["B.Tech", "CS", "fee"],
        )

    def _make_validation_verdict(self, is_valid=True):
        return ValidationVerdict(
            is_valid=is_valid,
            reason="Test verdict",
            final_answer="The fee is INR 85,000." if is_valid else FALLBACK_MESSAGE,
            confidence=0.9 if is_valid else 0.1,
        )

    @patch("src.graph.workflow.run_planner")
    def test_planner_node_sets_intent(self, mock_planner):
        from src.graph.workflow import planner_node
        mock_planner.return_value = self._make_planner_output("fees")

        state: GraphStateDict = {"query": "What is the fee?", "retry_count": 0}
        result = planner_node(state)

        assert result["intent"] == "fees"
        assert result["plan"] is not None
        assert result["plan"]["sub_query"] == "B.Tech CS tuition fee?"

    @patch("src.graph.workflow.run_information_agent")
    def test_information_node_sets_draft(self, mock_info):
        from src.graph.workflow import information_node
        mock_info.return_value = ("The fee is INR 85,000.", [{"id": "fee-001"}])

        state: GraphStateDict = {
            "query": "What is the fee?",
            "intent": "fees",
            "plan": {"intent": "fees", "sub_query": "fee?", "needs_tool": True,
                     "tool_name": "campus_data_lookup", "reasoning": "", "keywords": []},
            "retry_count": 0,
        }
        result = information_node(state)

        assert result["draft_answer"] == "The fee is INR 85,000."
        assert len(result["sources"]) == 1

    @patch("src.graph.workflow.run_validation_agent")
    def test_validation_node_valid_sets_final_answer(self, mock_val):
        from src.graph.workflow import validation_node
        mock_val.return_value = self._make_validation_verdict(is_valid=True)

        state: GraphStateDict = {
            "query": "What is the fee?",
            "draft_answer": "The fee is INR 85,000.",
            "sources": [{"id": "fee-001"}],
            "retry_count": 0,
        }
        result = validation_node(state)

        assert result["is_valid"] is True
        assert result["final_answer"] == "The fee is INR 85,000."

    @patch("src.graph.workflow.run_validation_agent")
    def test_validation_node_invalid_no_final_answer(self, mock_val):
        from src.graph.workflow import validation_node
        mock_val.return_value = self._make_validation_verdict(is_valid=False)

        state: GraphStateDict = {
            "query": "What is the fee?",
            "draft_answer": "The fee is INR 1,20,000.",
            "sources": [{"id": "fee-001"}],
            "retry_count": 0,
        }
        result = validation_node(state)

        assert result["is_valid"] is False
        assert result["final_answer"] is None


class TestRetryLogic:

    def test_should_retry_on_invalid_within_limit(self):
        from src.graph.workflow import should_retry
        state: GraphStateDict = {
            "query": "test",
            "is_valid": False,
            "retry_count": 0,
        }
        decision = should_retry(state)
        assert decision == "retry"
        assert state["retry_count"] == 1

    def test_should_end_on_valid(self):
        from src.graph.workflow import should_retry
        state: GraphStateDict = {
            "query": "test",
            "is_valid": True,
            "retry_count": 0,
        }
        decision = should_retry(state)
        assert decision == "end"

    def test_should_end_when_retries_exhausted(self):
        from src.graph.workflow import should_retry
        state: GraphStateDict = {
            "query": "test",
            "is_valid": False,
            "retry_count": MAX_RETRIES,  # already at max
        }
        decision = should_retry(state)
        assert decision == "end"
        assert state["final_answer"] == FALLBACK_MESSAGE

    def test_max_retries_constant_is_2(self):
        assert MAX_RETRIES == 2

    def test_retry_increments_counter(self):
        from src.graph.workflow import should_retry
        state: GraphStateDict = {
            "query": "test",
            "is_valid": False,
            "retry_count": 1,
        }
        decision = should_retry(state)
        assert decision == "retry"
        assert state["retry_count"] == 2
        # One more invalid should now exhaust
        state["is_valid"] = False
        decision2 = should_retry(state)
        assert decision2 == "end"


class TestFallbackMessage:

    def test_fallback_is_graceful(self):
        assert "sorry" in FALLBACK_MESSAGE.lower() or "don't have" in FALLBACK_MESSAGE.lower()
        assert len(FALLBACK_MESSAGE) > 20

    def test_fallback_not_empty(self):
        assert FALLBACK_MESSAGE.strip() != ""
