"""
campus_data_lookup.py — Custom tool: structured data lookup for CampusAI Lite

Implements the campus_data_lookup tool as both:
  - A LangChain @tool (for use in LangChain/LangGraph/CrewAI)
  - A plain Python function (for direct use in AG2 / BeeAI)

The tool reads from data/campus_data.json which is populated by:
  1. Hand-authored mock data (source: "mock")
  2. Docling-parsed PDF data (source: "docling_pdf")

Query algorithm:
  1. Filter records by category (exact match, case-insensitive)
  2. Score remaining records by keyword overlap with the query string
  3. Return top-N matching records (default N=5)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain.tools import tool

# ── Data path ───────────────────────────────────────────────────────────────
_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "campus_data.json"

# Module-level cache — loaded once
_CAMPUS_DATA: dict[str, list[dict]] | None = None


def _load_data() -> dict[str, list[dict]]:
    """Load (and cache) campus_data.json."""
    global _CAMPUS_DATA
    if _CAMPUS_DATA is None:
        if not _DATA_PATH.exists():
            raise FileNotFoundError(
                f"Campus data file not found: {_DATA_PATH}. "
                "Run the project setup to generate it."
            )
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            _CAMPUS_DATA = json.load(f)
    return _CAMPUS_DATA


def reload_data() -> None:
    """Force a reload of the dataset (e.g., after Docling ingestion)."""
    global _CAMPUS_DATA
    _CAMPUS_DATA = None
    _load_data()


def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens, removing common stop words."""
    _STOP = {"the", "a", "an", "is", "are", "what", "when", "where", "how",
             "for", "of", "in", "on", "at", "to", "and", "or", "my", "me",
             "i", "do", "does", "can", "please", "tell", "give", "show"}
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _STOP


def _score_record(record: dict, query_tokens: set[str]) -> int:
    """Score a record by counting how many query tokens appear in its string representation."""
    record_str = json.dumps(record, ensure_ascii=False).lower()
    return sum(1 for tok in query_tokens if tok in record_str)


def campus_data_lookup_fn(category: str, query: str, top_n: int = 5) -> list[dict]:
    """
    Core lookup function (provider-agnostic).

    Args:
        category: Intent category key (fees, timetable, exam, faculty,
                  library, hostel, canteen, general).
        query:    Natural-language sub-query from the Planner Agent.
        top_n:    Maximum number of records to return.

    Returns:
        List of matching record dicts (may be empty).
    """
    data = _load_data()
    category_lower = category.strip().lower()

    # Flexible category matching
    cat_map = {
        "fee": "fees", "fees": "fees", "tuition": "fees", "payment": "fees",
        "timetable": "timetable", "schedule": "timetable", "class": "timetable", "classes": "timetable",
        "exam": "exam", "examination": "exam", "test": "exam", "exams": "exam",
        "faculty": "faculty", "professor": "faculty", "teacher": "faculty", "staff": "faculty",
        "library": "library", "book": "library", "books": "library", "reading": "library",
        "hostel": "hostel", "dorm": "hostel", "accommodation": "hostel", "mess": "hostel",
        "canteen": "canteen", "food": "canteen", "cafeteria": "canteen", "eat": "canteen",
        "general": "general", "other": "general", "policy": "general", "attendance": "general",
    }
    mapped_category = cat_map.get(category_lower, category_lower)

    # Get candidate records
    candidates: list[dict] = data.get(mapped_category, [])

    # If no exact category match, search across all
    if not candidates:
        candidates = [rec for recs in data.values() for rec in recs]

    if not candidates:
        return []

    # Score and rank
    query_tokens = _tokenize(query)
    if query_tokens:
        scored = [(rec, _score_record(rec, query_tokens)) for rec in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        # Return records with score > 0 first; if none, return first top_n
        top = [rec for rec, score in scored if score > 0][:top_n]
        if not top:
            top = candidates[:top_n]
    else:
        top = candidates[:top_n]

    return top


# ── LangChain @tool wrapper ──────────────────────────────────────────────────
@tool
def campus_data_lookup(category: str, query: str) -> str:
    """
    Look up CampusAI University information from the structured campus database.

    Use this tool to answer questions about:
    - fees: tuition fees, hostel fees, payment deadlines, fee structure
    - timetable: class schedules, room numbers, day-wise timetable
    - exam: exam dates, exam schedules, result declaration dates
    - faculty: faculty names, contact info, office hours, subjects taught
    - library: library hours, borrowing rules, e-resources, fines
    - hostel: hostel facilities, rent, mess timings, rules, Wi-Fi
    - canteen: canteen timings, menu items, smart card info
    - general: attendance policy, academic calendar, placement, IT services

    Args:
        category: One of: fees, timetable, exam, faculty, library, hostel, canteen, general
        query: The specific question or sub-query to look up

    Returns:
        JSON string with matching records, or a "no data found" message.
    """
    import json as _json
    results = campus_data_lookup_fn(category=category, query=query)
    if not results:
        return _json.dumps({
            "found": False,
            "category": category,
            "query": query,
            "message": f"No data found for category='{category}', query='{query}'. This information may not be in the database.",
            "records": []
        }, ensure_ascii=False)
    return _json.dumps({
        "found": True,
        "category": category,
        "query": query,
        "count": len(results),
        "records": results
    }, ensure_ascii=False, indent=2)
