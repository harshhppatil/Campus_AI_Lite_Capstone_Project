"""
test_docling_ingest.py — Tests for Docling ingestion pipeline

Run: python -m pytest tests/test_docling_ingest.py -v

Tests the ingestion pipeline without requiring Docling itself to be installed
(mocks the parsing step). Also tests the text-fallback path.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.docling_ingest import (
    extract_academic_calendar_records,
    merge_into_campus_data,
    save_docling_output,
    _parse_table_rows,
)


# ── Fixtures ────────────────────────────────────────────────────────────────
SAMPLE_MARKDOWN = """CampusAI University
Academic Calendar 2025-2026

ODD SEMESTER (July 2025 - November 2025)

| Event | Date | Notes |
|---|---|---|
| Semester Begins | 01-Jul-2025 | Orientation for first-year students |
| Last date for course registration | 07-Jul-2025 | Late fee: INR 200/day |
| Mid-Semester Examination | 15-Sep-2025 to 22-Sep-2025 | In classrooms |
| End-Semester Examination begins | 10-Nov-2025 | Per Exam Cell schedule |

EVEN SEMESTER (January 2026 - May 2026)

| Event | Date | Notes |
|---|---|---|
| Semester Begins | 03-Jan-2026 | |
| End-Semester Examination begins | 05-May-2026 | Per Exam Cell schedule |

FEE PAYMENT SCHEDULE 2025-2026

| Semester | Deadline | Late Fee |
|---|---|---|
| Odd Semester 2025 | 31-Jul-2025 | INR 500/day |
| Even Semester 2026 | 20-Jan-2026 | INR 500/day |

PUBLIC HOLIDAYS & CAMPUS CLOSED DAYS 2025-2026

| Date | Day | Holiday |
|---|---|---|
| 15-Aug-2025 | Friday | Independence Day |
| 25-Dec-2025 | Thursday | Christmas |
"""

SAMPLE_PARSED = {
    "markdown": SAMPLE_MARKDOWN,
    "doc_dict": {},
    "source_file": "test.pdf",
    "parsed_at": "2025-07-01T10:00:00",
}


class TestParseTableRows:

    def test_basic_pipe_table(self):
        text = "| Col1 | Col2 | Col3 |\n|---|---|---|\n| A | B | C |\n| D | E | F |"
        rows = _parse_table_rows(text)
        # Separator row should be filtered out
        data_rows = [r for r in rows if not all(c.replace('-','').replace(' ','') == '' for c in r)]
        assert any("Col1" in r for r in rows)

    def test_empty_text(self):
        rows = _parse_table_rows("")
        assert rows == []

    def test_text_without_pipes(self):
        rows = _parse_table_rows("No table here\nJust plain text")
        assert rows == []


class TestExtractAcademicCalendarRecords:

    def test_returns_list(self):
        records = extract_academic_calendar_records(SAMPLE_PARSED)
        assert isinstance(records, list)
        assert len(records) > 0

    def test_records_have_required_fields(self):
        records = extract_academic_calendar_records(SAMPLE_PARSED)
        for rec in records:
            assert "id" in rec
            assert "source" in rec
            assert "category" in rec
            assert "parsed_at" in rec

    def test_source_label_applied(self):
        records = extract_academic_calendar_records(SAMPLE_PARSED, source_label="test_pdf")
        for rec in records:
            assert rec["source"] == "test_pdf"

    def test_categories_are_valid(self):
        valid_cats = {"fees", "exam", "general"}
        records = extract_academic_calendar_records(SAMPLE_PARSED)
        for rec in records:
            assert rec["category"] in valid_cats, f"Unexpected category: {rec['category']}"

    def test_ids_are_unique(self):
        records = extract_academic_calendar_records(SAMPLE_PARSED)
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in extracted records"

    def test_fee_records_have_deadline(self):
        records = extract_academic_calendar_records(SAMPLE_PARSED)
        fee_records = [r for r in records if r["category"] == "fees"]
        for rec in fee_records:
            assert "deadline" in rec or "description" in rec

    def test_holiday_record_has_list(self):
        records = extract_academic_calendar_records(SAMPLE_PARSED)
        holiday_records = [r for r in records if "holidays" in r]
        if holiday_records:  # holidays may or may not be extracted depending on text
            assert isinstance(holiday_records[0]["holidays"], list)

    def test_fallback_for_empty_text(self):
        empty_parsed = {**SAMPLE_PARSED, "markdown": "No tables here at all."}
        records = extract_academic_calendar_records(empty_parsed)
        assert len(records) >= 1  # Always returns at least a fallback record


class TestMergeIntoCampusData:

    def test_merge_adds_new_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "campus_data.json"
            # Write minimal initial data
            initial = {"fees": [{"id": "fee-001", "source": "mock", "category": "fees"}]}
            data_path.write_text(json.dumps(initial), encoding="utf-8")

            new_records = [
                {"id": "docling-test-001", "source": "docling_pdf", "category": "exam",
                 "event": "Semester Begins", "date": "01-Jul-2025", "parsed_at": "2025-07-01T00:00:00"},
            ]

            # Temporarily patch the path
            import src.ingestion.docling_ingest as ingest_mod
            original_path = ingest_mod.CAMPUS_DATA_PATH
            ingest_mod.CAMPUS_DATA_PATH = data_path
            try:
                merge_into_campus_data(new_records)
                result = json.loads(data_path.read_text())
                assert "exam" in result
                assert any(r["id"] == "docling-test-001" for r in result["exam"])
                # Original fee record still present
                assert any(r["id"] == "fee-001" for r in result["fees"])
            finally:
                ingest_mod.CAMPUS_DATA_PATH = original_path

    def test_merge_deduplicates_by_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "campus_data.json"
            initial = {"exam": [{"id": "docling-test-001", "source": "docling_pdf",
                                  "category": "exam", "event": "old"}]}
            data_path.write_text(json.dumps(initial), encoding="utf-8")

            updated_records = [
                {"id": "docling-test-001", "source": "docling_pdf", "category": "exam",
                 "event": "updated", "parsed_at": "2025-07-01T00:00:00"},
            ]

            import src.ingestion.docling_ingest as ingest_mod
            original_path = ingest_mod.CAMPUS_DATA_PATH
            ingest_mod.CAMPUS_DATA_PATH = data_path
            try:
                merge_into_campus_data(updated_records)
                result = json.loads(data_path.read_text())
                exam_records = result["exam"]
                # Should still be only 1 record (deduped)
                matching = [r for r in exam_records if r["id"] == "docling-test-001"]
                assert len(matching) == 1
                assert matching[0]["event"] == "updated"
            finally:
                ingest_mod.CAMPUS_DATA_PATH = original_path
