"""
docling_ingest.py — Docling PDF → structured data → merged into campus_data.json

Usage:
    # First, generate the sample PDF (once):
    python data/source_pdfs/create_sample_pdf.py

    # Then run ingestion:
    python src/ingestion/docling_ingest.py

    # Optional: point to a different PDF
    python src/ingestion/docling_ingest.py --pdf path/to/your.pdf

The script:
1. Parses the PDF with Docling (text + tables extracted)
2. Converts relevant sections into structured campus data records
3. Saves intermediate output to data/docling_output/
4. Merges new records into data/campus_data.json (deduplicates by id)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "data" / "source_pdfs" / "academic_calendar_2025_26.pdf"
FALLBACK_TXT = ROOT / "data" / "source_pdfs" / "academic_calendar_2025_26.txt"
DOCLING_OUTPUT_DIR = ROOT / "data" / "docling_output"
CAMPUS_DATA_PATH = ROOT / "data" / "campus_data.json"

DOCLING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Docling helpers ─────────────────────────────────────────────────────────
def parse_with_docling(pdf_path: Path) -> dict:
    """Parse a PDF with Docling and return a dict with text and tables."""
    from docling.document_converter import DocumentConverter

    print(f"[docling_ingest] Parsing PDF: {pdf_path}")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Export as markdown (preserves table structure)
    markdown_text = doc.export_to_markdown()

    # Also export as structured dict if available
    try:
        doc_dict = doc.export_to_dict()
    except Exception:
        doc_dict = {"text": markdown_text}

    return {
        "markdown": markdown_text,
        "doc_dict": doc_dict,
        "source_file": str(pdf_path),
        "parsed_at": datetime.now().isoformat(),
    }


def parse_text_fallback(txt_path: Path) -> dict:
    """Fallback parser for plain-text version when Docling is unavailable."""
    print(f"[docling_ingest] Using text fallback: {txt_path}")
    text = txt_path.read_text(encoding="utf-8")
    return {
        "markdown": text,
        "doc_dict": {"text": text},
        "source_file": str(txt_path),
        "parsed_at": datetime.now().isoformat(),
        "fallback": True,
    }


# ── Structured extraction from parsed text ─────────────────────────────────
def _parse_table_rows(text_block: str) -> list[list[str]]:
    """Parse a markdown/pipe-delimited table into a list of row lists."""
    rows = []
    for line in text_block.splitlines():
        line = line.strip()
        if "|" in line and not re.match(r"^\|[-| ]+\|$", line):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if any(c for c in cells):
                rows.append(cells)
    return rows


def extract_academic_calendar_records(parsed: dict, source_label: str = "docling_pdf") -> list[dict]:
    """
    Convert Docling-parsed academic calendar text into structured records
    that fit the campus_data.json schema.
    """
    text = parsed["markdown"]
    records: list[dict] = []
    now_ts = parsed["parsed_at"]

    # ── Extract ODD semester events ──────────────────────────────────────
    odd_match = re.search(
        r"ODD SEMESTER.*?(?=EVEN SEMESTER|FEE PAYMENT|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if odd_match:
        odd_text = odd_match.group(0)
        rows = _parse_table_rows(odd_text)
        if len(rows) > 1:  # first row is header
            for row in rows[1:]:
                if len(row) >= 2 and row[0] and row[1]:
                    records.append({
                        "id": f"docling-odd-{re.sub(r'[^a-z0-9]', '-', row[0].lower()[:30])}",
                        "source": source_label,
                        "category": "exam",
                        "extracted_from": "Academic Calendar PDF — Odd Semester",
                        "event": row[0].strip(),
                        "date": row[1].strip(),
                        "notes": row[2].strip() if len(row) > 2 else "",
                        "semester": "Odd Semester 2025",
                        "parsed_at": now_ts,
                    })

    # ── Extract EVEN semester events ─────────────────────────────────────
    even_match = re.search(
        r"EVEN SEMESTER.*?(?=FEE PAYMENT|PUBLIC HOLIDAYS|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if even_match:
        even_text = even_match.group(0)
        rows = _parse_table_rows(even_text)
        if len(rows) > 1:
            for row in rows[1:]:
                if len(row) >= 2 and row[0] and row[1]:
                    records.append({
                        "id": f"docling-even-{re.sub(r'[^a-z0-9]', '-', row[0].lower()[:30])}",
                        "source": source_label,
                        "category": "exam",
                        "extracted_from": "Academic Calendar PDF — Even Semester",
                        "event": row[0].strip(),
                        "date": row[1].strip(),
                        "notes": row[2].strip() if len(row) > 2 else "",
                        "semester": "Even Semester 2026",
                        "parsed_at": now_ts,
                    })

    # ── Extract fee payment schedule ─────────────────────────────────────
    fee_match = re.search(
        r"FEE PAYMENT SCHEDULE.*?(?=PUBLIC HOLIDAYS|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if fee_match:
        fee_text = fee_match.group(0)
        rows = _parse_table_rows(fee_text)
        if len(rows) > 1:
            for row in rows[1:]:
                if len(row) >= 2 and row[0] and row[1]:
                    records.append({
                        "id": f"docling-fee-{re.sub(r'[^a-z0-9]', '-', row[0].lower()[:30])}",
                        "source": source_label,
                        "category": "fees",
                        "extracted_from": "Academic Calendar PDF — Fee Payment Schedule",
                        "description": row[0].strip(),
                        "deadline": row[1].strip(),
                        "late_fee": row[2].strip() if len(row) > 2 else "",
                        "parsed_at": now_ts,
                    })

    # ── Extract holidays ─────────────────────────────────────────────────
    holiday_match = re.search(
        r"PUBLIC HOLIDAYS.*?$",
        text, re.IGNORECASE | re.DOTALL
    )
    if holiday_match:
        holiday_text = holiday_match.group(0)
        rows = _parse_table_rows(holiday_text)
        holidays = []
        if len(rows) > 1:
            for row in rows[1:]:
                if len(row) >= 3 and row[0] and row[2]:
                    holidays.append({
                        "date": row[0].strip(),
                        "day": row[1].strip() if len(row) > 1 else "",
                        "holiday": row[2].strip(),
                    })
        if holidays:
            records.append({
                "id": "docling-holidays-2025-26",
                "source": source_label,
                "category": "general",
                "extracted_from": "Academic Calendar PDF — Public Holidays",
                "topic": "Public Holidays & Campus Closed Days 2025-2026",
                "holidays": holidays,
                "parsed_at": now_ts,
            })

    # ── Fallback: add a single summary record if no tables found ─────────
    if not records:
        records.append({
            "id": "docling-academic-calendar-2025-26",
            "source": source_label,
            "category": "general",
            "extracted_from": "Academic Calendar PDF",
            "topic": "Academic Calendar 2025-2026 (raw extract)",
            "content": text[:3000],  # first 3000 chars as a preview
            "parsed_at": now_ts,
            "note": "Structured table extraction failed; raw text stored for reference.",
        })

    print(f"[docling_ingest] Extracted {len(records)} record(s) from PDF")
    return records


# ── Merge into campus_data.json ─────────────────────────────────────────────
def merge_into_campus_data(records: list[dict]) -> None:
    """Merge new records into campus_data.json, deduplicating by 'id'."""
    # Load existing data
    if CAMPUS_DATA_PATH.exists():
        with open(CAMPUS_DATA_PATH, "r", encoding="utf-8") as f:
            campus_data: dict = json.load(f)
    else:
        campus_data = {}

    # Build an id → (category, index) lookup for deduplication
    existing_ids: set[str] = set()
    for cat_records in campus_data.values():
        for rec in cat_records:
            if isinstance(rec, dict) and "id" in rec:
                existing_ids.add(rec["id"])

    added = 0
    updated = 0
    for rec in records:
        category = rec.get("category", "general")
        if category not in campus_data:
            campus_data[category] = []

        if rec["id"] in existing_ids:
            # Update in place
            for i, existing in enumerate(campus_data[category]):
                if isinstance(existing, dict) and existing.get("id") == rec["id"]:
                    campus_data[category][i] = rec
                    updated += 1
                    break
        else:
            campus_data[category].append(rec)
            existing_ids.add(rec["id"])
            added += 1

    # Write back
    with open(CAMPUS_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(campus_data, f, indent=2, ensure_ascii=False)

    print(f"[docling_ingest] Merged into {CAMPUS_DATA_PATH}: {added} added, {updated} updated")


def save_docling_output(parsed: dict, records: list[dict]) -> None:
    """Save intermediate Docling output for inspection."""
    # Save raw markdown text
    md_path = DOCLING_OUTPUT_DIR / "academic_calendar_parsed.md"
    md_path.write_text(parsed["markdown"], encoding="utf-8")

    # Save structured records
    json_path = DOCLING_OUTPUT_DIR / "academic_calendar_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[docling_ingest] Saved raw markdown -> {md_path}")
    print(f"[docling_ingest] Saved structured records -> {json_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def run_ingestion(pdf_path: Path | None = None, force_text_fallback: bool = False) -> list[dict]:
    """
    Full ingestion pipeline:
    1. Parse PDF with Docling (or use text fallback)
    2. Extract structured records
    3. Save to docling_output/
    4. Merge into campus_data.json
    Returns the list of extracted records.
    """
    target_pdf = pdf_path or DEFAULT_PDF
    txt_fallback = target_pdf.with_suffix(".txt") if target_pdf else FALLBACK_TXT

    parsed = None

    # Try text fallback first if forced, or if PDF doesn't exist
    if force_text_fallback or not target_pdf.exists():
        if txt_fallback.exists():
            parsed = parse_text_fallback(txt_fallback)

    # Try Docling only if not already parsed
    if parsed is None and target_pdf.exists() and not force_text_fallback:
        try:
            parsed = parse_with_docling(target_pdf)
        except ImportError:
            print("[docling_ingest] WARNING: Docling not installed.")
        except Exception as exc:
            print(f"[docling_ingest] WARNING: Docling parse failed ({exc}). Using text fallback.")
            if txt_fallback.exists():
                parsed = parse_text_fallback(txt_fallback)

    # Last resort: generate sample files
    if parsed is None:
        print("[docling_ingest] No source found — generating sample files...")
        gen_script = ROOT / "data" / "source_pdfs" / "create_sample_pdf.py"
        if gen_script.exists():
            import subprocess
            subprocess.run([sys.executable, str(gen_script)], check=False)
        if txt_fallback.exists():
            parsed = parse_text_fallback(txt_fallback)
        if parsed is None:
            raise FileNotFoundError(
                f"Could not find source file at {target_pdf} or {txt_fallback}. "
                "Run: python data/source_pdfs/create_sample_pdf.py"
            )

    records = extract_academic_calendar_records(parsed)
    save_docling_output(parsed, records)
    merge_into_campus_data(records)

    print("[docling_ingest] Ingestion complete.")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docling PDF ingestion for CampusAI Lite")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF file")
    parser.add_argument("--text-only", action="store_true", help="Skip Docling, use text fallback directly (faster on low-memory machines)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf) if args.pdf else None
    records = run_ingestion(pdf_path, force_text_fallback=args.text_only)

    print("\n── Sample extracted records ──────────────────────────────────")
    for r in records[:5]:
        print(json.dumps(r, indent=2))
