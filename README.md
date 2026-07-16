# CampusAI Lite 🎓

**Agentic University Information Assistant**

A multi-agent AI assistant that answers university student queries (fees, timetables, exams, faculty, library, hostel, canteen, policies) using a coordinated pipeline of AI agents grounded in structured campus data — including data parsed from real PDFs via Docling.

Built as a capstone project demonstrating all 7 mandatory frameworks: **CrewAI · LangChain · LangGraph · PydanticAI · Docling · AG2 (AutoGen) · BeeAI**

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PART A — CORE PIPELINE                   │
│                    (CrewAI + LangGraph + LangChain)             │
│                                                                 │
│   User Query                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐   PlannerOutput    ┌──────────────────────┐   │
│  │  Planner    │──(PydanticAI)─────▶│   Information Agent  │   │
│  │  Agent      │                    │                      │   │
│  │ (CrewAI)    │                    │  campus_data_lookup  │   │
│  └─────────────┘                    │  ┌────────────────┐  │   │
│                                     │  │ campus_data    │  │   │
│                                     │  │ .json          │  │   │
│                                     │  │ ├── mock data  │  │   │
│                                     │  │ └── Docling    │  │   │
│                                     │  │     PDF data   │  │   │
│                                     │  └────────────────┘  │   │
│                                     └──────────┬───────────┘   │
│                                                │ draft_answer  │
│                                                ▼               │
│                                     ┌──────────────────────┐   │
│                                     │  Validation Agent    │   │
│                                     │  (CrewAI)            │   │
│                                     │                      │   │
│                                     │  ValidationVerdict   │   │
│                                     │  (PydanticAI)        │   │
│                                     └──────────┬───────────┘   │
│                                                │               │
│                            ┌───────────────────┤               │
│                            │ is_valid=False     │ is_valid=True │
│                            │ retry_count < 2    ▼               │
│                            │            Final Answer           │
│                            ▼                                   │
│                   Back to Information Agent                     │
│                   (max 2 retries, then fallback)               │
│                                                                 │
│  LangGraph StateGraph orchestrates all nodes + conditional edge │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐   ┌──────────────────────────────────────┐
│  PART B — AG2        │   │  PART B — BeeAI                      │
│  (AutoGen)           │   │  (Proof of Concept)                  │
│                      │   │                                      │
│  PlannerAgent        │   │  BeeAIStylePlannerAgent              │
│      ↓               │   │         ↓                            │
│  InformationAgent    │   │  BeeAIStyleResponderAgent            │
│      ↓               │   │  (ReActAgent if native available)    │
│  ValidationAgent     │   │                                      │
│  (retry loop)        │   │  Same tool + dataset reused          │
│                      │   │                                      │
│  ConversableAgent    │   │  Demonstrates BeeAI's async-first,  │
│  sequential handoff  │   │  composable agent pattern            │
└──────────────────────┘   └──────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A free Google Gemini API key: https://aistudio.google.com/apikey

### 2. Setup

```bash
# Clone / download the project
cd campusai-lite

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies
pip install langchain langchain-google-genai langgraph pydantic "pydantic-ai>=0.0.14"
pip install crewai docling gradio python-dotenv ag2
pip install "beeai-framework" --prefer-binary   # Windows: add --prefer-binary

# Copy and fill in your API key
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY=your_key_here
```

### 3. Generate sample PDF + run Docling ingestion

```bash
# Generate a realistic academic calendar PDF
python data/source_pdfs/create_sample_pdf.py

# Parse it with Docling and merge into campus_data.json
python src/ingestion/docling_ingest.py
```

You should see output like:
```
[docling_ingest] Parsing PDF: data/source_pdfs/academic_calendar_2025_26.pdf
[docling_ingest] Extracted 23 record(s) from PDF
[docling_ingest] Merged into data/campus_data.json: 23 added, 0 updated
```

### 4. Run the Gradio UI (Part A)

```bash
python src/app.py
# Open http://localhost:7860
```

With verbose agent logging:
```bash
python src/app.py --verbose
```

### 5. Run the LangGraph workflow directly

```bash
python src/graph/workflow.py "What is the tuition fee for B.Tech Computer Science?"
python src/graph/workflow.py "When does the odd semester exam start?" --verbose
```

### 6. Run Part B — AG2 (AutoGen)

```bash
python part_b_autogen/autogen_workflow.py
python part_b_autogen/autogen_workflow.py --query "What are the library hours?" --verbose
```

### 7. Run Part B — BeeAI

```bash
python part_b_beeai/beeai_poc.py
python part_b_beeai/beeai_poc.py --query "Tell me about hostel facilities" --verbose
```

### 8. Run tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
campusai-lite/
├── README.md                          # This file
├── DECISIONS.md                       # Architecture decisions log
├── comparison_report.md               # CrewAI vs AG2 vs BeeAI comparison
├── requirements.txt                   # All dependencies
├── .env.example                       # API key template
├── test_smoke.py                      # Quick pre-LLM smoke tests
│
├── data/
│   ├── campus_data.json               # Unified dataset (mock + Docling-parsed)
│   ├── source_pdfs/
│   │   ├── create_sample_pdf.py       # Generates academic_calendar_2025_26.pdf
│   │   └── academic_calendar_2025_26.pdf  (generated at setup)
│   └── docling_output/
│       ├── academic_calendar_parsed.md    (Docling raw markdown output)
│       └── academic_calendar_records.json (structured extracted records)
│
├── src/
│   ├── config.py                      # LLM setup, env loading, retry decorator
│   ├── schemas.py                     # PydanticAI models: PlannerOutput, ValidationVerdict, GraphState
│   ├── ingestion/
│   │   └── docling_ingest.py          # PDF → structured data → campus_data.json
│   ├── tools/
│   │   └── campus_data_lookup.py      # Custom LangChain tool + plain Python fn
│   ├── agents/
│   │   ├── planner_agent.py           # CrewAI Agent + LangChain LLM → PlannerOutput
│   │   ├── information_agent.py       # CrewAI Agent + tool invocation → draft answer
│   │   └── validation_agent.py        # CrewAI Agent + fact-checking → ValidationVerdict
│   ├── graph/
│   │   └── workflow.py                # LangGraph StateGraph + retry edge
│   └── app.py                         # Gradio chat UI
│
├── part_b_autogen/
│   └── autogen_workflow.py            # AG2 (AutoGen) full pipeline reimplementation
│
├── part_b_beeai/
│   └── beeai_poc.py                   # BeeAI two-agent proof of concept
│
└── tests/
    ├── test_workflow.py               # LangGraph workflow + retry logic tests
    ├── test_docling_ingest.py         # Docling ingestion tests
    └── test_schemas.py                # PydanticAI schema validation tests
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini API key (required) | — |
| `LLM_MODEL` | Model name — swap provider here | `gemini-2.5-flash` |
| `VERBOSE` | Enable verbose agent logging | `false` |

---

## Frameworks Used

| Framework | Version | Role |
|---|---|---|
| **CrewAI** | ≥0.80 | Agent role definitions (role/goal/backstory) |
| **LangChain** | ≥0.3 | LLM wrapper, prompt templates, `@tool` decorator |
| **LangGraph** | ≥0.2 | StateGraph orchestration, retry conditional edge |
| **PydanticAI** | ≥0.0.14 | Typed I/O schemas: `PlannerOutput`, `ValidationVerdict` |
| **Docling** | ≥2.0 | PDF parsing → structured data extraction |
| **AG2 (AutoGen)** | ≥0.3 | Part B: ConversableAgent sequential pipeline |
| **BeeAI** | ≥0.1 | Part B: async ReActAgent / two-agent PoC |

---

## Hallucination Validation Demo

The Validation Agent is designed to catch hallucinated facts. Example:

> **Query**: "What is the tuition fee for B.Tech Computer Science?"
>
> **Information Agent draft** (if hallucinated): "The fee is INR 1,20,000 per semester."
>
> **Validation Agent**: `is_valid=False`, `issues_found=["Fee stated as INR 1,20,000 but source record shows INR 85,000"]`
>
> **After retry**: Corrected answer using the actual record value.

The retry loop (max 2 attempts) ensures the pipeline self-corrects rather than propagating wrong information.

---

## Note on Section 2 vs Section 3 Ambiguity

The original brief's Section 2 ("Mandatory Requirements") lists 7 frameworks. Section 3 ("Tasks") implies Part B is a choice between AG2 *or* BeeAI, and doesn't explicitly task PydanticAI or Docling separately.

**This project implements all 7 frameworks** (the safer interpretation). See [`DECISIONS.md`](DECISIONS.md) — Decision D-001 — for the full rationale.

---

## See Also

- [`DECISIONS.md`](DECISIONS.md) — All architectural decisions with rationale
- [`comparison_report.md`](comparison_report.md) — Three-way framework comparison (CrewAI vs AG2 vs BeeAI)
