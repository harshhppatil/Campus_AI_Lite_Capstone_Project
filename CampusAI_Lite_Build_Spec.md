# CampusAI Lite — Agentic University Information Assistant
### Build Specification for AI Coding Agents — Revision 2 (all 7 frameworks in scope)

**Instruction to the AI agent reading this file:** you are building this entire project autonomously. Read this document fully before writing any code. Then execute Section 6 ("Build Order") top to bottom, one step at a time, without stopping to ask the user what to do next — everything you need to know (architecture, tech choices, file structure, API key setup, what "done" looks like) is already specified below. Only pause and ask the user a question if you hit something genuinely undecidable from this document (e.g. a missing API key, a package that fails to install). After each numbered build step, briefly report what you did before moving to the next one, so the user can follow along — but keep going through the full list rather than waiting for a go-ahead each time. Where a decision is left open, pick the most standard, well-documented option and record the choice in `DECISIONS.md` rather than asking.

---

## 0. Why this revision exists — read this first

The original brief contradicts itself, and this revision resolves it in the safer direction.

- **Section 2 ("Mandatory Requirements")** lists seven frameworks with no qualifiers: CrewAI, LangChain, LangGraph, AG2 (AutoGen), PydanticAI, Docling, BeeAI.
- **Section 3 ("Tasks")** then narrows this: Part A only asks for CrewAI, LangChain, LangGraph. Part B says to explore "Option 1" (AG2) *or* "Option 2" (BeeAI) — phrased as a choice. PydanticAI and Docling aren't mentioned in the task breakdown at all.

Since a grader is more likely to check submissions against the "mandatory requirements" list than to remember the softer wording buried in the task section, **this spec now treats all seven frameworks as required**:

| Framework | Where it lands |
|---|---|
| CrewAI | Part A — agent roles |
| LangChain | Part A — LLM wrapper, prompts, tool interface |
| LangGraph | Part A — orchestration |
| **PydanticAI** | **Folded into Part A core** — typed schemas for agent I/O |
| **Docling** | **Folded into Part A core** — real-PDF ingestion feeding the custom tool |
| **AG2 (AutoGen)** | **Part B — both options required**, not a choice |
| **BeeAI** | **Part B — both options required**, not a choice |

**Also send your instructor/TA a one-line message flagging the section 2 vs. section 3 mismatch.** Building everything protects your grade either way, but a paper trail showing you caught the ambiguity and asked is extra insurance if grading turns out to follow the narrower reading.

---

## 1. Project Summary

**CampusAI Lite** is a multi-agent AI assistant that answers university-related student queries (timetables, fees, exam schedules, faculty info, library rules, hostel/canteen info, etc.) using a coordinated pipeline of agents, a custom tool grounded in both mock and real (PDF-derived) data, and a Gradio chat UI — with the same workflow rebuilt in two additional frameworks for comparison.

---

## 2. Agent Architecture (Part A)

Three agents, run as a pipeline (Planner → Information → Validation), orchestrated with **LangGraph** as the state machine and **CrewAI** for agent/role definitions. LangChain provides LLM wrappers, prompt templates, and the tool interface. **PydanticAI** governs the typed contracts between agents (see section 2.5).

### 2.1 Planner Agent
- Input: raw user query.
- Job: classify intent (e.g. `fees`, `timetable`, `exam`, `faculty`, `library`, `hostel`, `general`), decide which tool(s)/data source(s) are needed, and produce a structured plan.
- Output: a `PlannerOutput` object (PydanticAI model) — e.g. `intent`, `sub_query`, `needs_tool`, `tool_name`.

### 2.2 Information Agent
- Input: the Planner's plan.
- Job: executes the custom tool and/or retrieves relevant info, drafts a natural-language answer.
- Output: draft answer + the raw data/citations used.

### 2.3 Validation Agent
- Input: draft answer + original query + source data.
- Job: checks the draft is factually grounded in the retrieved data, not hallucinated, and directly answers the question. If it fails validation, sends it back to the Information Agent (loop, max 2 retries) or returns a graceful "I don't have that information" fallback.
- Output: a `ValidationVerdict` object (PydanticAI model) — `is_valid`, `reason`, `final_answer`.

### 2.4 State graph (LangGraph)
```
START → planner_node → information_node → validation_node
                              ↑                  |
                              └── retry (≤2) ─────┘ (if invalid)
validation_node → END (on valid, or retries exhausted → fallback message)
```
Use a `TypedDict` (or a PydanticAI-backed model) for shared graph state: `query, intent, plan, draft_answer, sources, is_valid, retry_count, final_answer`.

### 2.5 PydanticAI — core, not optional
Define every structured agent input/output as a PydanticAI model rather than a loose dict:
- `PlannerOutput` (Planner's plan)
- `ValidationVerdict` (Validation Agent's verdict)
- Optionally wrap the LLM calls themselves in PydanticAI's typed-agent pattern (`Agent[DepsType, OutputType]`) so the framework enforces the schema at the model-call boundary, not just after parsing.

This directly satisfies the "mandatory" PydanticAI requirement and also makes the LangGraph state transitions safer — a genuine engineering win, not just a checkbox.

---

## 3. Custom Tool & Data (Docling folded in as core)

Build **one** well-defined custom tool (LangChain `@tool` or CrewAI `BaseTool`): **`campus_data_lookup`**:
- Reads from a local structured dataset (JSON/SQLite) — mock but realistic university data: departments, faculty, timetables, fee structures, exam dates, library hours.
- Input: `(category: str, query: str)`.
- Output: matching record(s) as structured data, not free text — this is what the Validation Agent grounds against.

**Docling ingestion (core requirement, not a bonus):**
- Take at least one real or realistic PDF — an academic calendar, a fee circular, an exam-schedule notice — and use **Docling** to parse it into structured text/tables at setup time.
- Feed the parsed output into the same dataset `campus_data_lookup` reads from, so the tool is genuinely grounded in two sources: hand-authored mock JSON and a Docling-parsed PDF.
- Keep the ingestion as a standalone script (`src/ingestion/docling_ingest.py`) that's easy to demo independently — run it, show the structured output, show it landing in `campus_data.json`.

---

## 4. Tech Stack

| Layer | Choice | Status |
|---|---|---|
| Orchestration/state machine | LangGraph | Part A — core |
| Agent roles & crew definition | CrewAI | Part A — core |
| LLM wrapper, prompts, tool interface | LangChain | Part A — core |
| Typed agent I/O schemas | **PydanticAI** | **Part A — core (folded in)** |
| PDF ingestion for the custom tool | **Docling** | **Part A — core (folded in)** |
| LLM provider | Google Gemini API (free tier, e.g. `gemini-2.5-flash`) via `langchain-google-genai` — keep it swappable via `.env` so any provider can be dropped in later | — |
| UI | Gradio (`gr.ChatInterface`) | Part A — core |
| Data store | local JSON/SQLite mock + Docling-parsed data | Part A — core |
| Framework comparison A | **AG2 (AutoGen)** | **Part B — required (not optional)** |
| Framework comparison B | **BeeAI** | **Part B — required (not optional)** |
| Env/config | `python-dotenv`, `.env` for API keys | — |

---

## 5. Repository Structure

```
campusai-lite/
├── README.md
├── DECISIONS.md
├── requirements.txt
├── .env.example
├── data/
│   ├── campus_data.json          # mock structured university data
│   ├── source_pdfs/               # real PDF(s) for Docling ingestion
│   └── docling_output/            # parsed structured output, merged into campus_data.json
├── src/
│   ├── config.py                  # LLM client setup, env loading
│   ├── schemas.py                  # PydanticAI models: PlannerOutput, ValidationVerdict, GraphState
│   ├── ingestion/
│   │   └── docling_ingest.py      # Docling PDF → structured data → merged into campus_data.json
│   ├── tools/
│   │   └── campus_data_lookup.py
│   ├── agents/
│   │   ├── planner_agent.py       # CrewAI agent, PydanticAI-typed output
│   │   ├── information_agent.py
│   │   └── validation_agent.py    # CrewAI agent, PydanticAI-typed output
│   ├── graph/
│   │   └── workflow.py            # LangGraph StateGraph definition
│   └── app.py                     # Gradio entrypoint
├── part_b_autogen/                 # REQUIRED — AG2 (AutoGen) rebuild
│   └── autogen_workflow.py
├── part_b_beeai/                    # REQUIRED — BeeAI rebuild
│   └── beeai_poc.py
├── comparison_report.md            # Part B write-up: CrewAI vs AG2 vs BeeAI
└── tests/
    ├── test_workflow.py
    ├── test_docling_ingest.py
    └── test_schemas.py
```

---

## 6. Build Order (step-by-step for the agent)

1. **Scaffold** the repo structure above; write `requirements.txt` (`crewai`, `langchain`, `langgraph`, `pydantic-ai`, `docling`, `ag2`, `beeai`, `langchain-google-genai`, `gradio`, `python-dotenv`). `.env.example` should have a single `GOOGLE_API_KEY=` (or `GEMINI_API_KEY=`, check whichever the installed `langchain-google-genai` version expects) — get a free key at https://aistudio.google.com/apikey. Every agent/LLM call in the codebase should read the model name from `.env` (e.g. `LLM_MODEL=gemini-2.5-flash`) rather than being hardcoded, so swapping providers later is a one-line change.
2. **Mock dataset** — create `data/campus_data.json` with realistic, varied entries across all intent categories (aim for ≥5 entries per category).
3. **Docling ingestion** — drop a sample PDF into `data/source_pdfs/`, write `src/ingestion/docling_ingest.py` to parse it and merge the output into `campus_data.json`. Run it and confirm the merged data looks correct before moving on.
4. **PydanticAI schemas** (`schemas.py`) — define `PlannerOutput`, `ValidationVerdict`, `GraphState` as PydanticAI/Pydantic models.
5. **Custom tool** — implement and unit-test `campus_data_lookup` in isolation, confirm it returns records from both the mock and Docling-derived data.
6. **Agents** — implement Planner, Information, Validation as CrewAI `Agent` objects with clear `role`, `goal`, `backstory`, and system prompts, using the PydanticAI schemas for structured output.
7. **LangGraph workflow** — wire the three agents as nodes, add the conditional retry edge, compile the graph.
8. **Gradio UI** — simple chat interface calling the compiled graph per message; show which intent/tool was used (good for demoing).
9. **Manual test pass** — run ≥10 varied queries covering every intent, including at least one query answerable only from the Docling-ingested PDF, and one with no matching data (fallback path).
10. **Part B — AG2 (AutoGen)** — reimplement the same Planner → Information → Validation pipeline, reusing the same tool and dataset.
11. **Part B — BeeAI** — build the BeeAI proof-of-concept per the brief's own scope note ("small," e.g. a two-agent workflow), also reusing the tool and dataset where practical.
12. **Comparison report** — fill in `comparison_report.md`, now covering three frameworks (CrewAI vs AG2 vs BeeAI), not two.
13. **README** — usage instructions, architecture diagram (ASCII is fine), setup steps, and a short note on the section 2/3 ambiguity and how it was resolved.

---

## 7. Part B — Framework Exploration (both options required)

Reimplement the same Planner → Information → Validation pipeline in **both** AG2 and BeeAI, reusing the same custom tool and dataset so the three-way comparison is apples-to-apples.

### AG2 (AutoGen)
`ConversableAgent` roles with a `GroupChatManager` or sequential handoff, mirroring the CrewAI role split.

### BeeAI
A minimal two-agent (or three-agent, if time allows) BeeAI workflow — the brief itself scopes this as a "small proof of concept," so full feature parity isn't required, just enough to evaluate developer experience against the other two.

### Comparison report — required sections
- **Setup/DX**: install friction, docs quality, boilerplate required — across all three frameworks.
- **Agent definition ergonomics**: how roles/tools are declared in each.
- **Orchestration model**: how control flow (planner → info → validation, retries) is expressed in each, vs. LangGraph's explicit state graph.
- **Debuggability**: how easy it was to trace what each agent did, in each framework.
- **Output quality**: any noticeable differences in the answers themselves.
- **Verdict**: which framework you'd pick for a larger version of this project, and why — with CrewAI, AG2, and BeeAI all weighed against each other.

---

## 8. Evaluation Checklist (map to grading criteria)

- [ ] Planner Agent correctly classifies intent for varied queries
- [ ] Information Agent correctly invokes the custom tool and drafts grounded answers
- [ ] Validation Agent catches at least one deliberately-injected hallucination in testing (demonstrate this in the README with a before/after example)
- [ ] Retry loop works and terminates (no infinite loops)
- [ ] Graceful fallback when no data matches
- [ ] Custom tool is genuinely custom (not just a wrapped LLM call — must hit structured data)
- [ ] **PydanticAI models used for Planner output and Validation verdict, not plain dicts**
- [ ] **Docling ingestion is a working, demoable script, and its output is genuinely used by the tool**
- [ ] Gradio UI runs locally with `python src/app.py`
- [ ] **AG2 (AutoGen) implementation runs and produces comparable outputs**
- [ ] **BeeAI implementation runs and produces comparable outputs**
- [ ] Comparison report covers all three frameworks (CrewAI, AG2, BeeAI) with specific, evidence-based observations
- [ ] All seven frameworks named in the brief's "Mandatory Requirements" section are present and demoable somewhere in the submission
- [ ] Code is modular (agents/tools/graph cleanly separated) and commented

---

## 9. Guidance Notes for the Building Agent

- **Keep prompts deterministic-ish**: ask each agent for strict structured output and parse it with PydanticAI; this avoids brittle string-parsing between graph nodes.
- **Don't over-engineer the dataset** — it's a demo, not a production DB. JSON is fine; SQLite only if you want to show query-building as part of the tool.
- **Docling scope control**: one well-chosen PDF is enough to satisfy the requirement convincingly — don't spend disproportionate time building a PDF corpus when the brief only implies "grounded in real documents," not a full ingestion pipeline.
- **BeeAI scope control**: the brief explicitly calls this a "small proof of concept" — don't over-build it relative to AG2; a clean two-agent demo with a fair comparison write-up is the actual deliverable, not framework parity.
- **Log everything** — print/log each node's input and output during development; strip down for the final demo but keep a `--verbose` flag.
- **Gemini free-tier rate limits**: each query through the pipeline triggers several LLM calls (Planner, Information, Validation, and potentially a retry) — comfortably fine for manual testing, but add a small delay or simple retry-with-backoff around LLM calls so a burst of test queries (step 9) doesn't trip the free tier's requests-per-minute limit. AG2 and BeeAI will add their own call volume on top when you get to Part B, so test those separately rather than back-to-back with Part A.
- **API costs**: batch test queries rather than re-running the whole suite on every small change; cache LLM calls during dev if possible.
- **Build Part A completely before starting any Part B work** — both AG2 and BeeAI reuse the tool and dataset, so a shaky Part A compounds into two shakier comparisons instead of one.
- **Write `DECISIONS.md`** as you go — this is especially important now, since it's also where you record the section 2/3 ambiguity and the "implement everything" resolution. This is a defensible, evidenced answer if a grader asks why the structure looks the way it does.

---

## 10. How to Run This With Any Coding Agent

Put this file in the repo root as `SPEC.md`. Then give the agent one message:

> "Read SPEC.md completely, then build this project end-to-end by working through Section 6 in order. Commit after each numbered step. Report progress as you go, but don't stop for approval between steps unless you hit something this document doesn't answer."

That's it — one message, not a running conversation. The agent has everything it needs in Section 6 (build order), Section 8 (what "correct" looks like), and Section 9 (scope guardrails so it doesn't over- or under-build any one piece). If a step produces something wrong, point at the specific step number when you correct it (*"redo step 6 — the retry edge in the LangGraph never terminates"*) rather than re-explaining the whole project.

If the agent stalls, loses context, or you're switching tools mid-build (e.g. Bob → Antigravity), the same opening message works unchanged — it doesn't matter which agent is reading it.

---

## 11. Definition of Done

Part A runs locally via Gradio, correctly answers queries across all intent categories using the three-agent LangGraph pipeline, the custom tool grounded in both mock and Docling-parsed data, and PydanticAI-typed agent I/O; it visibly validates/corrects at least one bad draft during a demo. Both AG2 and BeeAI rebuilds run and produce comparable outputs, and the comparison report evaluates all three frameworks with specific, evidence-based observations. Every framework named in the brief's "Mandatory Requirements" list is present and demoable.
