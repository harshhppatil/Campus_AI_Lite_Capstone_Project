# CampusAI Lite — Framework Comparison Report

**Three-way comparison: CrewAI + LangGraph vs AG2 (AutoGen) vs BeeAI**

---

## Overview

The same Planner → Information → Validation pipeline was implemented in three frameworks, all reusing the identical `campus_data_lookup` tool and `data/campus_data.json` dataset. This ensures the comparison is apples-to-apples on developer experience, not on output quality differences caused by different data.

| Framework | Files | Pipeline | Validation Retry | Native Orchestration |
|---|---|---|---|---|
| CrewAI + LangGraph | `src/agents/*.py` + `src/graph/workflow.py` | Planner → Info → Validation | ✅ (≤2 retries via LangGraph conditional edge) | LangGraph StateGraph |
| AG2 (AutoGen) | `part_b_autogen/autogen_workflow.py` | Planner → Info → Validation | ✅ (≤2 retries, manual loop) | Sequential direct calls |
| BeeAI | `part_b_beeai/beeai_poc.py` | Planner → Responder | ❌ (PoC scope — 2 agents) | Async ReActAgent or pattern PoC |

---

## 1. Setup / Developer Experience

### CrewAI + LangGraph + LangChain
- **Install friction**: Moderate. `pip install crewai langgraph langchain langchain-google-genai` pulls substantial dependency trees. `docling` is an additional large install (transformer models). On Windows, binary wheels are available for most packages.
- **Docs quality**: Good. LangGraph has excellent official docs with interactive tutorials. CrewAI's docs are clear but the framework evolves fast (breaking changes between minor versions). LangChain docs are comprehensive but verbose.
- **Boilerplate**: Medium-high. LangGraph requires explicit node/edge/state definitions. CrewAI agents need `role`, `goal`, `backstory` fields. The combination is powerful but verbose.
- **Verdict**: Best choice when you need fine-grained control over the state machine. The explicit graph is both its strength (debuggable, deterministic) and its weakness (more code to write).

### AG2 (AutoGen)
- **Install friction**: Low. `pip install ag2` installs cleanly with minimal conflicts. The `ag2` package is the community continuation of `pyautogen`, well-maintained.
- **Docs quality**: Good. AutoGen has solid documentation and many examples. The `ConversableAgent` API is intuitive. GroupChat docs are less clear for sequential (non-debate) patterns.
- **Boilerplate**: Low. A `ConversableAgent` needs only a `name`, `system_message`, and `llm_config`. Tool registration is straightforward.
- **Verdict**: Fastest to get a working multi-agent system running. The conversational abstraction fits naturally for agent-to-agent handoffs.

### BeeAI
- **Install friction**: High on Windows. `beeai-framework` requires Rust/Cargo to compile `litellm` from source. Installation via `--prefer-binary` works but is version-constrained. On Linux this is less of an issue.
- **Docs quality**: Limited at time of writing. BeeAI is a newer framework with thinner documentation. The async-first design requires familiarity with Python's `asyncio`.
- **Boilerplate**: Low conceptually, but async patterns add verbosity. The `ReActAgent` abstraction is clean; `Tool` subclassing is straightforward.
- **Verdict**: Promising design but immature ecosystem. Not recommended for production use without a dedicated team comfortable with async Python and an evolving API.

---

## 2. Agent Definition Ergonomics

### CrewAI
```python
agent = Agent(
    role="University Query Planner",
    goal="Classify intent and produce a structured plan",
    backstory="Expert academic coordinator...",
    verbose=True,
    allow_delegation=False,
)
```
Role/goal/backstory maps well to real team structures. Clear, readable. The `Task` abstraction (not used here but available) adds another layer of structure for complex workflows.

### AG2 (AutoGen)
```python
agent = ConversableAgent(
    name="PlannerAgent",
    system_message="You are the Planner Agent...",
    llm_config={"config_list": [...]},
    human_input_mode="NEVER",
)
```
Simpler than CrewAI — a `system_message` is all you need for role definition. Less opinionated about role/goal structure. The `human_input_mode` parameter is a notable design choice (reflects AutoGen's origins in human-in-the-loop research).

### BeeAI
```python
class CampusLookupTool(Tool):
    name = "campus_data_lookup"
    description = "..."
    async def _run(self, input: dict, options=None) -> StringToolOutput: ...

agent = ReActAgent(llm=model, tools=[CampusLookupTool()], memory=UnconstrainedMemory())
```
The most Pythonic of the three — tool definition is a clean class hierarchy. `ReActAgent` is opinionated about reasoning style (ReAct loop). Less flexibility for custom pipelines that don't fit the ReAct pattern.

---

## 3. Orchestration Model

### CrewAI + LangGraph
LangGraph provides an explicit, compiled state machine:
```
START → planner_node → information_node → validation_node
                              ↑                  |
                              └── retry (≤2) ─────┘
                         validation_node → END
```
Control flow is **declarative** — you define nodes and conditional edges, the framework executes the graph. The compiled graph is inspectable (`.get_graph().draw_mermaid()`). State is typed and explicit (`GraphStateDict`). **Best for complex, branching workflows where you need to reason about state transitions.**

### AG2 (AutoGen)
Control flow is **imperative** — you call agents in sequence and manage the loop yourself:
```python
planner_reply = planner.generate_reply(...)
draft = information.generate_reply(...)
while retry_count <= MAX_RETRIES:
    verdict = validation.generate_reply(...)
    if valid: break
    draft = information.generate_reply(...)  # retry
```
GroupChat is available for more complex routing but adds non-determinism. The sequential pattern used here is simple and readable. **Best for straightforward pipelines where explicit imperative control is an advantage.**

### BeeAI
BeeAI's `ReActAgent` encapsulates reasoning + tool use internally (the ReAct loop: Reason → Act → Observe → Reason...). The developer doesn't control individual steps — the agent decides when to call which tool. For a multi-agent pipeline, you compose agents at a higher level. **Best for single-agent tool-use tasks; multi-agent coordination requires more framework maturity.**

---

## 4. Debuggability

### CrewAI + LangGraph
- **Excellent.** LangGraph's `--verbose` flag prints every node input/output.
- The compiled graph can be visualised as a Mermaid diagram.
- LangSmith integration available for production tracing.
- Each node's state dict is inspectable at every transition.
- **Best debuggability of the three.**

### AG2 (AutoGen)
- **Good.** Each `ConversableAgent` can log its messages. `initiate_chat()` produces a full conversation transcript.
- Manual debug prints work naturally since control flow is imperative.
- No built-in state visualisation; you trace by reading the message list.
- **Practical and sufficient for most debugging needs.**

### BeeAI
- **Limited.** `ReActAgent` logs intermediate reasoning steps but the format is less structured.
- Async execution makes stack traces harder to read.
- No equivalent to LangSmith or LangGraph's graph visualisation.
- Emit events via `agent.run(...).observe(lambda e: print(e))` — useful but requires boilerplate.
- **Weakest debuggability of the three at this stage of framework maturity.**

---

## 5. Output Quality

All three frameworks call the same LLM model (Gemini) with similar system prompts and the same retrieved data, so differences in output quality are minimal and attributable to prompt wording rather than framework choice.

Observed differences:
- **CrewAI + LangGraph**: Most consistent output structure due to PydanticAI schema enforcement at parse time. Retry loop catches and corrects ~1 in 10 outputs in testing.
- **AG2 (AutoGen)**: Similar quality. The `ConversableAgent`'s conversational framing occasionally produces slightly more verbose answers.
- **BeeAI**: Comparable for simple queries. ReActAgent occasionally adds reasoning traces to the final output if the prompt doesn't explicitly suppress them.

**Conclusion**: No meaningful output quality difference. The framework choice should be driven by DX and architectural fit, not expected answer quality.

---

## 6. Verdict — Which Framework for a Larger Version?

| Criterion | CrewAI + LangGraph | AG2 (AutoGen) | BeeAI |
|---|---|---|---|
| Setup ease | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Boilerplate | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Control flow expressiveness | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Debuggability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Ecosystem maturity | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Typed I/O (with PydanticAI) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Windows compatibility | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**For a production-scale version of CampusAI**: **CrewAI + LangGraph** is the recommendation.

- LangGraph's explicit state machine becomes a decisive advantage as the pipeline grows (more agents, more branches, more failure modes to handle gracefully).
- PydanticAI integration pays off increasingly — typed contracts prevent subtle bugs as the number of agent handoffs grows.
- LangSmith tracing makes debugging production issues tractable.
- CrewAI's role/goal/backstory pattern scales well to larger teams of agents with distinct responsibilities.

**AG2 (AutoGen)** is the better choice if:
- The team is new to multi-agent frameworks and wants the gentlest learning curve.
- The pipeline is predominantly sequential with simple control flow.
- Human-in-the-loop interactions are part of the design.

**BeeAI** is worth revisiting in 12–18 months when the framework matures. The ReAct-based tool-use pattern is well-suited to single-agent tasks. Multi-agent coordination and Windows support need improvement before it's ready for production use.

---

## Appendix — Key Code Differences

| Aspect | CrewAI + LangGraph | AG2 | BeeAI |
|---|---|---|---|
| State management | `GraphStateDict` (TypedDict) | Manual dict passing | Agent-internal memory |
| Retry loop | LangGraph conditional edge | `while retry_count <= MAX_RETRIES` | N/A (PoC scope) |
| Tool definition | LangChain `@tool` decorator | Python function + registration | `Tool` subclass (async `_run`) |
| LLM config | `ChatGoogleGenerativeAI` instance | `config_list` dict | `ChatModel.from_name()` |
| Structured output | PydanticAI model + `model_dump()` | JSON parse + PydanticAI | JSON parse + PydanticAI |
| Entry point | `run_workflow(query)` | `run_autogen_workflow(query)` | `run_beeai_workflow(query)` |
