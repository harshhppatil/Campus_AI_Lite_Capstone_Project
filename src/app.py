"""
app.py — Gradio chat UI entrypoint for CampusAI Lite (Gradio 6 compatible)

Run with:
    python src/app.py
    python src/app.py --verbose     # show agent debug output
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is on path when run as: python src/app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from src.graph.workflow import run_workflow, FALLBACK_MESSAGE

# ── Example queries covering all intent categories ─────────────────────────
EXAMPLE_QUERIES = [
    ["What is the tuition fee for B.Tech Computer Science?"],
    ["When does the odd semester end-semester exam start?"],
    ["Show me the Monday timetable for Computer Science year 2"],
    ["What are the office hours of Dr. Priya Sharma?"],
    ["What time does the library close on Sundays?"],
    ["What is the hostel mess dinner timing?"],
    ["Is there a canteen near the engineering block?"],
    ["What is the minimum attendance required?"],
    ["When is the fee payment deadline for odd semester 2025?"],
    ["Tell me about the placement cell"],
]


def chat_fn(message: str, history: list) -> tuple[str, list]:
    """
    Gradio 6 chat callback.
    history is a list of {"role": "user"/"assistant", "content": "..."} dicts.
    """
    if not message.strip():
        return "", history

    start_time = time.time()

    try:
        state = run_workflow(message.strip())
        elapsed = time.time() - start_time

        answer = state.get("final_answer") or FALLBACK_MESSAGE
        intent = state.get("intent", "unknown")
        retries = state.get("retry_count", 0)
        sources = state.get("sources", [])
        is_valid = state.get("is_valid", None)
        error = state.get("error")

        # Identify Docling-sourced records
        docling_records = [
            s for s in sources
            if isinstance(s, dict) and s.get("source") == "docling_pdf"
        ]

        # Build metadata footer
        meta_parts = [
            f"Intent: `{intent}`",
            f"Time: {elapsed:.1f}s",
        ]
        if retries > 0:
            meta_parts.append(f"Retries: {retries}")
        if is_valid is not None:
            meta_parts.append(f"Validated: {'Yes' if is_valid else 'No (fallback)'}")
        if docling_records:
            meta_parts.append(f"Docling records: {len(docling_records)}")
        if sources:
            meta_parts.append(f"Sources: {len(sources)}")
        if error:
            meta_parts.append(f"Note: {error}")

        meta = "\n\n---\n*" + " | ".join(meta_parts) + "*"
        full_response = answer + meta

    except Exception as exc:
        elapsed = time.time() - start_time
        full_response = (
            f"{FALLBACK_MESSAGE}\n\n---\n"
            f"*Error: {exc} | Time: {elapsed:.1f}s*"
        )

    # Gradio 6: history uses role/content dicts
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})
    return "", history


# ── Gradio UI layout ───────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    # Gradio 6: theme and css belong in launch(), not Blocks()
    with gr.Blocks(title="CampusAI Lite") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.Markdown(
            """
# CampusAI Lite — Agentic University Information Assistant

Powered by: **CrewAI** · **LangChain** · **LangGraph** · **PydanticAI** · **Docling** · **Gemini**

Ask anything about fees, timetables, exams, faculty, library, hostel, canteen, or university policies.
"""
        )

        # ── Chat area ─────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="CampusAI Assistant",
                    height=480,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask your university question here...",
                        label="Your question",
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear chat", variant="secondary")

            # ── Sidebar ───────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Example Queries")
                gr.Examples(
                    examples=EXAMPLE_QUERIES,
                    inputs=msg_input,
                    label="",
                )
                gr.Markdown(
                    """
### Pipeline
```
User Query
    ↓
Planner Agent
(intent classification)
    ↓
Information Agent
(tool lookup + draft)
    ↓
Validation Agent
(fact-check, retry ≤ 2)
    ↓
Final Answer
```

### Frameworks
- **CrewAI** — agent roles
- **LangChain** — LLM + tools
- **LangGraph** — state machine
- **PydanticAI** — typed I/O
- **Docling** — PDF ingestion
"""
                )

        # ── Event wiring ──────────────────────────────────────────────────
        msg_input.submit(chat_fn, [msg_input, chatbot], [msg_input, chatbot])
        send_btn.click(chat_fn, [msg_input, chatbot], [msg_input, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

    return demo


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CampusAI Lite — Gradio UI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "true"

    print("=" * 60)
    print("  CampusAI Lite — Starting Gradio UI")
    print(f"  URL: http://localhost:{args.port}")
    print("=" * 60)

    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )
