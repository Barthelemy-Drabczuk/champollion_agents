from __future__ import annotations

import json
import os
from typing import Generator

import gradio as gr
import httpx

TECHNICIAN_URL = os.environ.get("CHAMPOLLION_TECHNICIAN_URL", "http://localhost:8001")
ANALYST_URL = os.environ.get("CHAMPOLLION_ANALYST_URL", "http://localhost:8002")


def _stream_agent(base_url: str, message: str, history: list) -> Generator[list, None, None]:
    history = list(history or [])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{base_url}/runs", json={"message": message})
            resp.raise_for_status()
            run_id = resp.json()["run_id"]

        with httpx.Client(timeout=300) as client:
            with client.stream("GET", f"{base_url}/runs/{run_id}/stream") as stream:
                for line in stream.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = json.loads(line[5:].strip())
                    event = payload.get("event", "")
                    data = payload.get("data", {})

                    if event == "message":
                        history[-1]["content"] += data.get("text", "")
                        yield history
                    elif event == "log":
                        history[-1]["content"] += f"\n`{data.get('line', '')}`"
                        yield history
                    elif event == "done":
                        if data.get("status") == "failed":
                            history[-1]["content"] += f"\n\n**Error:** {data.get('error', 'unknown')}"
                            yield history
                        break

    except Exception as exc:
        history[-1]["content"] = f"**Connection error:** {exc}"
        yield history


def technician_chat(message: str, history: list) -> Generator[list, None, None]:
    yield from _stream_agent(TECHNICIAN_URL, message, history)


def analyst_chat(message: str, history: list) -> Generator[list, None, None]:
    yield from _stream_agent(ANALYST_URL, message, history)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Champollion Agents", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Champollion Sulcal Pipeline Agents")

        with gr.Tabs():
            with gr.Tab("Pipeline Technician"):
                gr.Markdown("Run and monitor the sulcal pipeline.")
                tech_chat = gr.Chatbot(type="messages", height=500)
                tech_input = gr.Textbox(placeholder="e.g. Run the full pipeline for sub-001...")
                tech_input.submit(
                    technician_chat,
                    inputs=[tech_input, tech_chat],
                    outputs=tech_chat,
                ).then(lambda: "", outputs=tech_input)

            with gr.Tab("Data Analyst"):
                gr.Markdown("Explore sulcal embeddings and subject similarities.")
                analyst_chatbot = gr.Chatbot(type="messages", height=500)
                analyst_input = gr.Textbox(placeholder="e.g. Find subjects similar to sub-001 in SC-sylv_left")
                analyst_input.submit(
                    analyst_chat,
                    inputs=[analyst_input, analyst_chatbot],
                    outputs=analyst_chatbot,
                ).then(lambda: "", outputs=analyst_input)

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
