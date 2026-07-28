# champollion-agents

LangGraph/ACP agents wrapping the Champollion sulcal pipeline — a conversational layer that lets researchers run and monitor a sulcal-shape neuroimaging pipeline and explore its resulting embeddings, all via chat.

## Overview

The project exposes two independent agents, each backed by the [Claude Code SDK](https://github.com/anthropics/claude-code-sdk-python), behind a shared REST/SSE protocol (referred to here as ACP — Agent Communication Protocol), plus a small Gradio UI in front of both:

- **Technician** — runs and monitors the Champollion sulcal pipeline (preflight checks, parameter adaptation, staged job launches, polling, log inspection), then hands its output off to the Analyst.
- **Analyst** — indexes pipeline output embeddings into ChromaDB and answers questions about subject similarity and per-region statistics.
- **Gradio UI** — a two-tab chat front end that talks to both agents over HTTP.

Each agent is a standalone FastAPI service; they don't share a process. The Technician calls the Analyst's REST API directly to hand off completed runs.

## Architecture

```
                 ┌────────────────────┐
                 │   Gradio UI (7860)  │
                 └─────────┬───────────┘
                HTTP/SSE   │   HTTP/SSE
            ┌──────────────┴──────────────┐
            ▼                             ▼
 ┌────────────────────┐        ┌────────────────────┐
 │ Technician (8001)   │  POST  │  Analyst (8002)     │
 │ FastAPI + SdkAgent   │──────▶ │ FastAPI + SdkAgent   │
 │                      │/index  │                      │
 └──────────┬───────────┘        └──────────┬───────────┘
            │ MCP (stdio, via pixi)          │ MCP (stdio)
            ▼                                ▼
 ┌────────────────────┐        ┌────────────────────┐
 │ champollion_sulcal_ │        │ ChromaDB            │
 │ mcp (external repo) │        │ (<output_dir>/.chromadb) │
 │  → pipeline stages   │        └────────────────────┘
 └────────────────────┘
```

Shared building blocks in `src/champollion_agents/`:

- **`_acp.py`** — a generic FastAPI router implementing the run/stream protocol (`POST /runs`, `GET /runs/{id}/stream` via SSE, `GET /runs/{id}`, session endpoints). Mounted into both the Technician and Analyst apps.
- **`sdk_agent.py`** — `SdkAgent` wraps `claude_code_sdk.query()` behind a LangGraph-style `.ainvoke()` / `.astream()` interface, so both agents run through the Claude Code CLI/SDK rather than calling a model API directly. Includes a compatibility patch that skips unknown Claude CLI event types instead of raising.
- **`llm.py`** — a strategy-pattern `BaseChatModel` builder (`AnthropicStrategy` / `OpenAICompatibleStrategy`, chosen by API key prefix). This is a secondary code path kept for LangChain-based use; the Technician and Analyst agents themselves are driven by `sdk_agent.py`, not this module.
- **`handoff.py`** — posts a `HandoffPayload` from the Technician to the Analyst's `/runs` endpoint once a pipeline run completes.

## Project layout

```
src/champollion_agents/
├── _acp.py            # shared ACP-style FastAPI router (runs, SSE streaming, sessions)
├── sdk_agent.py        # Claude Code SDK → LangGraph-compatible agent wrapper
├── llm.py              # strategy-pattern LLM builder (Anthropic / OpenAI-compatible)
├── handoff.py           # Technician → Analyst handoff client
├── analyst/
│   ├── agent.py         # builds the Analyst SdkAgent + its MCP tool server config
│   ├── server.py        # FastAPI app (ACP router + /index + /health)
│   ├── indexer.py       # loads full_embeddings.csv into ChromaDB
│   ├── tools.py         # LangChain tool versions of the analyst tools
│   └── mcp_server.py     # stdio MCP server exposing the same tools to the agent
└── technician/
    ├── agent.py          # builds the Technician SdkAgent + its MCP server config
    ├── server.py         # FastAPI app (ACP router + /health)
    └── tools.py           # job polling / log tailing / handoff-to-analyst tool
gui/
└── app.py               # Gradio chat UI (Technician + Analyst tabs)
tests/                    # pytest suite (unit + integration), one file per module/server
```

## Requirements

- Python 3.11 or 3.12
- [pixi](https://pixi.sh) for dependency management and task running
- The `claude` CLI on `PATH` (required by `claude-code-sdk` for both agents; integration tests that need it are skipped automatically if it's missing)
- Two external repos this project expects to run alongside, referenced only via env vars — **not included here**:
  - `champollion_sulcal_mcp` — the MCP server exposing the actual pipeline stages (`CHAMPOLLION_MCP_DIR`, default `../champollion_sulcal_mcp`)
  - `champollion_pipeline` — the pipeline itself (`CHAMPOLLION_PIPELINE_DIR`, default `../champollion_pipeline`)

## Installation

```bash
pixi install
```

This installs the `champollion-agents` package in editable mode along with pytest, ruff, and other dev dependencies (see `pixi.toml`).

## Configuration

All configuration is via environment variables (a `.env` file at the repo root is loaded automatically, including in tests):

| Variable | Used by | Default |
|---|---|---|
| `CHAMPOLLION_OUTPUT_DIR` | Analyst | `./outputs` |
| `CHAMPOLLION_MCP_DIR` | Technician | `../champollion_sulcal_mcp` |
| `CHAMPOLLION_PIPELINE_DIR` | Technician | `../champollion_pipeline` |
| `CHAMPOLLION_ANALYST_URL` | Technician (handoff target), Gradio UI | `http://localhost:8002` |
| `CHAMPOLLION_TECHNICIAN_URL` | Gradio UI | `http://localhost:8001` |
| `CHAMPOLLION_LLM_MODEL` | Technician, Analyst (SDK agent model) | `claude-haiku-4-5-20251001` |
| `CHAMPOLLION_LLM_API_KEY` / `CHAMPOLLION_API_KEY` | `llm.py` strategy selection, tests | — |
| `CHAMPOLLION_LLM_BASE_URL` | `llm.py` (OpenAI-compatible strategy) | — |

## Running

Each component is a separate process:

```bash
pixi run run-technician   # FastAPI/uvicorn on :8001
pixi run run-analyst      # FastAPI/uvicorn on :8002
pixi run run-gui          # Gradio UI on :7860
```

## Usage

1. Start the Technician and Analyst services (and optionally the Gradio UI).
2. Ask the Technician to run the pipeline (e.g. via the Gradio "Pipeline Technician" tab, or `POST /runs` directly). It runs a preflight check, adapts parameters to the environment (CPU fallback, job count), launches pipeline stages through the `champollion_sulcal_mcp` MCP server, and polls each job's status file until it succeeds or fails.
3. On successful completion, the Technician hands the combined embeddings off to the Analyst (`POST /index` on the Analyst service), which indexes them into ChromaDB.
4. Ask the Analyst about the results (e.g. via the "Data Analyst" tab) — it can find subjects with similar embeddings in a given region/hemisphere, list indexed regions, and report per-subject embedding statistics.

## API surface

Both the Technician and Analyst expose the same ACP-style endpoints (`_acp.py`):

- `POST /runs` — start a run with a `{"message": ...}` body, returns `{"run_id": ...}`
- `GET /runs/{run_id}/stream` — Server-Sent Events stream of `message` / `log` / `done` events
- `GET /runs/{run_id}` — run status
- `POST /sessions`, `POST /sessions/{session_id}/runs` — session-scoped runs
- `GET /health` — liveness/readiness check

The Analyst additionally exposes:

- `POST /index?combined_embeddings_path=...&run_id=...` — index a completed pipeline run's embeddings into ChromaDB

## Testing

```bash
pixi run test              # full suite
pixi run test-unit          # unit-marked tests only
pixi run test-integration    # integration-marked tests only
pixi run test-cov            # with coverage report
```

Integration tests are skipped automatically when the `claude` CLI or a valid Anthropic API key isn't available, and any test that fails due to billing/auth issues is converted to a skip so CI stays green (see `tests/conftest.py`).

## Linting

```bash
pixi run lint       # ruff check
pixi run lint-fix    # ruff check --fix
pixi run format       # ruff format
```
