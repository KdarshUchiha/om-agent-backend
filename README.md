---
title: Om Agent Backend
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Om Agent Backend

Multi-agent AI orchestration system built with FastAPI. Send a single prompt and watch 5 specialized AI agents collaborate to build your project — streamed live to the browser via Server-Sent Events.

## Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────┐
│      Orchestrator 🎯    │  Analyzes prompt → structured task brief
└───────────┬─────────────┘
            │
    ┌───────┴────────┐
    ▼                ▼
┌──────────┐   ┌──────────┐
│Architect │   │Designer  │  Run in PARALLEL
│  🏗️      │   │  🎨      │
└────┬─────┘   └────┬─────┘
     └──────┬───────┘
            ▼
    ┌───────────────┐
    │   Coder 💻    │  Writes complete code
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  Reviewer ✅  │  Reviews, fixes, packages first candidate build
    └───────┬───────┘
            ▼
    ┌───────────────────────────────────────┐
    │  Verify 🔎 → Repair 🔧  (loop, max 2)  │  Self-correction
    └───────┬───────────────────────────────┘
            ▼
     Final files list
```

### Verify → Repair loop (agentic self-correction)

After the Reviewer produces a candidate build, a **deterministic verifier**
(no LLM) statically inspects the generated HTML/JS for the failure modes
one-shot generation commonly produces:

- JavaScript syntax errors (`node --check`, with a brace-balance fallback)
- Truncated output / leftover placeholders (`// TODO`, `...rest of code`)
- Inline `onclick="foo()"` handlers referencing **undefined** functions
- `getContext()` calls with no `<canvas>` element
- Missing core HTML structure

If any **error**-level finding is present, the `RepairAgent` is dispatched with
the concrete defect list and re-emits the complete corrected file(s). The loop
runs up to `MAX_REPAIR_ITERATIONS` (default 2) and always ships a `final_output`
even if it cannot fully converge.

**Optional upgrade — browser verification:** the current verifier is static so
it deploys with zero extra dependencies. For runtime error capture (uncaught
exceptions, null refs at load), add Playwright + headless Chromium and extend
`agents/verifier.py` with a browser tier. This is intentionally left out of the
default HF Spaces image to keep cold starts fast.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dev server
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API

### `GET /health`
Returns `{"status": "ok", "version": "1.0.0"}`.

### `POST /agent/run`
Runs the agent pipeline and streams progress as SSE.

**Request body:**
```json
{
  "prompt": "build me a snake game",
  "provider": "gemini",
  "api_key": "your-key-here",
  "mode": "game"
}
```

**Providers** (the user's own key — always the guaranteed fallback floor):
| Provider | Model | Free tier |
|---|---|---|
| `gemini` | `gemini-2.5-flash` | Yes — get key at [aistudio.google.com](https://aistudio.google.com) |
| `groq` | `llama-3.3-70b-versatile` | Yes — get key at [console.groq.com](https://console.groq.com) |

### Model router (backend selection)

`agents/router.py` picks an ordered backend chain per agent — Om (style layer)
and frontier models (the brains) — and `BaseAgent` streams from the first that
works, falling back down the chain. Configured via env vars:

| Env var | Default | Effect |
|---|---|---|
| `OM_ROUTER_MODE` | `hybrid` | `frontier` (brains + style → frontier, Om last), `hybrid` (brains → frontier; style/chat → Om first), `om` (Om first everywhere) |
| `ANTHROPIC_API_KEY` | *(unset)* | When set, Claude joins the frontier tier ahead of the user's provider. Unset → fully backward-compatible (provider + Om only). |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | Claude model ID used when routed to Claude (adaptive thinking, streamed). |

The user's `provider` key is always present in every chain, so a request never
hard-fails because a preferred backend is down.

**SSE event types:**

| `type` | Fields | Description |
|---|---|---|
| `agent_start` | `agent`, `emoji`, `message` | Agent begins work |
| `agent_thinking` | `agent`, `emoji`, `message` | Pipeline status update |
| `agent_output` | `agent`, `emoji`, `chunk` | Streamed text chunk |
| `agent_done` | `agent`, `emoji`, `message` | Agent finished |
| `verify_start` | `agent`, `emoji`, `message` | Verifier begins static checks |
| `verify_result` | `agent`, `emoji`, `passed`, `error_count`, `warning_count`, `findings[]`, `message` | Verification outcome |
| `repair_start` | `agent`, `emoji`, `message` | RepairAgent begins fixing defects |
| `final_output` | `files[]`, `summary` | Complete file list |
| `done` | — | Stream closed |
| `error` | `agent`, `message` | Error occurred |

## Deploying

Any WSGI/ASGI host works. Example with Railway, Render, or Fly.io:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

CORS is set to `allow_origins=["*"]` so GitHub Pages frontends can call it directly.

## Project Structure

```
om-agent-backend/
├── main.py                        # FastAPI app, CORS, /health, /agent/run
├── agents/
│   ├── __init__.py
│   ├── base.py                    # BaseAgent — multi-backend streaming + fallback
│   ├── router.py                  # Model router (Claude/Gemini/Groq/Om) per agent
│   ├── orchestrator.py            # OrchestratorAgent
│   ├── architect.py               # ArchitectAgent
│   ├── designer.py                # DesignerAgent
│   ├── coder.py                   # CoderAgent
│   ├── editor.py                  # EditorAgent (surgical diff edits)
│   ├── diffing.py                 # Deterministic search/replace engine
│   ├── reviewer.py                # ReviewerAgent + output parser
│   ├── verifier.py                # Static verifier (no LLM)
│   ├── runtime_sandbox.py         # Runtime verifier (executes JS via Node)
│   └── orchestrator_pipeline.py  # Pipeline wiring (parallel execution)
├── models/
│   ├── __init__.py
│   └── schemas.py                 # Pydantic request/response models
├── requirements.txt
└── .gitignore
```
