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
    │  Reviewer ✅  │  Reviews, fixes, packages final output
    └───────┬───────┘
            ▼
     Final files list
```

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

**Providers:**
| Provider | Model | Free tier |
|---|---|---|
| `gemini` | `gemini-2.5-flash` | Yes — get key at [aistudio.google.com](https://aistudio.google.com) |
| `groq` | `llama-3.3-70b-versatile` | Yes — get key at [console.groq.com](https://console.groq.com) |

**SSE event types:**

| `type` | Fields | Description |
|---|---|---|
| `agent_start` | `agent`, `emoji`, `message` | Agent begins work |
| `agent_thinking` | `agent`, `emoji`, `message` | Pipeline status update |
| `agent_output` | `agent`, `emoji`, `chunk` | Streamed text chunk |
| `agent_done` | `agent`, `emoji`, `message` | Agent finished |
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
│   ├── base.py                    # BaseAgent — Gemini & Groq streaming
│   ├── orchestrator.py            # OrchestratorAgent
│   ├── architect.py               # ArchitectAgent
│   ├── designer.py                # DesignerAgent
│   ├── coder.py                   # CoderAgent
│   ├── reviewer.py                # ReviewerAgent + output parser
│   └── orchestrator_pipeline.py  # Pipeline wiring (parallel execution)
├── models/
│   ├── __init__.py
│   └── schemas.py                 # Pydantic request/response models
├── requirements.txt
└── .gitignore
```
