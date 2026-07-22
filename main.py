"""
Om Agent Backend — FastAPI application entry point.

Routes:
    GET  /health        — liveness check
    POST /agent/run     — stream the multi-agent pipeline via SSE
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents.base import get_http_client
from agents.orchestrator_pipeline import run_pipeline, run_refine_pipeline
from agents.workspace_chat import run_workspace_chat
from models.schemas import AgentChatRequest, AgentRefineRequest, AgentRunRequest, HealthResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the shared HTTP client on startup; close it on shutdown."""
    logger.info("Om Agent backend starting up…")
    get_http_client()  # ensure client is created
    yield
    client = get_http_client()
    if not client.is_closed:
        await client.aclose()
    logger.info("Om Agent backend shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Om Agent Backend",
    description=(
        "Multi-agent AI orchestration system. "
        "Streams progress from 5 specialist agents via SSE."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins so that a GitHub Pages frontend can reach this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["Meta"])
async def root() -> dict:
    """Landing route so the Space preview shows status instead of a 404."""
    return {
        "service": "Om Agent Backend",
        "status": "ok",
        "version": "1.0.0",
        "endpoints": ["/health", "/agent/run", "/agent/refine", "/agent/chat"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse()


@app.post("/agent/run", tags=["Agent"])
async def agent_run(request: AgentRunRequest) -> StreamingResponse:
    """
    Run the full Om Agent pipeline and stream progress as Server-Sent Events.

    The client should open this as an EventSource (or read the body as a stream).
    Each event is a JSON object on a `data:` line followed by a blank line.
    """
    if not request.api_key or request.api_key.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="api_key is required. Please provide your Gemini or Groq API key.",
        )

    async def event_stream():
        try:
            async for event in run_pipeline(
                user_prompt=request.prompt,
                api_key=request.api_key.strip(),
                provider=request.provider,
                mode=request.mode,
            ):
                # Strip internal-only keys before sending to client
                event.pop("_full_text", None)
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in event stream")
            error_event = {
                "type": "error",
                "agent": "Pipeline",
                "message": f"Internal server error: {exc}",
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Disable buffering on proxies / Nginx
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/agent/refine", tags=["Agent"])
async def agent_refine(request: AgentRefineRequest) -> StreamingResponse:
    """
    Refine an existing project. Takes current files + a refinement instruction,
    runs only Coder + Reviewer to apply the changes.
    """
    if not request.api_key or request.api_key.strip() == "":
        raise HTTPException(status_code=400, detail="api_key is required.")

    async def event_stream():
        try:
            async for event in run_refine_pipeline(
                refinement_prompt=request.prompt,
                current_files=request.files,
                conversation=request.conversation,
                api_key=request.api_key.strip(),
                provider=request.provider,
            ):
                event.pop("_full_text", None)
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in refine stream")
            yield f'data: {json.dumps({"type": "error", "agent": "Pipeline", "message": str(exc)})}\n\n'
            yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/agent/chat", tags=["Agent"])
async def agent_chat(request: AgentChatRequest) -> StreamingResponse:
    """
    Unified workspace conversation endpoint.
    Agent decides what to do based on message + context:
    - Fresh build (no files, user describes what to build)
    - Add feature (files exist, user wants something new)
    - Fix bug (files exist, user reports issue)
    - Answer question (no code change needed)
    """
    if not request.api_key or request.api_key.strip() == "":
        raise HTTPException(status_code=400, detail="api_key is required.")

    async def event_stream():
        try:
            async for event in run_workspace_chat(
                message=request.message,
                files=request.files,
                history=request.history,
                api_key=request.api_key.strip(),
                provider=request.provider,
            ):
                event.pop("_full_text", None)
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in chat stream")
            yield f'data: {json.dumps({"type": "error", "agent": "Pipeline", "message": str(exc)})}\n\n'
            yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
