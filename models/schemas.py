"""Pydantic models for request/response schemas."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    """Request body for the /agent/run endpoint."""

    prompt: str = Field(..., description="The user's prompt describing what to build")
    provider: Literal["gemini", "groq"] = Field(
        default="gemini",
        description="AI provider to use: 'gemini' or 'groq'",
    )
    api_key: str = Field(..., description="API key for the selected provider")
    mode: Optional[str] = Field(
        default=None,
        description="Optional hint about the project type, e.g. 'game', 'app', 'website'",
    )


class AgentRefineRequest(BaseModel):
    """Request body for the /agent/refine endpoint."""

    prompt: str = Field(..., description="Refinement instruction from the user")
    provider: Literal["gemini", "groq"] = Field(default="gemini")
    api_key: str = Field(..., description="API key for the selected provider")
    files: list[dict[str, Any]] = Field(
        ..., description="Current project files: [{name, content}, ...]"
    )
    conversation: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversation turns: [{role, content}, ...]",
    )


class WorkspaceFile(BaseModel):
    name: str
    content: str


class AgentChatRequest(BaseModel):
    """Request body for the /agent/chat endpoint — unified workspace conversation."""

    message: str = Field(..., description="User's message")
    provider: Literal["gemini", "groq"] = Field(default="gemini")
    api_key: str = Field(..., description="API key for the selected provider")
    files: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Current project files (empty for first message)",
    )
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Full conversation history: [{role, content}, ...]",
    )
    conversation: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversation turns: [{role, content}, ...]",
    )


class OutputFile(BaseModel):
    """A single output file produced by the agent pipeline."""

    name: str = Field(..., description="Filename, e.g. 'index.html'")
    content: str = Field(..., description="Complete file content")


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str = "ok"
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# SSE event shapes (used as TypedDicts / plain dicts at runtime, defined here
# purely for documentation purposes).
# ---------------------------------------------------------------------------

class SSEEventBase(BaseModel):
    """Base class for all SSE event payloads."""

    type: str


class AgentStartEvent(SSEEventBase):
    type: Literal["agent_start"] = "agent_start"
    agent: str
    emoji: str
    message: str


class AgentThinkingEvent(SSEEventBase):
    type: Literal["agent_thinking"] = "agent_thinking"
    agent: str
    emoji: str
    message: str


class AgentOutputEvent(SSEEventBase):
    type: Literal["agent_output"] = "agent_output"
    agent: str
    emoji: str
    chunk: str


class AgentDoneEvent(SSEEventBase):
    type: Literal["agent_done"] = "agent_done"
    agent: str
    emoji: str
    message: str


class FinalOutputEvent(SSEEventBase):
    type: Literal["final_output"] = "final_output"
    files: list[OutputFile]
    summary: str


class DoneEvent(SSEEventBase):
    type: Literal["done"] = "done"


class ErrorEvent(SSEEventBase):
    type: Literal["error"] = "error"
    agent: str
    message: str
