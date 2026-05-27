"""
Workspace Chat — Unified conversational endpoint.

Classifies user intent and routes to the appropriate pipeline:
  - FRESH_BUILD: No files, user describes what to build → full 6-agent pipeline
  - ADD_FEATURE: Files exist, user wants something new → full pipeline with context
  - FIX_BUG: Files exist, user reports issue → Debugger → Coder → Reviewer
  - QUESTION: User asks about the project → quick LLM answer (no agents)
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from .base import BaseAgent, get_http_client
from .orchestrator_pipeline import run_pipeline, run_refine_pipeline, _is_bug_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------

def classify_intent(message: str, files: list[dict], history: list[dict]) -> str:
    """
    Classify user intent based on message, existing files, and conversation history.
    Returns: 'fresh_build', 'add_feature', 'fix_bug', 'question'
    """
    has_files = len(files) > 0
    msg_lower = message.lower()

    # No files = first message = fresh build
    if not has_files:
        # Unless it's a pure question with no build intent
        question_starts = ["what", "how", "why", "can you explain", "tell me about", "what is"]
        if any(msg_lower.startswith(q) for q in question_starts) and "build" not in msg_lower and "create" not in msg_lower and "make" not in msg_lower:
            return "question"
        return "fresh_build"

    # Has files — check if bug report
    if _is_bug_report(message):
        return "fix_bug"

    # Check if it's a question (no action needed)
    question_indicators = [
        "how does", "what does", "explain", "why did you", "can you explain",
        "what is", "tell me about", "how would", "what approach",
    ]
    if any(q in msg_lower for q in question_indicators) and not any(
        a in msg_lower for a in ["add", "change", "make", "update", "remove", "build", "create", "implement"]
    ):
        return "question"

    # Default: feature addition/modification
    return "add_feature"


# ---------------------------------------------------------------------------
# Question Answering (no code changes, just respond)
# ---------------------------------------------------------------------------

async def answer_question(
    message: str,
    files: list[dict],
    history: list[dict],
    api_key: str,
    provider: str,
) -> AsyncGenerator[dict, None]:
    """Answer a question about the project without modifying code."""
    from .base import get_http_client, GEMINI_STREAM_URL, GROQ_CHAT_URL, GROQ_MODEL

    yield {
        "type": "agent_start",
        "agent": "Om",
        "emoji": "ॐ",
        "message": "Thinking…",
    }

    # Build context
    files_summary = ", ".join(f.get("name", "?") for f in files) if files else "No files yet"
    recent_history = "\n".join(
        f"{h['role']}: {h['content'][:200]}" for h in history[-6:]
    ) if history else "No prior conversation"

    system = (
        "You are Om — The Divine Architect. Answer the user's question about their project. "
        "Be concise, helpful, and reference the project context. "
        "If they ask about code, explain clearly. "
        "Do NOT output file content or code blocks — just answer the question naturally."
    )
    user_msg = (
        f"Project files: {files_summary}\n\n"
        f"Recent conversation:\n{recent_history}\n\n"
        f"User asks: {message}"
    )

    # Call LLM for a quick answer
    import httpx

    client = get_http_client()
    full_text = ""

    if provider == "groq":
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            "temperature": 0.7, "max_tokens": 1024, "stream": False,
        }
        resp = await client.post(GROQ_CHAT_URL, json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout=30)
        resp.raise_for_status()
        full_text = resp.json()["choices"][0]["message"]["content"]
    else:
        url = GEMINI_STREAM_URL.format(api_key=api_key).replace(":streamGenerateContent?alt=sse&", ":generateContent?")
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
        }
        resp = await client.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        full_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    yield {
        "type": "agent_output",
        "agent": "Om",
        "emoji": "ॐ",
        "chunk": full_text,
    }

    yield {
        "type": "agent_done",
        "agent": "Om",
        "emoji": "ॐ",
        "message": "Answered.",
    }

    # Send as a text response (no files)
    yield {
        "type": "answer",
        "content": full_text,
    }

    yield {"type": "done"}


# ---------------------------------------------------------------------------
# Main Chat Router
# ---------------------------------------------------------------------------

async def run_workspace_chat(
    message: str,
    files: list[dict],
    history: list[dict],
    api_key: str,
    provider: str,
) -> AsyncGenerator[dict, None]:
    """
    Main workspace conversation handler.
    Classifies intent and routes to appropriate pipeline.
    """
    intent = classify_intent(message, files, history)

    # Emit intent classification so frontend knows what's happening
    intent_labels = {
        "fresh_build": "🚀 Building your project from scratch…",
        "add_feature": "✨ Adding new features to your project…",
        "fix_bug": "🔍 Diagnosing and fixing the issue…",
        "question": "💬 Answering your question…",
    }

    yield {
        "type": "intent",
        "intent": intent,
        "message": intent_labels.get(intent, "Processing…"),
    }

    if intent == "fresh_build":
        async for event in run_pipeline(
            user_prompt=message,
            api_key=api_key,
            provider=provider,
        ):
            yield event

    elif intent == "fix_bug":
        async for event in run_refine_pipeline(
            refinement_prompt=message,
            current_files=files,
            conversation=history,
            api_key=api_key,
            provider=provider,
        ):
            yield event

    elif intent == "add_feature":
        async for event in run_refine_pipeline(
            refinement_prompt=message,
            current_files=files,
            conversation=history,
            api_key=api_key,
            provider=provider,
        ):
            yield event

    elif intent == "question":
        async for event in answer_question(message, files, history, api_key, provider):
            yield event
