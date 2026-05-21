"""BaseAgent — shared logic for calling Gemini or Groq and streaming SSE events."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------

GEMINI_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:streamGenerateContent?alt=sse&key={api_key}"
)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Shared async HTTP client (created once, reused across requests)
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    return _http_client


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class for all Om agents."""

    name: str = "BaseAgent"
    emoji: str = "🤖"
    system_prompt: str = "You are a helpful assistant."

    # ------------------------------------------------------------------
    # Public streaming entry-point
    # ------------------------------------------------------------------

    async def run(
        self,
        context: dict[str, Any],
        api_key: str,
        provider: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Run the agent and yield SSE event dicts.

        Subclasses build the `user_message` from `context`, then delegate to
        `_stream_events` which handles the actual API calls and yielding.
        """
        user_message = self._build_prompt(context)
        async for event in self._stream_events(user_message, api_key, provider):
            yield event

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the user-turn message from the shared pipeline context."""

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    async def _stream_events(
        self,
        user_message: str,
        api_key: str,
        provider: str,
    ) -> AsyncGenerator[dict, None]:
        """Call the selected provider and yield SSE event dicts."""

        # Announce start
        yield {
            "type": "agent_start",
            "agent": self.name,
            "emoji": self.emoji,
            "message": f"{self.name} is starting…",
        }

        full_text = ""
        try:
            if provider == "groq":
                async for chunk in self._stream_groq(user_message, api_key):
                    full_text += chunk
                    yield {
                        "type": "agent_output",
                        "agent": self.name,
                        "emoji": self.emoji,
                        "chunk": chunk,
                    }
            else:
                # Default: Gemini
                async for chunk in self._stream_gemini(user_message, api_key):
                    full_text += chunk
                    yield {
                        "type": "agent_output",
                        "agent": self.name,
                        "emoji": self.emoji,
                        "chunk": chunk,
                    }
        except httpx.HTTPStatusError as exc:
            logger.error("%s HTTP error: %s", self.name, exc)
            # exc.response.text may not be readable if we're outside the stream
            # context — use the already-read error_body if available, else status only
            error_body = getattr(exc.response, "_content", None)
            if error_body:
                try:
                    error_body = error_body.decode("utf-8", errors="replace")[:300]
                except Exception:
                    error_body = str(exc.response.status_code)
            else:
                error_body = str(exc.response.status_code)
            yield {
                "type": "error",
                "agent": self.name,
                "message": f"HTTP {exc.response.status_code}: {error_body}",
            }
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s unexpected error", self.name)
            yield {
                "type": "error",
                "agent": self.name,
                "message": str(exc),
            }
            return

        yield {
            "type": "agent_done",
            "agent": self.name,
            "emoji": self.emoji,
            "message": f"{self.name} finished.",
            # Stash full text so orchestrator can read it from the event stream
            "_full_text": full_text,
        }

    # ------------------------------------------------------------------
    # Gemini streaming
    # ------------------------------------------------------------------

    async def _stream_gemini(
        self, user_message: str, api_key: str
    ) -> AsyncGenerator[str, None]:
        """Stream text chunks from the Gemini API (SSE)."""
        url = GEMINI_STREAM_URL.format(api_key=api_key)
        payload = {
            "system_instruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_message}],
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 65536,
            },
        }

        client = get_http_client()
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                await response.aread()  # read body before leaving stream context
                response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                    candidates = data.get("candidates", [])
                    for candidate in candidates:
                        parts = (
                            candidate.get("content", {}).get("parts", [])
                        )
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                except (json.JSONDecodeError, KeyError):
                    continue

    # ------------------------------------------------------------------
    # Groq streaming (OpenAI-compatible)
    # ------------------------------------------------------------------

    async def _stream_groq(
        self, user_message: str, api_key: str
    ) -> AsyncGenerator[str, None]:
        """Stream text chunks from the Groq API (OpenAI-compatible SSE)."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 32768,
        }

        client = get_http_client()
        async with client.stream(
            "POST", GROQ_CHAT_URL, headers=headers, json=payload
        ) as response:
            if response.status_code != 200:
                await response.aread()  # read body before leaving stream context
                response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                    delta = (
                        data.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        yield delta
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue
