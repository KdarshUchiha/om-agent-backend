"""BaseAgent — shared logic for calling Gemini or Groq and streaming SSE events."""

from __future__ import annotations

import asyncio
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

# Retry config for 429 rate-limit responses
MAX_RETRIES = 4
RETRY_BASE_DELAY = 15  # seconds — Gemini free tier resets every 60s, start conservative

# Shared async HTTP client (created once, reused across requests)
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
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

        yield {
            "type": "agent_start",
            "agent": self.name,
            "emoji": self.emoji,
            "message": f"{self.name} is starting…",
        }

        full_text = ""
        attempt = 0

        while attempt <= MAX_RETRIES:
            full_text = ""
            error_event = None

            try:
                stream_fn = self._stream_groq if provider == "groq" else self._stream_gemini
                async for chunk in stream_fn(user_message, api_key):
                    full_text += chunk
                    yield {
                        "type": "agent_output",
                        "agent": self.name,
                        "emoji": self.emoji,
                        "chunk": chunk,
                    }
                # Success — break out of retry loop
                break

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                error_body = getattr(exc.response, "_content", b"")
                try:
                    error_body = error_body.decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""

                if status == 429 and attempt < MAX_RETRIES:
                    # Parse retry-after from header or body, else use exponential backoff
                    retry_after = self._parse_retry_after(exc.response, error_body, attempt)
                    logger.warning(
                        "%s rate-limited (429). Retrying in %ds (attempt %d/%d)",
                        self.name, retry_after, attempt + 1, MAX_RETRIES,
                    )
                    yield {
                        "type": "agent_thinking",
                        "agent": self.name,
                        "emoji": self.emoji,
                        "message": (
                            f"⏳ Rate limit hit — waiting {retry_after}s then retrying "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})…"
                        ),
                    }
                    await asyncio.sleep(retry_after)
                    attempt += 1
                    continue

                # Non-429 or out of retries
                short_body = error_body[:300] if error_body else str(status)
                if status == 429:
                    msg = (
                        f"Rate limit exceeded after {MAX_RETRIES} retries. "
                        "Please wait a minute and try again, or switch to Groq in ⚙️ Settings."
                    )
                elif status == 401 or status == 403:
                    msg = "Invalid API key. Check your key in ⚙️ Settings."
                else:
                    msg = f"HTTP {status}: {short_body}"

                logger.error("%s HTTP error %s: %s", self.name, status, short_body)
                error_event = {"type": "error", "agent": self.name, "message": msg}
                break

            except Exception as exc:  # noqa: BLE001
                logger.exception("%s unexpected error", self.name)
                error_event = {"type": "error", "agent": self.name, "message": str(exc)}
                break

        if error_event:
            yield error_event
            return

        yield {
            "type": "agent_done",
            "agent": self.name,
            "emoji": self.emoji,
            "message": f"{self.name} finished.",
            "_full_text": full_text,
        }

    # ------------------------------------------------------------------
    # Retry-after helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_retry_after(response: httpx.Response, body: str, attempt: int) -> int:
        """
        Determine how long to wait before retrying.
        Priority: Retry-After header > retryDelay in body > exponential backoff.
        """
        # 1. Retry-After header (standard)
        header_val = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if header_val:
            try:
                return max(int(float(header_val)), 5)
            except ValueError:
                pass

        # 2. retryDelay in Google error body e.g. "retryDelay": "30s"
        import re
        match = re.search(r'"retryDelay"\s*:\s*"(\d+)s?"', body)
        if match:
            return max(int(match.group(1)), 5)

        # 3. Exponential backoff: 15s, 30s, 60s, 60s
        delays = [15, 30, 60, 60]
        return delays[min(attempt, len(delays) - 1)]

    # ------------------------------------------------------------------
    # Gemini streaming
    # ------------------------------------------------------------------

    async def _stream_gemini(
        self, user_message: str, api_key: str
    ) -> AsyncGenerator[str, None]:
        """Stream text chunks from the Gemini API (SSE)."""
        url = GEMINI_STREAM_URL.format(api_key=api_key)
        payload = {
            "system_instruction": {"parts": [{"text": self.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 65536,
            },
        }

        client = get_http_client()
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                await response.aread()
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
                    for candidate in data.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
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
        async with client.stream("POST", GROQ_CHAT_URL, headers=headers, json=payload) as response:
            if response.status_code != 200:
                await response.aread()
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
