"""BaseAgent — shared logic for streaming from multiple model backends.

Backend selection is delegated to ``agents.router``, which returns an ordered
chain (e.g. Claude → Gemini → Om) per agent. ``_stream_events`` walks that
chain: it streams from the first backend that works and falls back to the next
on failure, so a request never hard-fails while any backend is up. The user's
own provider (Gemini/Groq) is always the guaranteed floor of the chain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

import httpx

from .router import Backend, resolve as resolve_backends

logger = logging.getLogger(__name__)

# Human-friendly labels for the "using backend…" progress event.
_BACKEND_LABEL: dict[Backend, str] = {
    Backend.CLAUDE: "Claude",
    Backend.GEMINI: "Gemini",
    Backend.GROQ: "Groq",
    Backend.OM_THINK: "Om-Think",
    Backend.OM_CODE: "Om-Code",
}

# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------

GEMINI_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:streamGenerateContent?alt=sse&key={api_key}"
)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Claude (Anthropic) — the frontier "brains" tier. Key comes from the
# ANTHROPIC_API_KEY env var (operator-set), model overridable via ANTHROPIC_MODEL.
# Raw httpx SSE to match the Gemini/Groq path and keep requirements SDK-free.
CLAUDE_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
CLAUDE_VERSION = "2023-06-01"

# Om fine-tuned models (self-hosted HF Spaces, OpenAI-compatible, no key).
OM_THINK_URL = "https://johankira-om-think.hf.space/v1/chat/completions"
OM_CODE_URL = "https://johankira-om-code.hf.space/v1/chat/completions"
OM_TIMEOUT = 120.0  # CPU inference is slow

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
        """Walk the router's backend chain, streaming from the first that works.

        Each backend is attempted in order; on terminal failure the loop falls
        back to the next. Within a single frontier backend we keep the original
        429/503 retry-with-backoff behavior. Only when every backend in the
        chain fails do we surface an error event.
        """
        yield {
            "type": "agent_start",
            "agent": self.name,
            "emoji": self.emoji,
            "message": f"{self.name} is starting…",
        }

        chain = resolve_backends(self.name, provider)
        full_text = ""
        last_error_event: dict | None = None
        succeeded = False

        for idx, backend in enumerate(chain):
            is_last = idx == len(chain) - 1
            # Announce a fallback hop (not the first backend).
            if idx > 0:
                yield {
                    "type": "agent_thinking",
                    "agent": self.name,
                    "emoji": self.emoji,
                    "message": f"Falling back to {_BACKEND_LABEL[backend]}…",
                }

            attempt = 0
            backend_failed = False
            while attempt <= MAX_RETRIES:
                full_text = ""
                try:
                    async for chunk in self._stream_backend(
                        backend, user_message, api_key, provider
                    ):
                        full_text += chunk
                        yield {
                            "type": "agent_output",
                            "agent": self.name,
                            "emoji": self.emoji,
                            "chunk": chunk,
                        }
                    succeeded = True
                    break  # backend succeeded

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    error_body = getattr(exc.response, "_content", b"")
                    try:
                        error_body = error_body.decode("utf-8", errors="replace")
                    except Exception:
                        error_body = ""

                    # Retry 429/503 in place ONLY on the last backend (no
                    # further fallback available); otherwise fall back fast.
                    if status in (429, 503) and is_last and attempt < MAX_RETRIES:
                        retry_after = self._parse_retry_after(exc.response, error_body, attempt)
                        reason = "Rate limit hit" if status == 429 else "Server busy"
                        logger.warning(
                            "%s %s on %s (%d). Retrying in %ds (attempt %d/%d)",
                            self.name, reason, backend.value, status,
                            retry_after, attempt + 1, MAX_RETRIES,
                        )
                        yield {
                            "type": "agent_thinking",
                            "agent": self.name,
                            "emoji": self.emoji,
                            "message": (
                                f"⏳ {reason} — waiting {retry_after}s then retrying "
                                f"(attempt {attempt + 1}/{MAX_RETRIES})…"
                            ),
                        }
                        await asyncio.sleep(retry_after)
                        attempt += 1
                        continue

                    short_body = error_body[:300] if error_body else str(status)
                    logger.warning(
                        "%s backend %s failed (HTTP %s): %s",
                        self.name, backend.value, status, short_body,
                    )
                    last_error_event = {
                        "type": "error",
                        "agent": self.name,
                        "message": self._error_message(status, short_body),
                    }
                    backend_failed = True
                    break

                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s backend %s error: %s", self.name, backend.value, exc)
                    last_error_event = {
                        "type": "error", "agent": self.name, "message": str(exc)
                    }
                    backend_failed = True
                    break

            if succeeded:
                break
            if backend_failed:
                continue  # try the next backend in the chain

        if not succeeded:
            yield last_error_event or {
                "type": "error", "agent": self.name,
                "message": "All model backends failed. Please try again.",
            }
            return

        yield {
            "type": "agent_done",
            "agent": self.name,
            "emoji": self.emoji,
            "message": f"{self.name} finished.",
            "_full_text": full_text,
        }

    @staticmethod
    def _error_message(status: int, short_body: str) -> str:
        """Map an HTTP status to a user-facing error message."""
        if status in (429, 503):
            return (
                f"{'Rate limit' if status == 429 else 'Server overloaded'} after "
                f"{MAX_RETRIES} retries. Please wait a minute and try again, or "
                "switch provider in ⚙️ Settings."
            )
        if status == 413:
            return (
                "Request too large for the provider's free tier. "
                "Try a simpler prompt, or switch provider in ⚙️ Settings."
            )
        if status in (401, 403):
            return "Invalid API key. Check your key in ⚙️ Settings."
        return f"HTTP {status}: {short_body}"

    async def _stream_backend(
        self, backend: Backend, user_message: str, api_key: str, provider: str
    ) -> AsyncGenerator[str, None]:
        """Dispatch to the streamer for a specific backend."""
        if backend == Backend.CLAUDE:
            async for chunk in self._stream_claude(user_message):
                yield chunk
        elif backend == Backend.GROQ:
            async for chunk in self._stream_groq(user_message, api_key):
                yield chunk
        elif backend == Backend.GEMINI:
            async for chunk in self._stream_gemini(user_message, api_key):
                yield chunk
        elif backend in (Backend.OM_THINK, Backend.OM_CODE):
            url = OM_THINK_URL if backend == Backend.OM_THINK else OM_CODE_URL
            async for chunk in self._stream_om(url, user_message):
                yield chunk
        else:  # pragma: no cover - unreachable given the router's enum
            raise ValueError(f"Unknown backend: {backend}")

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
            "max_tokens": 8192,
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

    # ------------------------------------------------------------------
    # Claude (Anthropic) streaming — frontier "brains" tier
    # ------------------------------------------------------------------

    async def _stream_claude(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream text chunks from the Anthropic Messages API (SSE).

        Uses adaptive thinking (the only on-mode for Opus 4.8) and streams so
        large outputs don't hit request timeouts. The key is operator-set via
        ANTHROPIC_API_KEY — never the per-request provider key.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            # Router shouldn't route here without a key, but guard anyway so a
            # misconfiguration falls back instead of hanging.
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        headers = {
            "x-api-key": api_key,
            "anthropic-version": CLAUDE_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": 16000,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "thinking": {"type": "adaptive"},
            "stream": True,
        }

        client = get_http_client()
        async with client.stream(
            "POST", CLAUDE_MESSAGES_URL, headers=headers, json=payload
        ) as response:
            if response.status_code != 200:
                await response.aread()
                response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                # Only surface visible text deltas; thinking deltas are omitted.
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text

    # ------------------------------------------------------------------
    # Om fine-tuned models — self-hosted "style layer" (non-streaming HTTP)
    # ------------------------------------------------------------------

    async def _stream_om(self, url: str, user_message: str) -> AsyncGenerator[str, None]:
        """Call an Om model (OpenAI-compatible, CPU, no key) and yield its text.

        The Om Spaces run CPU inference and don't stream reliably, so we make a
        single non-streaming request and yield the full text once. A 503 (cold
        start) surfaces as an HTTPStatusError so ``_stream_events`` falls back.
        """
        payload = {
            "model": "om",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": False,
        }
        client = get_http_client()
        response = await client.post(url, json=payload, timeout=OM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        if text:
            yield text
