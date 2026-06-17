"""OrchestratorAgent — analyzes the user prompt and produces a structured task brief.

Uses Om-Think model (our own reasoning engine) when available,
falls back to user's provider (Gemini/Groq) if Om-Think is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx

from .base import BaseAgent, get_http_client

logger = logging.getLogger(__name__)

OM_THINK_URL = "https://johankira-om-think.hf.space/v1/chat/completions"
OM_THINK_TIMEOUT = 120.0  # CPU inference is slow


class OrchestratorAgent(BaseAgent):
    name = "Orchestrator"
    emoji = "🎯"
    system_prompt = (
        "You are Om-Think — The Divine Reasoning Engine. You are the planning and "
        "architecture layer of a software team.\n\n"
        "Analyze the user's request and produce a structured task brief. "
        "Think deeply about WHY, HOW, TRADEOFFS, and EDGE CASES.\n\n"
        "Your brief MUST include ALL of the following sections:\n\n"
        "## Project Type\n"
        "One-line description of what is being built.\n\n"
        "## Tech Stack\n"
        "Recommended technologies with reasoning. "
        "Prefer self-contained single-file HTML+CSS+JS for games and small apps.\n\n"
        "## Key Features\n"
        "Bullet list of the core features to implement.\n\n"
        "## Architecture Decisions\n"
        "Key technical decisions with tradeoff analysis.\n\n"
        "## Instructions for Architect\n"
        "File structure, data models, key functions/classes.\n\n"
        "## Instructions for Designer\n"
        "Visual style, color palette, UX feel.\n\n"
        "## Instructions for Asset Artist\n"
        "What sprites/graphics are needed, style direction.\n\n"
        "## Instructions for Coder\n"
        "Implementation notes, edge cases to handle.\n\n"
        "## Instructions for Reviewer\n"
        "What to focus on: correctness, performance, completeness.\n\n"
        "## Edge Cases & Risks\n"
        "What could go wrong, and how to handle it.\n\n"
        "Be thorough but concise. Think like a senior staff engineer. "
        "Do not write any code — only the plan."
    )

    async def run(
        self,
        context: dict[str, Any],
        api_key: str,
        provider: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Try Om-Think model first (our own reasoning engine).
        If unavailable or too slow, fall back to user's provider.
        """
        user_message = self._build_prompt(context)

        # Try Om-Think first
        try:
            yield {
                "type": "agent_thinking",
                "agent": self.name,
                "emoji": self.emoji,
                "message": "🧠 Using Om-Think reasoning engine…",
            }
            async for event in self._call_om_think(user_message):
                yield event
            return  # success — don't fall through
        except Exception as e:
            logger.warning("Om-Think unavailable (%s), falling back to %s", e, provider)
            yield {
                "type": "agent_thinking",
                "agent": self.name,
                "emoji": self.emoji,
                "message": f"Om-Think unavailable, using {provider}…",
            }

        # Fallback to user's provider
        async for event in self._stream_events(user_message, api_key, provider):
            yield event

    async def _call_om_think(self, user_message: str) -> AsyncGenerator[dict, None]:
        """Call our Om-Think model for reasoning, with cold-start retry."""
        yield {
            "type": "agent_start",
            "agent": self.name,
            "emoji": self.emoji,
            "message": "Orchestrator is reasoning…",
        }

        payload = {
            "model": "om-think-v1",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": False,
        }

        client = get_http_client()
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = await client.post(
                    OM_THINK_URL,
                    json=payload,
                    timeout=OM_THINK_TIMEOUT,
                )
                if response.status_code == 503 and attempt < max_retries:
                    yield {
                        "type": "agent_thinking",
                        "agent": self.name,
                        "emoji": self.emoji,
                        "message": f"🧠 Om-Think is waking up… retry {attempt + 1}/{max_retries}",
                    }
                    import asyncio
                    await asyncio.sleep(30)
                    continue
                response.raise_for_status()
                break
            except httpx.TimeoutException:
                if attempt < max_retries:
                    yield {
                        "type": "agent_thinking",
                        "agent": self.name,
                        "emoji": self.emoji,
                        "message": f"🧠 Om-Think cold start… retry {attempt + 1}/{max_retries}",
                    }
                    import asyncio
                    await asyncio.sleep(15)
                    continue
                raise

        data = response.json()
        full_text = data["choices"][0]["message"]["content"]

        yield {
            "type": "agent_output",
            "agent": self.name,
            "emoji": self.emoji,
            "chunk": full_text,
        }

        yield {
            "type": "agent_done",
            "agent": self.name,
            "emoji": self.emoji,
            "message": "Orchestrator finished reasoning.",
            "_full_text": full_text,
        }

    def _build_prompt(self, context: dict[str, Any]) -> str:
        prompt = context.get("user_prompt", "")
        mode = context.get("mode", "")
        extra = f"\n\nHint about project mode: {mode}" if mode else ""
        return f"User request: {prompt}{extra}"
