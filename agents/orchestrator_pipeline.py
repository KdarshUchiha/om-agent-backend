"""
Pipeline orchestration — runs all six agents in the correct order,
wires their outputs into a shared context dict, and yields SSE event dicts
for the FastAPI endpoint to forward to the browser.

Flow:
  Orchestrator
      ↓
  Architect ──── (parallel) ──── Designer ──── (parallel) ──── Asset Artist
      └─────────────────────────────┬──────────────────────────────┘
                                    ↓
                                  Coder
                                    ↓
                                 Reviewer
                                    ↓
                            final_output / done
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from .architect import ArchitectAgent
from .asset_artist import AssetArtistAgent
from .coder import CoderAgent
from .designer import DesignerAgent
from .orchestrator import OrchestratorAgent
from .reviewer import ReviewerAgent

logger = logging.getLogger(__name__)


async def run_pipeline(
    user_prompt: str,
    api_key: str,
    provider: str,
    mode: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Execute the full Om Agent pipeline and yield SSE event dicts.

    Each agent's full-text output is captured from the `agent_done` event
    (where `_full_text` is stashed by BaseAgent) and stored in a shared
    `context` dict so later agents can read it.
    """

    context: dict[str, Any] = {
        "user_prompt": user_prompt,
        "mode": mode or "",
    }

    orchestrator = OrchestratorAgent()
    architect = ArchitectAgent()
    designer = DesignerAgent()
    asset_artist = AssetArtistAgent()
    coder = CoderAgent()
    reviewer = ReviewerAgent()

    # ------------------------------------------------------------------
    # 1. Orchestrator
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": orchestrator.name,
        "emoji": orchestrator.emoji,
        "message": "Analyzing your request and creating a task brief…",
    }

    orchestrator_text = ""
    async for event in orchestrator.run(context, api_key, provider):
        if event.get("type") == "agent_done":
            orchestrator_text = event.pop("_full_text", "")
        yield event

    context["orchestrator_output"] = orchestrator_text

    # ------------------------------------------------------------------
    # 2. Architect + Designer + Asset Artist in parallel
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": "Architect & Designer & Asset Artist",
        "emoji": "⚡",
        "message": "Architect, Designer, and Asset Artist working in parallel…",
    }

    architect_text = ""
    designer_text = ""
    asset_artist_text = ""

    architect_events: list[dict] = []
    designer_events: list[dict] = []
    asset_artist_events: list[dict] = []

    async def _collect(agent_gen: AsyncGenerator[dict, None], out: list[dict]) -> None:
        async for ev in agent_gen:
            out.append(ev)

    async def _delayed_collect(
        agent_gen: AsyncGenerator[dict, None], out: list[dict], delay: float
    ) -> None:
        """Stagger start to avoid simultaneous API hits on free-tier providers."""
        if delay > 0:
            await asyncio.sleep(delay)
        async for ev in agent_gen:
            out.append(ev)

    # Stagger by 3s each to avoid 503 "high demand" on Gemini free tier
    await asyncio.gather(
        _delayed_collect(architect.run(context, api_key, provider), architect_events, 0),
        _delayed_collect(designer.run(context, api_key, provider), designer_events, 3),
        _delayed_collect(asset_artist.run(context, api_key, provider), asset_artist_events, 6),
    )

    # Emit events in order: architect → designer → asset artist
    for event in architect_events:
        if event.get("type") == "agent_done":
            architect_text = event.pop("_full_text", "")
        yield event

    for event in designer_events:
        if event.get("type") == "agent_done":
            designer_text = event.pop("_full_text", "")
        yield event

    for event in asset_artist_events:
        if event.get("type") == "agent_done":
            asset_artist_text = event.pop("_full_text", "")
        yield event

    context["architect_output"] = architect_text
    context["designer_output"] = designer_text
    context["asset_artist_output"] = asset_artist_text

    # ------------------------------------------------------------------
    # 3. Coder
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": coder.name,
        "emoji": coder.emoji,
        "message": "Writing complete code with assets, design, and architecture…",
    }

    coder_text = ""
    async for event in coder.run(context, api_key, provider):
        if event.get("type") == "agent_done":
            coder_text = event.pop("_full_text", "")
        yield event

    context["coder_output"] = coder_text

    # ------------------------------------------------------------------
    # 4. Reviewer (also emits final_output)
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": reviewer.name,
        "emoji": reviewer.emoji,
        "message": "Reviewing, fixing bugs, and packaging the final output…",
    }

    async for event in reviewer.run(context, api_key, provider):
        yield event

    # ------------------------------------------------------------------
    # 5. Done
    # ------------------------------------------------------------------
    yield {"type": "done"}
