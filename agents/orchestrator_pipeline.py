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

    await asyncio.gather(
        _collect(architect.run(context, api_key, provider), architect_events),
        _collect(designer.run(context, api_key, provider), designer_events),
        _collect(asset_artist.run(context, api_key, provider), asset_artist_events),
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


async def run_refine_pipeline(
    refinement_prompt: str,
    current_files: list[dict],
    conversation: list[dict],
    api_key: str,
    provider: str,
) -> AsyncGenerator[dict, None]:
    """
    Refine an existing project — only runs Coder + Reviewer.
    Takes current files and a refinement instruction, produces updated files.
    """
    coder = CoderAgent()
    reviewer = ReviewerAgent()

    # Build context with existing files + conversation history
    files_text = ""
    for f in current_files:
        name = f.get("name", "file")
        content = f.get("content", "")
        # Trim each file for token budget
        trimmed = content[:6000] + ("\n...[trimmed]..." if len(content) > 6000 else "")
        files_text += f"\n### {name}\n```\n{trimmed}\n```\n"

    conv_text = ""
    if conversation:
        for turn in conversation[-6:]:  # last 6 turns max
            role = turn.get("role", "user")
            msg = turn.get("content", "")[:500]
            conv_text += f"\n{role}: {msg}\n"

    context: dict[str, Any] = {
        "user_prompt": refinement_prompt,
        "mode": "refine",
        "orchestrator_output": (
            f"This is a REFINEMENT of an existing project.\n"
            f"The user wants to modify/improve their current code.\n\n"
            f"## User's Refinement Request\n{refinement_prompt}\n\n"
            f"## Conversation Context\n{conv_text}\n"
        ),
        "architect_output": f"## Current Project Files\n{files_text}",
        "designer_output": "",
        "asset_artist_output": "",
    }

    # ------------------------------------------------------------------
    # 1. Coder (applies the refinement)
    # ------------------------------------------------------------------
    yield {
        "type": "agent_start",
        "agent": "Refiner",
        "emoji": "🔧",
        "message": "Applying your changes…",
    }

    coder_text = ""
    async for event in coder.run(context, api_key, provider):
        # Rebrand coder as "Refiner" for the UI
        if "agent" in event:
            event["agent"] = "Refiner"
            event["emoji"] = "🔧"
        if event.get("type") == "agent_done":
            coder_text = event.pop("_full_text", "")
        yield event

    context["coder_output"] = coder_text

    # ------------------------------------------------------------------
    # 2. Reviewer (produces final output)
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": reviewer.name,
        "emoji": reviewer.emoji,
        "message": "Reviewing refined code…",
    }

    async for event in reviewer.run(context, api_key, provider):
        yield event

    yield {"type": "done"}
