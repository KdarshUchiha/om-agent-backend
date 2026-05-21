"""
Pipeline orchestration — runs all five agents in the correct order,
wires their outputs into a shared context dict, and yields SSE event dicts
for the FastAPI endpoint to forward to the browser.

Flow:
  Orchestrator
      ↓
  Architect ──── (parallel) ──── Designer
      └──────────────┬──────────────┘
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
    # 2. Architect + Designer in parallel
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": "Architect & Designer",
        "emoji": "⚡",
        "message": "Architect and Designer are working in parallel…",
    }

    architect_text = ""
    designer_text = ""

    # Run both agents concurrently, merging their event streams in arrival order.
    architect_events: list[dict] = []
    designer_events: list[dict] = []

    async def _collect(agent_gen: AsyncGenerator[dict, None], out: list[dict]) -> None:
        async for ev in agent_gen:
            out.append(ev)

    await asyncio.gather(
        _collect(architect.run(context, api_key, provider), architect_events),
        _collect(designer.run(context, api_key, provider), designer_events),
    )

    # Emit architect events first, then designer (both are already complete)
    for event in architect_events:
        if event.get("type") == "agent_done":
            architect_text = event.pop("_full_text", "")
        yield event

    for event in designer_events:
        if event.get("type") == "agent_done":
            designer_text = event.pop("_full_text", "")
        yield event

    context["architect_output"] = architect_text
    context["designer_output"] = designer_text

    # ------------------------------------------------------------------
    # 3. Coder
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": coder.name,
        "emoji": coder.emoji,
        "message": "Writing the complete code based on the plan and design…",
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
