"""OrchestratorAgent — analyzes the user prompt and produces a structured task brief."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class OrchestratorAgent(BaseAgent):
    name = "Orchestrator"
    emoji = "🎯"
    system_prompt = (
        "You are the lead orchestrator of a software engineering team. "
        "Analyze the user's request and produce a structured task brief that will be "
        "handed off to specialist agents. Your brief MUST include ALL of the following "
        "clearly labeled sections:\n\n"
        "## Project Type\n"
        "One-line description of what is being built.\n\n"
        "## Tech Stack\n"
        "Recommended technologies (e.g., vanilla HTML/CSS/JS, React, Python, etc.). "
        "Prefer self-contained single-file HTML+CSS+JS for games and small apps.\n\n"
        "## Key Features\n"
        "Bullet list of the core features to implement.\n\n"
        "## Instructions for Architect\n"
        "Specific guidance on file structure, data models, key functions/classes.\n\n"
        "## Instructions for Designer\n"
        "Specific guidance on visual style, color palette direction, UX feel.\n\n"
        "## Instructions for Coder\n"
        "Important implementation notes, edge cases to handle, coding standards.\n\n"
        "## Instructions for Reviewer\n"
        "What to focus on during review: correctness, performance, completeness.\n\n"
        "Be concise and actionable. Do not write any code here — only the brief."
    )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        prompt = context.get("user_prompt", "")
        mode = context.get("mode", "")
        extra = f"\n\nHint about project mode: {mode}" if mode else ""
        return f"User request: {prompt}{extra}"
