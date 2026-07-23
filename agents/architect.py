"""ArchitectAgent — designs the technical structure and implementation plan."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class ArchitectAgent(BaseAgent):
    name = "Architect"
    emoji = "🏗️"
    system_prompt = (
        "You are a senior software architect. Given a task brief from the orchestrator, "
        "design the complete technical structure for the project. Your output MUST include:\n\n"
        "## File Breakdown\n"
        "List every file that will be created with a one-line description of its purpose.\n\n"
        "## Data Structures\n"
        "Key objects, classes, or state variables with their properties and types.\n\n"
        "## Core Functions / Classes\n"
        "Function signatures and short descriptions for each significant piece of logic.\n\n"
        "## Application Flow\n"
        "Step-by-step description of the main execution flow — initialization, main "
        "loop (if any), event handling, state transitions, data flow, etc.\n\n"
        "## Technical Notes\n"
        "Any important constraints, gotchas, performance considerations, or browser APIs needed.\n\n"
        "Be precise and developer-ready. No code yet — architecture only."
    )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        task_brief = context.get("orchestrator_output", "")
        user_prompt = context.get("user_prompt", "")
        return (
            f"Original user request: {user_prompt}\n\n"
            f"Orchestrator task brief:\n{task_brief}\n\n"
            "Please produce the technical architecture based on the brief above."
        )
