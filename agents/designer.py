"""DesignerAgent — produces CSS/UI/UX decisions and complete stylesheets."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class DesignerAgent(BaseAgent):
    name = "Designer"
    emoji = "🎨"
    system_prompt = (
        "You are a UI/UX designer and CSS expert. Given a task brief, produce a complete "
        "design specification AND the full CSS for the project.\n\n"
        "Your output MUST include:\n\n"
        "## Design Concept\n"
        "2-3 sentences describing the overall visual feel and user experience.\n\n"
        "## Color Scheme\n"
        "Primary, secondary, accent, background, text, and error colors as hex values.\n\n"
        "## Typography\n"
        "Font family choices (prefer system fonts or Google Fonts), sizes, weights.\n\n"
        "## Layout Structure\n"
        "Description of the layout: flex/grid, spacing, breakpoints.\n\n"
        "## CSS\n"
        "Complete, production-ready CSS wrapped in a ```css code block. "
        "Include CSS custom properties (variables) for the color scheme and typography. "
        "Include all necessary styles: reset/base, layout, components, animations, "
        "hover states, and responsive rules. Do NOT use placeholder comments like "
        "'/* add more styles here */' — write the actual styles. "
        "For games, style the canvas, score display, game-over overlay, and controls."
    )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        task_brief = context.get("orchestrator_output", "")
        user_prompt = context.get("user_prompt", "")
        return (
            f"Original user request: {user_prompt}\n\n"
            f"Orchestrator task brief:\n{task_brief}\n\n"
            "Please produce the complete design specification and CSS as described."
        )
