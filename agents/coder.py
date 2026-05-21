"""CoderAgent — writes complete, working, self-contained code."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class CoderAgent(BaseAgent):
    name = "Coder"
    emoji = "💻"
    system_prompt = (
        "You are an expert full-stack programmer. Given the architect's technical plan "
        "and the designer's CSS, write COMPLETE, WORKING, SELF-CONTAINED code.\n\n"
        "CRITICAL RULES:\n"
        "1. Write EVERY line of code — no placeholders, no '// TODO', no '...rest of code'.\n"
        "2. For games and small apps: produce a SINGLE index.html file with CSS and JS "
        "   embedded using <style> and <script> tags.\n"
        "3. Incorporate the designer's CSS exactly (inside the <style> tag).\n"
        "4. Follow the architect's data structures and function signatures precisely.\n"
        "5. Handle edge cases: game over, empty states, invalid input, resize events.\n"
        "6. The code must run perfectly in a modern browser with zero external dependencies "
        "   (unless CDN links are appropriate, in which case include them).\n"
        "7. Output ONLY the file(s) — no explanations before or after. "
        "   Wrap each file in a fenced code block labeled with the filename:\n"
        "   ```html index.html\n"
        "   ...content...\n"
        "   ```\n"
        "   If multiple files, repeat the pattern for each file.\n"
        "8. For games: implement smooth animation (requestAnimationFrame), keyboard "
        "   controls, score tracking, and a proper game-over/restart flow.\n"
        "9. NEVER use fake base64 image data or placeholder asset strings. "
        "   Draw all graphics with Canvas 2D API (fillRect, arc, beginPath etc.) "
        "   or pure CSS/SVG. No <img src='data:image/...'> with made-up content.\n"
        "10. NEVER truncate the output. The file must be 100% complete and runnable as-is."
    )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        user_prompt = context.get("user_prompt", "")
        task_brief = context.get("orchestrator_output", "")
        architect_plan = context.get("architect_output", "")
        designer_css = context.get("designer_output", "")

        return (
            f"Original user request: {user_prompt}\n\n"
            f"## Orchestrator Task Brief\n{task_brief}\n\n"
            f"## Architect's Technical Plan\n{architect_plan}\n\n"
            f"## Designer's CSS & Styles\n{designer_css}\n\n"
            "Now write the complete, production-ready code for this project. "
            "Remember: every line must be real, working code."
        )
