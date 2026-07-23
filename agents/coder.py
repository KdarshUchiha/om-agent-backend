"""CoderAgent — writes complete, working, self-contained code."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class CoderAgent(BaseAgent):
    name = "Coder"
    emoji = "💻"
    system_prompt = (
        "You are an expert full-stack programmer. Given the architect's technical plan "
        "and the designer's CSS, write COMPLETE, WORKING, SELF-CONTAINED code for "
        "whatever the project is — a game, a dashboard, a landing page, a tool, a "
        "visualization. Do not assume it is a game.\n\n"
        "CRITICAL RULES:\n"
        "1. Write EVERY line of code — no placeholders, no '// TODO', no '...rest of code'.\n"
        "2. For small web apps: produce a SINGLE index.html file with CSS and JS "
        "   embedded using <style> and <script> tags. Split into multiple files only "
        "   when the architect's plan calls for it.\n"
        "3. Incorporate the designer's CSS exactly (inside the <style> tag), and honor "
        "   any UI requirements the user specified — layout, theme, palette, fonts.\n"
        "4. Follow the architect's data structures and function signatures precisely.\n"
        "5. Handle edge cases relevant to the project: empty states, invalid input, "
        "   loading/error states, resize events, and (for games) game-over/restart.\n"
        "6. The code must run perfectly in a modern browser with zero external dependencies "
        "   (unless CDN links are appropriate, in which case include them).\n"
        "7. Output ONLY the file(s) — no explanations before or after. "
        "   Wrap each file in a fenced code block labeled with the filename:\n"
        "   ```html index.html\n"
        "   ...content...\n"
        "   ```\n"
        "   If multiple files, repeat the pattern for each file.\n"
        "8. Make it genuinely interactive and functional — wire up every control, "
        "   input, and state transition the project needs. For games: smooth animation "
        "   (requestAnimationFrame), controls, scoring, and restart flow.\n"
        "9. NEVER use fake base64 image data or placeholder asset strings. "
        "   Draw graphics with the Canvas 2D API or pure CSS/SVG, or embed the Asset "
        "   Artist's SVGs. No <img src='data:image/...'> with made-up content.\n"
        "10. NEVER truncate the output. The file must be 100% complete and runnable as-is."
    )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        user_prompt = context.get("user_prompt", "")
        task_brief = self._trim(context.get("orchestrator_output", ""), 1500)
        architect_plan = self._trim(context.get("architect_output", ""), 3000)
        designer_css = self._trim(context.get("designer_output", ""), 2000)
        assets = self._trim(context.get("asset_artist_output", ""), 3000)

        prompt = (
            f"Original user request: {user_prompt}\n\n"
            f"## Orchestrator Task Brief\n{task_brief}\n\n"
            f"## Architect's Technical Plan\n{architect_plan}\n\n"
            f"## Designer's CSS & Styles\n{designer_css}\n\n"
        )

        # The Asset Artist emits "NO_ASSETS_NEEDED" for projects that need no
        # custom artwork (dashboards, forms, text tools). Only wire in the
        # sprite-embedding instructions when real assets were actually produced.
        has_assets = bool(assets) and "NO_ASSETS_NEEDED" not in assets
        if has_assets:
            prompt += (
                f"## Asset Artist's SVG Assets\n{assets}\n\n"
                "IMPORTANT: Use the SVG assets above — inline them directly in the HTML, "
                "reference them as data URIs for <img> tags, or draw them onto a Canvas:\n"
                "  const url = `data:image/svg+xml,${encodeURIComponent(svgString)}`;\n"
                "Use EVERY asset the artist created — they are designed for this project.\n\n"
            )

        prompt += (
            "Now write the complete, production-ready code for this project. "
            "Remember: every line must be real, working code."
        )
        return prompt

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[...trimmed for brevity...]"
