"""ReviewerAgent — reviews code, fixes bugs, and emits the final file list."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from .base import BaseAgent
from .parsing import parse_output

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    name = "Reviewer"
    emoji = "✅"
    system_prompt = (
        "You are a senior code reviewer and QA engineer. Review the provided code "
        "thoroughly and produce the final, polished deliverable — for whatever the "
        "project is (game, dashboard, site, tool, visualization).\n\n"
        "Your responsibilities:\n"
        "1. Fix ALL bugs — syntax errors, logic errors, off-by-one errors, missing event "
        "   listeners, uninitialized variables, broken loops, etc.\n"
        "2. Ensure the code is COMPLETE — no missing sections, no placeholders.\n"
        "3. Integrate the designer's CSS into the final file and confirm the UI matches "
        "   any requirements the user specified (theme, layout, palette).\n"
        "4. Ensure every interaction works: buttons, forms, inputs, navigation, state "
        "   changes — and for games, the game loop, collisions, score, and restart.\n"
        "5. Confirm empty/loading/error states behave correctly.\n\n"
        "OUTPUT FORMAT — output exactly this structure, nothing else:\n\n"
        "SUMMARY: <one or two sentences describing what was built and what was fixed>\n\n"
        "FILE: index.html\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        "...complete file content here, nothing truncated...\n"
        "```\n\n"
        "Rules:\n"
        "- Output the COMPLETE file. Never truncate. Never write '...' or 'rest of code here'.\n"
        "- If there are multiple files (e.g. separate CSS or JS), add more FILE: blocks.\n"
        "- The code must run correctly when opened in a browser with zero modifications.\n"
        "- Do NOT add any explanation before SUMMARY: or after the last ``` block."
    )

    async def run(
        self,
        context: dict[str, Any],
        api_key: str,
        provider: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Override run to also parse the final output and emit a `final_output` event.
        """
        full_text = ""
        async for event in self._stream_events(
            self._build_prompt(context), api_key, provider
        ):
            # Accumulate full text from done event
            if event.get("type") == "agent_done":
                full_text = event.pop("_full_text", "")
            yield event

        # Parse the reviewer's output into a structured final_output event.
        files, summary = parse_output(full_text, context.get("coder_output", ""))
        yield {
            "type": "final_output",
            "files": [{"name": f["name"], "content": f["content"]} for f in files],
            "summary": summary,
        }

    def _build_prompt(self, context: dict[str, Any]) -> str:
        user_prompt = context.get("user_prompt", "")
        coder_output = context.get("coder_output", "")
        architect_plan = self._trim(context.get("architect_output", ""), 2000)

        return (
            f"Original user request: {user_prompt}\n\n"
            f"## Architect's Plan (for reference)\n{architect_plan}\n\n"
            f"## Coder's Output (to review and finalize)\n{coder_output}\n\n"
            "Please review, fix any issues, and output the final files as specified."
        )

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[...trimmed for brevity...]"
