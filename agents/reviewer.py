"""ReviewerAgent — reviews code, fixes bugs, and emits the final file list."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    name = "Reviewer"
    emoji = "✅"
    system_prompt = (
        "You are a senior code reviewer and QA engineer. Review the provided code "
        "thoroughly and produce the final, polished deliverable.\n\n"
        "Your responsibilities:\n"
        "1. Fix ALL bugs — syntax errors, logic errors, off-by-one errors, missing event "
        "   listeners, uninitialized variables, broken game loops, etc.\n"
        "2. Ensure the code is COMPLETE — no missing sections, no placeholders.\n"
        "3. Verify the designer's styles are properly integrated.\n"
        "4. For games: confirm game loop, collision detection, score, and restart all work.\n"
        "5. Clean up code quality: remove dead code, fix naming, add brief comments where "
        "   useful for clarity.\n\n"
        "OUTPUT FORMAT — you MUST output ONLY the following JSON, nothing else:\n"
        "```json\n"
        "{\n"
        '  "files": [\n'
        '    {"name": "index.html", "content": "...complete file content..."},\n'
        '    {"name": "style.css", "content": "..."}  // only if separate CSS file\n'
        "  ],\n"
        '  "summary": "One or two sentences describing what was built and what was fixed."\n'
        "}\n"
        "```\n"
        "The 'content' values must be the COMPLETE file contents, properly escaped for JSON. "
        "Do not truncate. Do not add any text before or after the JSON block."
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

        # Parse the reviewer's JSON output into a structured final_output event
        files, summary = self._parse_output(full_text, context)
        yield {
            "type": "final_output",
            "files": [{"name": f["name"], "content": f["content"]} for f in files],
            "summary": summary,
        }

    def _build_prompt(self, context: dict[str, Any]) -> str:
        user_prompt = context.get("user_prompt", "")
        coder_output = context.get("coder_output", "")
        architect_plan = context.get("architect_output", "")

        return (
            f"Original user request: {user_prompt}\n\n"
            f"## Architect's Plan (for reference)\n{architect_plan}\n\n"
            f"## Coder's Output (to review and finalize)\n{coder_output}\n\n"
            "Please review, fix any issues, and output the final JSON as specified."
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(
        self, raw: str, context: dict[str, Any]
    ) -> tuple[list[dict], str]:
        """
        Extract the files list and summary from the reviewer's raw output.

        Falls back gracefully: if JSON parsing fails, wraps the coder's output
        as index.html so the user always gets something useful.
        """
        # Try to extract JSON from a ```json ... ``` fence first
        json_text = self._extract_fenced_json(raw)
        if json_text is None:
            # Try to find a bare JSON object
            json_text = self._extract_bare_json(raw)

        if json_text:
            try:
                data = json.loads(json_text)
                files = data.get("files", [])
                summary = data.get("summary", "Project completed successfully.")
                if files:
                    return files, summary
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Reviewer JSON parse failed: %s", exc)

        # Fallback: use the coder's output
        logger.warning("Reviewer did not return valid JSON — falling back to coder output")
        coder_output = context.get("coder_output", "")
        fallback_files = self._extract_files_from_coder(coder_output)
        if not fallback_files:
            # Last resort: wrap everything in a single HTML file
            fallback_files = [{"name": "index.html", "content": coder_output or raw}]

        return fallback_files, "Project generated. Please review the output."

    @staticmethod
    def _extract_fenced_json(text: str) -> str | None:
        """Extract content from ```json ... ``` fences."""
        pattern = r"```json\s*([\s\S]*?)```"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_bare_json(text: str) -> str | None:
        """Find the first { ... } block that looks like our output schema."""
        start = text.find('{"files"')
        if start == -1:
            start = text.find('{ "files"')
        if start == -1:
            return None
        # Walk forward to find the matching closing brace
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _extract_files_from_coder(coder_output: str) -> list[dict]:
        """
        Parse fenced code blocks from the coder's output into files.

        Expects blocks like:
            ```html index.html
            ...
            ```
        """
        files: list[dict] = []
        # Match ```<lang> <filename>\n<content>\n```
        pattern = r"```(?:\w+)?\s+([\w.\-/]+)\n([\s\S]*?)```"
        for match in re.finditer(pattern, coder_output):
            filename = match.group(1).strip()
            content = match.group(2)
            files.append({"name": filename, "content": content})

        if not files:
            # Try without filename label: ```html\n...\n```
            pattern2 = r"```(?:html|css|javascript|js)\n([\s\S]*?)```"
            extensions = {"html": "index.html", "css": "style.css", "javascript": "script.js", "js": "script.js"}
            seen: set[str] = set()
            for match in re.finditer(pattern2, coder_output, re.IGNORECASE):
                lang_match = re.match(r"```(\w+)", coder_output[match.start():])
                lang = lang_match.group(1).lower() if lang_match else "html"
                name = extensions.get(lang, f"{lang}.txt")
                # Avoid duplicates
                if name in seen:
                    name = f"{lang}_{len(seen)}.{lang}"
                seen.add(name)
                files.append({"name": name, "content": match.group(1)})

        return files
