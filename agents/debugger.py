"""DebuggerAgent — diagnoses issues in existing code before fixing them."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class DebuggerAgent(BaseAgent):
    name = "Debugger"
    emoji = "🔍"
    system_prompt = (
        "You are a senior debugging specialist. Given code and a user-reported issue, "
        "your job is to:\n\n"
        "1. DIAGNOSE: Identify the exact root cause of the bug\n"
        "2. LOCATE: Point to the specific lines/functions causing the issue\n"
        "3. EXPLAIN: Explain WHY it's broken in plain language\n"
        "4. SOLUTION: Describe the fix clearly (what to change and where)\n\n"
        "OUTPUT FORMAT:\n"
        "## Diagnosis\n"
        "One-line summary of the bug.\n\n"
        "## Root Cause\n"
        "Detailed explanation of WHY this breaks.\n\n"
        "## Location\n"
        "The specific code section causing the issue (quote it).\n\n"
        "## Fix\n"
        "Step-by-step description of what needs to change.\n\n"
        "## Additional Issues Found\n"
        "Any other bugs or improvements you noticed while investigating.\n\n"
        "Be precise and technical. Think like a debugger — trace the execution path "
        "that leads to the failure. Don't just guess — reason about the code."
    )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        user_prompt = context.get("user_prompt", "")
        files_text = ""
        current_files = context.get("current_files", [])
        for f in current_files:
            name = f.get("name", "file")
            content = f.get("content", "")
            trimmed = content[:8000] + ("\n...[trimmed]..." if len(content) > 8000 else "")
            files_text += f"\n### {name}\n```\n{trimmed}\n```\n"

        conversation = context.get("conversation", [])
        conv_text = ""
        if conversation:
            for turn in conversation[-6:]:
                role = turn.get("role", "user")
                msg = turn.get("content", "")[:300]
                conv_text += f"\n{role}: {msg}\n"

        return (
            f"## User's Issue Report\n{user_prompt}\n\n"
            f"## Conversation Context\n{conv_text}\n\n"
            f"## Current Code\n{files_text}\n\n"
            "Analyze the code and diagnose the issue the user described. "
            "Be specific — trace through the code logic to find the exact bug."
        )
