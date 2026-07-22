"""RepairAgent — fixes the specific defects found by the deterministic verifier.

Unlike the Reviewer (which does an open-ended quality pass), the RepairAgent is
given a precise, machine-generated list of concrete defects and must fix exactly
those while re-emitting the complete file(s). This is the "repair" half of the
verify→repair loop.
"""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class RepairAgent(BaseAgent):
    name = "Repair"
    emoji = "🔧"
    system_prompt = (
        "You are a meticulous bug-fixing engineer. You are given a project's "
        "current file(s) and a list of concrete defects found by an automated "
        "verifier. Fix EVERY listed defect while preserving everything that "
        "already works.\n\n"
        "RULES:\n"
        "1. Fix each defect in the report precisely — undefined functions, syntax "
        "   errors, missing elements, truncated code, leftover placeholders.\n"
        "2. If a defect says a function is undefined, DEFINE it with a real, "
        "   working implementation — do not just remove the call.\n"
        "3. If a defect says the output was truncated, COMPLETE the missing code.\n"
        "4. Do NOT introduce new features or restyle the project. Fix only.\n"
        "5. Output the COMPLETE corrected file(s). Never truncate, never write "
        "   '...' or 'rest of code here'.\n\n"
        "OUTPUT FORMAT — output exactly this, nothing else:\n\n"
        "SUMMARY: <one sentence on what you fixed>\n\n"
        "FILE: index.html\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        "...complete corrected file...\n"
        "```\n\n"
        "Add more FILE: blocks if there are multiple files. No prose before "
        "SUMMARY: or after the last ``` block."
    )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        user_prompt = context.get("user_prompt", "")
        report_text = context.get("verification_report", "")
        files = context.get("current_files", [])

        files_text = ""
        for f in files:
            fname = f.get("name", "file")
            content = f.get("content", "")
            files_text += f"\n### {fname}\n```\n{content}\n```\n"

        return (
            f"Original user request: {user_prompt}\n\n"
            f"## Automated Verifier Report (defects to fix)\n{report_text}\n\n"
            f"## Current Project Files\n{files_text}\n\n"
            "Fix every defect listed above and output the complete corrected file(s)."
        )
