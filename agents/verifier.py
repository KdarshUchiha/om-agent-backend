"""Deterministic verifier for generated browser projects.

This module runs *without* an LLM. It statically inspects the generated
HTML/JS/CSS files and reports problems that one-shot code generation commonly
produces:

  * JavaScript syntax errors (via ``node --check`` when Node is available,
    with a brace/paren/bracket balance fallback when it is not)
  * Truncated output or leftover placeholders ("// TODO", "...rest of code")
  * Inline event handlers (onclick="foo()") that reference undefined functions
  * ``canvas.getContext`` usage with no ``<canvas>`` element present
  * Missing core HTML structure

Findings are classified as ``error`` (should trigger a repair pass) or
``warning`` (surfaced but not blocking). The verifier is intentionally
conservative: it only flags an ``error`` when it is highly confident, to avoid
sending the RepairAgent on a wild goose chase.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Placeholder / truncation markers that should never appear in a finished file.
_PLACEHOLDER_PATTERNS = [
    r"//\s*\.\.\.\s*rest of (?:the )?code",
    r"//\s*TODO\b",
    r"/\*\s*\.\.\.\s*\*/",
    r"\.\.\.\s*rest of (?:the )?code\s*\.\.\.",
    r"\[\.\.\.trimmed",
    r"\[trimmed",
    r"<!--\s*rest of",
    r"your code here",
    r"implementation goes here",
    r"# TODO: implement",
]

# JS reserved words / globals we must not mistake for user-defined functions.
_JS_BUILTINS = {
    "if", "for", "while", "switch", "return", "function", "catch", "with",
    "alert", "confirm", "prompt", "console", "parseInt", "parseFloat",
    "setTimeout", "setInterval", "requestAnimationFrame", "Math", "Number",
    "String", "Boolean", "Array", "Object", "JSON", "Date", "isNaN",
}


@dataclass
class Finding:
    """A single verification finding."""

    severity: str  # "error" | "warning"
    file: str
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "file": self.file, "message": self.message}


@dataclass
class VerificationReport:
    """Aggregate result of verifying a set of files."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def passed(self) -> bool:
        """True when there are no error-level findings."""
        return len(self.errors) == 0

    def add(self, severity: str, file: str, message: str) -> None:
        self.findings.append(Finding(severity, file, message))

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }

    def as_prompt_text(self) -> str:
        """Render findings as a numbered list for the RepairAgent prompt."""
        lines = []
        for i, f in enumerate(self.findings, 1):
            lines.append(f"{i}. [{f.severity.upper()}] ({f.file}) {f.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node availability (probed once)
# ---------------------------------------------------------------------------

_NODE_BIN: str | None | bool = False  # False = not yet probed


def _node_available() -> str | None:
    global _NODE_BIN
    if _NODE_BIN is False:
        _NODE_BIN = shutil.which("node")
        logger.info("Verifier: node %s", "found" if _NODE_BIN else "not found")
    return _NODE_BIN  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def verify_files(files: list[dict]) -> VerificationReport:
    """Verify a list of ``{name, content}`` files and return a report."""
    report = VerificationReport()
    if not files:
        report.add("error", "(none)", "No files were produced.")
        return report

    for f in files:
        name = f.get("name", "unknown")
        content = f.get("content", "") or ""
        lower = name.lower()

        _check_placeholders(name, content, report)

        if lower.endswith((".html", ".htm")):
            _check_html_structure(name, content, report)
            scripts = _extract_scripts(content)
            js = "\n".join(scripts)
            _check_js_syntax(name, js, report)
            _check_inline_handlers(name, content, js, report)
            _check_canvas(name, content, js, report)
        elif lower.endswith((".js", ".mjs")):
            _check_js_syntax(name, content, report)

    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_placeholders(name: str, content: str, report: VerificationReport) -> None:
    for pat in _PLACEHOLDER_PATTERNS:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            report.add(
                "error", name,
                f"Placeholder/truncation marker found: '{m.group(0).strip()}'. "
                "The file must be complete with no stubs.",
            )
            return  # one is enough to trigger repair


def _check_html_structure(name: str, content: str, report: VerificationReport) -> None:
    low = content.lower()
    if "<html" not in low and "<!doctype html" not in low:
        report.add("warning", name, "No <html> or <!DOCTYPE html> tag found.")
    if "<body" not in low:
        report.add("warning", name, "No <body> tag found.")
    # Unclosed tag that strongly implies truncation.
    if low.count("<script") > low.count("</script>"):
        report.add(
            "error", name,
            "Unclosed <script> tag — output was likely truncated mid-file.",
        )


def _extract_scripts(html: str) -> list[str]:
    """Return the JS contents of every inline <script> (skips src-only tags)."""
    scripts: list[str] = []
    for m in re.finditer(
        r"<script\b([^>]*)>([\s\S]*?)</script>", html, re.IGNORECASE
    ):
        attrs, body = m.group(1), m.group(2)
        # Skip <script src="..."></script> with no inline body.
        if "src=" in attrs.lower() and not body.strip():
            continue
        # Skip non-JS script types (e.g. application/json, text/template).
        type_match = re.search(r'type\s*=\s*["\']([^"\']+)["\']', attrs, re.IGNORECASE)
        if type_match:
            t = type_match.group(1).lower()
            if t not in ("text/javascript", "module", "application/javascript"):
                continue
        scripts.append(body)
    return scripts


def _check_js_syntax(name: str, js: str, report: VerificationReport) -> None:
    if not js.strip():
        return

    node = _node_available()
    if node:
        err = _node_syntax_error(node, js)
        if err:
            report.add("error", name, f"JavaScript syntax error: {err}")
        return

    # Fallback: balance check (catches truncation and gross mistakes).
    imbalance = _bracket_imbalance(js)
    if imbalance:
        report.add(
            "error", name,
            f"Unbalanced {imbalance} in JavaScript — likely a syntax error or "
            "truncated output.",
        )


def _node_syntax_error(node: str, js: str) -> str | None:
    """Run ``node --check`` on the JS. Returns an error string or None."""
    # ES-module syntax must be checked as a module, else `node --check` errors.
    is_module = bool(re.search(r"^\s*(import|export)\s", js, re.MULTILINE))
    suffix = ".mjs" if is_module else ".js"
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=suffix, delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(js)
            tmp_path = tmp.name
    except OSError as exc:  # pragma: no cover - disk issues
        logger.warning("Verifier: could not write temp JS: %s", exc)
        return None

    try:
        proc = subprocess.run(
            [node, "--check", tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            # Keep the most useful line (the SyntaxError message).
            for line in stderr.splitlines():
                if "SyntaxError" in line:
                    return line.strip()
            return stderr.splitlines()[-1].strip() if stderr else "unknown syntax error"
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Verifier: node --check timed out")
        return None
    except OSError as exc:  # pragma: no cover
        logger.warning("Verifier: node --check failed to run: %s", exc)
        return None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def _bracket_imbalance(js: str) -> str | None:
    """Detect unbalanced (), [], {} outside of strings/comments.

    Best-effort: strips line/block comments and string literals, then counts.
    Returns the name of the unbalanced pair, or None if balanced.
    """
    stripped = _strip_js_noise(js)
    pairs = {")": "(", "]": "[", "}": "{"}
    names = {"(": "parentheses ()", "[": "brackets []", "{": "braces {}"}
    stack: list[str] = []
    for ch in stripped:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            opener = pairs[ch]
            if not stack or stack[-1] != opener:
                return names[opener]
            stack.pop()
    if stack:
        return names[stack[-1]]
    return None


def _strip_js_noise(js: str) -> str:
    """Remove comments and string/template literals so bracket counting is safe."""
    out: list[str] = []
    i, n = 0, len(js)
    while i < n:
        ch = js[i]
        two = js[i : i + 2]
        if two == "//":
            nl = js.find("\n", i)
            i = n if nl == -1 else nl
            continue
        if two == "/*":
            end = js.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch in "\"'`":
            quote = ch
            i += 1
            while i < n:
                if js[i] == "\\":
                    i += 2
                    continue
                if js[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _check_inline_handlers(
    name: str, html: str, js: str, report: VerificationReport
) -> None:
    """Flag onclick="foo()" style handlers whose function is never defined."""
    # Collect names defined anywhere in the inline JS.
    defined: set[str] = set()
    defined |= set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", js))
    defined |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", js))
    # window.foo = / this.foo = assignments
    defined |= set(re.findall(r"\b(?:window|globalThis)\.([A-Za-z_$][\w$]*)\s*=", js))

    handler_attr = re.compile(
        r'on\w+\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE
    )
    missing: set[str] = set()
    for m in handler_attr.finditer(html):
        expr = m.group(1)
        # Pull the leading identifier of each call in the handler expression.
        for call in re.finditer(r"([A-Za-z_$][\w$]*)\s*\(", expr):
            fn = call.group(1)
            if fn in _JS_BUILTINS or fn in defined:
                continue
            missing.add(fn)

    for fn in sorted(missing):
        report.add(
            "error", name,
            f"Inline handler calls '{fn}()' but no such function is defined in "
            "any <script>. Define it or the button/control will throw at click.",
        )


def _check_canvas(
    name: str, html: str, js: str, report: VerificationReport
) -> None:
    """Flag getContext usage when the HTML has no <canvas> element."""
    if re.search(r"\.getContext\s*\(", js) and not re.search(
        r"<canvas\b", html, re.IGNORECASE
    ):
        # Allow the case where the canvas is created in JS via createElement.
        if not re.search(r"createElement\s*\(\s*['\"]canvas['\"]", js):
            report.add(
                "error", name,
                "JavaScript calls getContext() but there is no <canvas> element "
                "and none is created via createElement — the context will be null.",
            )
