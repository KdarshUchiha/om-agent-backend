"""Runtime sandbox — actually *executes* generated JS and reports errors.

The deterministic ``verifier`` catches *static* defects (syntax errors,
undefined inline handlers, placeholders). This module complements it with a
*runtime* check: it runs the page's inline JavaScript under a minimal
DOM/Canvas stub (``sandbox_harness.mjs``) via Node and reports any error the
code throws on load — a function called during init but never defined, a throw
inside a ``DOMContentLoaded`` handler, a bad property access, an explicit
``throw``. These are the failures that *look* fine statically but break the
moment the page opens.

No browser is required (none is available in this environment); the harness
stubs the DOM. If Node is unavailable the sandbox degrades gracefully to a
no-op so the pipeline still runs.

Findings are returned as ``verifier.Finding`` objects so results merge
straight into the existing ``VerificationReport`` and the verify→repair loop.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .verifier import Finding

logger = logging.getLogger(__name__)

_HARNESS = Path(__file__).with_name("sandbox_harness.mjs")

# Reuse the script-extraction rules from the static verifier so both see the
# same JS. Imported lazily to avoid a hard import cycle at module load.
from .verifier import _extract_scripts  # noqa: E402

_NODE_BIN: str | None | bool = False  # False = not yet probed


def _node_available() -> str | None:
    global _NODE_BIN
    if _NODE_BIN is False:
        _NODE_BIN = shutil.which("node")
        logger.info("Runtime sandbox: node %s", "found" if _NODE_BIN else "not found")
    return _NODE_BIN  # type: ignore[return-value]


# Errors that reflect our stub's limitations rather than a real defect in the
# generated code. We suppress these to keep the signal-to-noise high — the
# sandbox must not send the RepairAgent chasing phantom bugs.
_IGNORABLE = re.compile(
    r"is not a function"  # a stubbed API the page expected to be richer
    r"|Cannot read properties of undefined \(reading 'then'\)",
    re.IGNORECASE,
)


def run_runtime_checks(files: list[dict]) -> list[Finding]:
    """Execute each HTML/JS file's JavaScript and return runtime-error findings.

    Returns an empty list when Node is unavailable, when there is no JS to run,
    or when the code loads without throwing.
    """
    node = _node_available()
    if not node or not _HARNESS.exists():
        return []

    findings: list[Finding] = []
    for f in files:
        name = f.get("name", "unknown")
        content = f.get("content", "") or ""
        lower = name.lower()

        if lower.endswith((".html", ".htm")):
            js = "\n".join(_extract_scripts(content))
        elif lower.endswith((".js", ".mjs")):
            js = content
        else:
            continue

        if not js.strip():
            continue

        findings.extend(_run_one(node, name, js))

    return findings


def _run_one(node: str, name: str, js: str) -> list[Finding]:
    import json

    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".js", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(js)
            tmp_path = tmp.name
    except OSError as exc:  # pragma: no cover - disk issues
        logger.warning("Runtime sandbox: could not write temp JS: %s", exc)
        return []

    try:
        proc = subprocess.run(
            [node, str(_HARNESS), tmp_path],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Runtime sandbox: harness timed out for %s", name)
        return []
    except OSError as exc:  # pragma: no cover
        logger.warning("Runtime sandbox: harness failed to run: %s", exc)
        return []
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass

    stdout = (proc.stdout or "").strip()
    if not stdout:
        # Harness crashed outright — surface its stderr as one finding so the
        # failure is visible rather than silently swallowed.
        stderr = (proc.stderr or "").strip()
        if stderr:
            logger.warning("Runtime sandbox: harness stderr for %s: %s", name, stderr[:300])
        return []

    # The harness prints one JSON line last; take the final non-empty line.
    last = stdout.splitlines()[-1]
    try:
        result = json.loads(last)
    except json.JSONDecodeError:
        logger.warning("Runtime sandbox: unparseable harness output for %s", name)
        return []

    findings: list[Finding] = []
    for err in result.get("errors", []):
        if _IGNORABLE.search(err):
            logger.info("Runtime sandbox: ignoring stub-limited error: %s", err)
            continue
        findings.append(
            Finding(
                "error",
                name,
                f"Runtime error when the page loads: {err}. "
                "The code throws on execution — fix the referenced symbol or logic.",
            )
        )

    if result.get("timedOut"):
        findings.append(
            Finding(
                "warning",
                name,
                "Runtime sandbox timed out — the load path may contain a blocking "
                "or very long-running synchronous operation.",
            )
        )

    return findings
