"""Tests for the runtime sandbox (agents/runtime_sandbox.py).

These verify that executing the generated JS surfaces load-time errors that
the *static* verifier cannot see, while leaving valid code untouched. They are
skipped automatically when Node is unavailable.
"""

from agents.runtime_sandbox import run_runtime_checks, _node_available
from agents.verifier import verify_files


def _html(script: str) -> list[dict]:
    return [{"name": "index.html", "content": f"<!DOCTYPE html><html><body><canvas id='c'></canvas><script>{script}</script></body></html>"}]


def test_runtime_only_bug_is_caught():
    """Undefined function called during init: valid syntax, passes static."""
    files = _html("function init(){ setupLevel(); } init();")
    assert verify_files(files).passed  # static is clean
    findings = run_runtime_checks(files)
    if not _node_available():
        assert findings == []  # graceful no-op without Node
        return
    assert any("setupLevel" in f.message for f in findings)
    assert all(f.severity == "error" for f in findings)


def test_throw_in_domcontentloaded_is_caught():
    files = _html("document.addEventListener('DOMContentLoaded', function(){ startGame(); });")
    assert verify_files(files).passed
    findings = run_runtime_checks(files)
    if not _node_available():
        assert findings == []
        return
    assert any("startGame" in f.message for f in findings)


def test_non_game_ui_bug_is_caught():
    """Runtime checks are domain-agnostic — a todo/form app is covered too."""
    files = [{"name": "index.html", "content":
        "<!DOCTYPE html><html><body><div id='app'></div><script>"
        "const app = document.getElementById('app'); const s = loadState(); render(s);"
        " function render(x){ app.innerHTML='<ul></ul>'; }"
        "</script></body></html>"}]
    assert verify_files(files).passed
    findings = run_runtime_checks(files)
    if not _node_available():
        assert findings == []
        return
    assert any("loadState" in f.message for f in findings)


def test_valid_app_is_clean():
    files = _html(
        "let score=0;"
        "function draw(){ const ctx=document.getElementById('c').getContext('2d'); ctx.fillRect(0,0,10,10); }"
        "function tick(){ score++; draw(); }"
        "document.addEventListener('DOMContentLoaded', tick);"
    )
    assert run_runtime_checks(files) == []


def test_no_js_is_clean():
    files = [{"name": "index.html", "content": "<!DOCTYPE html><html><body><h1>Static page</h1></body></html>"}]
    assert run_runtime_checks(files) == []


def test_touching_many_dom_elements_does_not_false_positive():
    """A page that reads/sets lots of element properties must not be flagged —
    the stub returns chainable proxies so only real JS errors surface."""
    files = _html(
        "const a=document.getElementById('c'); a.style.color='red'; a.classList.add('x');"
        "a.setAttribute('data-x','1'); const w=a.width; const r=a.getBoundingClientRect();"
        "document.querySelectorAll('.item').forEach(e=>e.remove());"
    )
    assert run_runtime_checks(files) == []


if __name__ == "__main__":
    import sys

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    node = "with Node" if _node_available() else "NO Node (graceful-degrade mode)"
    print(f"Running runtime-sandbox tests ({node})\n")
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
