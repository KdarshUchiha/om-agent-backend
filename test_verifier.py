"""Tests for the deterministic verifier.

Run with:  python -m pytest test_verifier.py -v
       or:  python test_verifier.py   (falls back to a plain runner)
"""

from __future__ import annotations

from agents.verifier import verify_files


# ---------------------------------------------------------------------------
# Clean code should pass
# ---------------------------------------------------------------------------

CLEAN_GAME = """<!DOCTYPE html>
<html>
<head><style>canvas { background: #000; }</style></head>
<body>
  <canvas id="game" width="400" height="400"></canvas>
  <button onclick="restart()">Restart</button>
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    let score = 0;
    function restart() {
      score = 0;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    function loop() {
      ctx.fillStyle = '#0f0';
      ctx.fillRect(10, 10, 20, 20);
      requestAnimationFrame(loop);
    }
    loop();
  </script>
</body>
</html>"""


def test_clean_code_passes():
    report = verify_files([{"name": "index.html", "content": CLEAN_GAME}])
    assert report.passed, f"Expected pass, got errors: {report.errors}"


# ---------------------------------------------------------------------------
# Undefined inline handler should error
# ---------------------------------------------------------------------------

UNDEFINED_HANDLER = """<!DOCTYPE html>
<html><body>
  <button onclick="startGame()">Start</button>
  <script>
    let running = false;
    function pause() { running = false; }
  </script>
</body></html>"""


def test_undefined_handler_errors():
    report = verify_files([{"name": "index.html", "content": UNDEFINED_HANDLER}])
    assert not report.passed
    assert any("startGame" in f.message for f in report.errors)


# ---------------------------------------------------------------------------
# getContext with no canvas should error
# ---------------------------------------------------------------------------

CANVAS_NO_ELEMENT = """<!DOCTYPE html>
<html><body>
  <script>
    const ctx = document.getElementById('x').getContext('2d');
    ctx.fillRect(0, 0, 10, 10);
  </script>
</body></html>"""


def test_canvas_without_element_errors():
    report = verify_files([{"name": "index.html", "content": CANVAS_NO_ELEMENT}])
    assert not report.passed
    assert any("canvas" in f.message.lower() for f in report.errors)


def test_canvas_created_in_js_passes_canvas_check():
    html = """<!DOCTYPE html><html><body><script>
      const c = document.createElement('canvas');
      const ctx = c.getContext('2d');
      ctx.fillRect(0,0,5,5);
    </script></body></html>"""
    report = verify_files([{"name": "index.html", "content": html}])
    canvas_errs = [f for f in report.errors if "canvas" in f.message.lower()]
    assert not canvas_errs


# ---------------------------------------------------------------------------
# Truncation / placeholders should error
# ---------------------------------------------------------------------------

def test_placeholder_errors():
    html = """<!DOCTYPE html><html><body><script>
      function move() {
        // ... rest of code ...
      }
    </script></body></html>"""
    report = verify_files([{"name": "index.html", "content": html}])
    assert not report.passed
    assert any("placeholder" in f.message.lower() for f in report.errors)


def test_unclosed_script_errors():
    html = """<!DOCTYPE html><html><body><script>
      function go() { console.log('hi'); }
    """  # no closing </script>
    report = verify_files([{"name": "index.html", "content": html}])
    assert not report.passed


# ---------------------------------------------------------------------------
# Bracket imbalance (fallback path — meaningful when node is absent)
# ---------------------------------------------------------------------------

def test_unbalanced_braces_errors():
    js = "function broken() { if (true) { console.log('x'); }"  # missing }
    report = verify_files([{"name": "script.js", "content": js}])
    assert not report.passed


def test_balanced_js_with_braces_in_strings_passes():
    js = "const s = '}{'; function ok() { return s; }"
    report = verify_files([{"name": "script.js", "content": js}])
    # The braces inside the string literal must not be counted.
    brace_errs = [f for f in report.errors if "brace" in f.message.lower()]
    assert not brace_errs


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_no_files_errors():
    report = verify_files([])
    assert not report.passed


# ---------------------------------------------------------------------------
# Plain runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
