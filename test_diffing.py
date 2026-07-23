"""Tests for the deterministic diff-based editing engine (agents/diffing.py)."""

from agents.diffing import parse_edit_blocks, apply_edits


def test_parse_single_block():
    raw = (
        "Here are the edits:\n"
        "FILE: index.html\n"
        "<<<<<<< SEARCH\n"
        "const speed = 5;\n"
        "=======\n"
        "const speed = 8;\n"
        ">>>>>>> REPLACE\n"
    )
    blocks = parse_edit_blocks(raw)
    assert len(blocks) == 1
    assert blocks[0].file == "index.html"
    assert blocks[0].search == "const speed = 5;"
    assert blocks[0].replace == "const speed = 8;"


def test_apply_single_edit_preserves_rest_and_input():
    files = [{"name": "index.html", "content": "let x=1;\nconst speed = 5;\nrun();"}]
    blocks = parse_edit_blocks(
        "FILE: index.html\n<<<<<<< SEARCH\nconst speed = 5;\n"
        "=======\nconst speed = 8;\n>>>>>>> REPLACE\n"
    )
    res = apply_edits(files, blocks)
    assert res.applied == 1
    assert res.failed == []
    assert res.all_applied
    assert "const speed = 8;" in res.files[0]["content"]
    assert "let x=1;" in res.files[0]["content"] and "run();" in res.files[0]["content"]
    # input list never mutated
    assert files[0]["content"] == "let x=1;\nconst speed = 5;\nrun();"


def test_missing_search_is_reported():
    files = [{"name": "a.js", "content": "let x=1;"}]
    res = apply_edits(
        files,
        parse_edit_blocks(
            "<<<<<<< SEARCH\nnonexistent line\n=======\nwhatever\n>>>>>>> REPLACE"
        ),
    )
    assert res.applied == 0
    assert len(res.failed) == 1
    assert not res.all_applied


def test_ambiguous_match_is_rejected():
    files = [{"name": "a.js", "content": "foo();\nfoo();"}]
    res = apply_edits(
        files,
        parse_edit_blocks("<<<<<<< SEARCH\nfoo();\n=======\nbar();\n>>>>>>> REPLACE"),
    )
    assert res.applied == 0
    assert "ambiguous" in res.failed[0]
    # nothing corrupted
    assert res.files[0]["content"] == "foo();\nfoo();"


def test_multiple_blocks_across_files():
    files = [
        {"name": "index.html", "content": "A\nB\nC"},
        {"name": "app.js", "content": "x\ny"},
    ]
    raw = (
        "FILE: index.html\n<<<<<<< SEARCH\nB\n=======\nB2\n>>>>>>> REPLACE\n"
        "FILE: app.js\n<<<<<<< SEARCH\ny\n=======\ny2\n>>>>>>> REPLACE"
    )
    res = apply_edits(files, parse_edit_blocks(raw))
    assert res.applied == 2 and res.failed == []
    byname = {f["name"]: f["content"] for f in res.files}
    assert byname["index.html"] == "A\nB2\nC"
    assert byname["app.js"] == "x\ny2"


def test_implicit_file_when_single():
    files = [{"name": "only.html", "content": "hello world"}]
    res = apply_edits(
        files,
        parse_edit_blocks("<<<<<<< SEARCH\nworld\n=======\nthere\n>>>>>>> REPLACE"),
    )
    assert res.applied == 1
    assert res.files[0]["content"] == "hello there"


def test_no_file_marker_with_multiple_files_fails():
    files = [
        {"name": "a.html", "content": "aaa"},
        {"name": "b.js", "content": "bbb"},
    ]
    res = apply_edits(
        files,
        parse_edit_blocks("<<<<<<< SEARCH\naaa\n=======\nzzz\n>>>>>>> REPLACE"),
    )
    assert res.applied == 0
    assert "could not determine target file" in res.failed[0]


def test_second_block_inherits_last_file():
    """A block with no FILE: marker targets the previously edited file."""
    files = [
        {"name": "a.html", "content": "one\ntwo"},
        {"name": "b.js", "content": "three"},
    ]
    raw = (
        "FILE: a.html\n<<<<<<< SEARCH\none\n=======\nONE\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\ntwo\n=======\nTWO\n>>>>>>> REPLACE"
    )
    res = apply_edits(files, parse_edit_blocks(raw))
    assert res.applied == 2 and res.failed == []
    assert {f["name"]: f["content"] for f in res.files}["a.html"] == "ONE\nTWO"


def test_full_rewrite_output_yields_no_blocks():
    """Plain fenced-code output must produce zero edit blocks so the caller
    knows to fall back to full-file parsing."""
    raw = "Here is the whole file:\n```html\n<div></div>\n```"
    assert parse_edit_blocks(raw) == []


def test_partial_apply_reports_the_failure():
    """One good block + one bad block: good applies, bad is reported."""
    files = [{"name": "index.html", "content": "keep\nchange-me"}]
    raw = (
        "<<<<<<< SEARCH\nchange-me\n=======\nchanged\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\nnot-here\n=======\nx\n>>>>>>> REPLACE"
    )
    res = apply_edits(files, parse_edit_blocks(raw))
    assert res.applied == 1
    assert len(res.failed) == 1
    assert not res.all_applied  # partial success is not full success
    assert "changed" in res.files[0]["content"]


if __name__ == "__main__":
    import sys

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
