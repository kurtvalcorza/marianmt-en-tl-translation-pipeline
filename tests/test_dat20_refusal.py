"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

import marianmt_translation_pipeline as package
from marianmt_translation_pipeline import pipeline as pipeline_module
from marianmt_translation_pipeline import samples as samples_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/marianmt_translation_colab.ipynb"
BYOD_CELL = "marianmt_translation-20"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = "marianmt_translation"
EARLIER = {"records": "old", "splits": "old", "test_records": "old", "dataset_manifests": "old", "frozen_test": "old", "adapt_result": "old", "result_payload": "old"}
SCRIPTED = "/content/sample_data/california_housing_test.csv"


def _namespace(extra: dict | None = None) -> dict:
    ns = {name: getattr(package, name) for name in package.__all__}
    ns.update({name: getattr(samples_module, name) for name in ("MIN_RECORDS", "MAX_RECORDS")})
    ns.update({"MAX_EVAL_RECORDS": pipeline_module.MAX_EVAL_RECORDS, "os": os, "Path": Path, "WEIGHTS_DIR": "w"})
    ns["pipe"] = types.SimpleNamespace(adapter=None)
    ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    ns.update(EARLIER, **(extra or {}))
    return ns


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_adapter").mkdir(parents=True)
    for name in ("train.csv", "evaluation_report.json", "result.json", "translations.csv"):
        (out / f"{STEM}_{name}").write_text("x", encoding="utf-8")


def _section4(monkeypatch, tmp_path, path: str = "", *, use_byod: bool = True, colab=None, extra: dict | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _seed_outputs(tmp_path)
    source = _cell(BYOD_CELL)
    source = source[: source.index("write_dataset_csv(train_records,")]
    if use_byod:
        source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    source = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    ns = _namespace(extra)
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _colab(uploaded: dict):
    module = types.ModuleType("google.colab")
    module.files = types.SimpleNamespace(upload=lambda: uploaded)
    return module


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "california_housing_test.csv").write_text("longitude,latitude,median_house_value\n-122.05,37.37,344700.0\n", encoding="utf-8")
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "bad.json").write_text("[{", encoding="utf-8")
    (tmp_path / "object.json").write_text('{"id": "a"}', encoding="utf-8")
    (tmp_path / "missing.jsonl").write_text("\n".join(json.dumps({"id": f"r{i}", "source": f"s{i}"}) for i in range(40)), encoding="utf-8")
    (tmp_path / "binary.csv").write_bytes(bytes(range(256)))
    (tmp_path / "folder").mkdir()
    return {
        "missing columns (the scripted sample_data CSV)": ("california_housing_test.csv", "CSV is missing columns ['id', 'source', 'target']"),
        "not a dataset suffix": ("hostname", "BYOD datasets must be .csv, .json or .jsonl"),
        "invalid JSON (JSONDecodeError)": ("bad.json", "JSONDecodeError"),
        "JSON that is not an array": ("object.json", "JSON dataset must be an array of records"),
        "a record without a target": ("missing.jsonl", "is missing 'target'"),
        "not UTF-8 (UnicodeDecodeError)": ("binary.csv", "UnicodeDecodeError"),
        "a folder": ("folder", "is not a file"),
        "missing": ("nope.csv", "is not a file"),
    }


CASES = ["missing columns (the scripted sample_data CSV)", "not a dataset suffix", "invalid JSON (JSONDecodeError)", "JSON that is not an array",
         "a record without a target", "not UTF-8 (UnicodeDecodeError)", "a folder", "missing"]


def _assert_refused(caught, file_name, rule, tmp_path):
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


@pytest.mark.parametrize("case", CASES)
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    file_name, rule = _cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / file_name))
    _assert_refused(caught, file_name, rule, tmp_path)


def test_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    source = _cell(BYOD_CELL).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", "BYOD_PATH = 'nope.csv'", 1)
    ns = _namespace()
    pipe = ns["pipe"]
    with pytest.raises(ValueError, match="^nope.csv: BYOD_PATH 'nope.csv' is not a file"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] is pipe


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path)
    _assert_refused(caught, "(no file)", "no Colab upload dialog", tmp_path)


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "upload cancelled or empty"), ({"a.csv": b"1", "b.csv": b"2"}, "a.csv, b.csv", "got 2")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, colab=_colab(uploaded))
    _assert_refused(caught, file_name, rule, tmp_path)


def test_valid_byod_file_still_splits(monkeypatch, tmp_path):
    rows = "id,source,target\n" + "".join(f"r{i},Sentence number {i} here.,Pangungusap bilang {i} dito.\n" for i in range(60))
    (tmp_path / "pairs.csv").write_text(rows, encoding="utf-8")
    ns = _section4(monkeypatch, tmp_path, str(tmp_path / "pairs.csv"))
    assert ns["data_source"] == "BYOD (pairs.csv)" and ns["train_records"] and ns["test_records"]


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "pairs.csv").write_text("id,source,target\n", encoding="utf-8")

    def broken(path):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "pairs.csv"), extra={"load_byod_dataset": broken})
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


def test_sample_corpus_errors_are_not_refusals(monkeypatch, tmp_path):
    def offline(*args, **kwargs):
        raise OSError("no network")

    with pytest.raises(OSError, match="no network") as caught:
        _section4(monkeypatch, tmp_path, use_byod=False, extra={"fetch_corpus": offline})
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()


@pytest.mark.skipif(not Path(SCRIPTED).is_file(), reason="needs Colab's /content/sample_data, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_sample_data_csv(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, SCRIPTED)
    assert _refusal_line(caught.value).startswith("Refused: california_housing_test.csv: ValueError: CSV is missing columns")
