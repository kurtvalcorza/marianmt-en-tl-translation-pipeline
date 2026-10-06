"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/marianmt_translation_colab.ipynb`
(review PR #9: MMT-M1, MMT-M2, MMT-m1..m8).

The notebook's own cells are executed from the committed JSON in a namespace of the package's real functions and
inert stand-ins (a fake pipeline, a fake `google.colab`). Nothing here loads the pinned checkpoint; the model-backed
checks of the same fixes live in `tests/test_model_backed.py` (skipped without the staged snapshot).
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

# Windows conda trap (fleet note, bioclip2 row 6): import torch before anything else touches NumPy in this process.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

import marianmt_translation_pipeline as package  # noqa: E402
from marianmt_translation_pipeline import (  # noqa: E402
    MarianMTTranslationPipeline,
    byod_size_bounds,
    split_dataset,
)
from marianmt_translation_pipeline import pipeline as pipeline_module  # noqa: E402
from marianmt_translation_pipeline import samples as samples_module  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "marianmt_translation_colab.ipynb"


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _code() -> list[str]:
    return [_source(c) for c in _cells() if c["cell_type"] == "code"]


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _cell_with(marker: str) -> str:
    hits = [s for s in _code() if marker in s]
    assert len(hits) == 1, marker
    return hits[0]


def _records(n: int, prefix: str = "r") -> list[dict]:
    return [{"id": f"{prefix}{i:05d}", "source": f"Sentence number {i} is here.", "target": f"Pangungusap {i}."} for i in range(n)]


# --- MMT-M1 / MMT-M2: every re-run starts from the published model ---------------------------------------------------


def test_adapt_refuses_an_already_adapted_pipeline():
    pipe = MarianMTTranslationPipeline(lambda texts, m, b: [(t, 1, "eos") for t in texts], lambda text: 3)
    pipe.adapter = {"best_epoch": 2}
    with pytest.raises(RuntimeError, match="already adapted .*reset_to_pretrained"):
        pipe.adapt(_records(12))


def test_sections_that_measure_or_train_the_pretrained_model_reset_first():
    section4 = _cell_with("def reset_to_pretrained():")
    assert "\nreset_to_pretrained()\nUSE_BYOD = False" in section4
    assert _cell_with("input_manifest = validate_inputs(").startswith("import time\n\nreset_to_pretrained()")
    assert "reset_to_pretrained()  # these outputs are labelled pretrained" in _cell_with("pretrained_outputs = translate_all(test_records)")
    assert _cell_with("frozen_test = pipe.evaluate(").startswith("reset_to_pretrained()")
    section8 = _cell_with("adapt_result = pipe.adapt(")
    assert section8.index("reset_to_pretrained()") < section8.index("adapt_result = pipe.adapt(")


def test_reset_to_pretrained_reloads_only_an_adapted_pipeline(capsys):
    section4 = _cell_with("def reset_to_pretrained():")
    tree = ast.parse(section4)
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reset_to_pretrained")
    loads = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            loads.append(kwargs)
            return types.SimpleNamespace(adapter=None)

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    ns = {"gc": __import__("gc"), "torch": fake_torch, "MarianMTTranslationPipeline": FakePipeline, "WEIGHTS_DIR": "w"}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "<reset>", "exec"), ns)
    base = types.SimpleNamespace(adapter=None)
    ns["pipe"] = base
    assert ns["reset_to_pretrained"]() is base and loads == []
    ns["pipe"] = types.SimpleNamespace(adapter={"best_epoch": 2})
    ns["reset_to_pretrained"]()
    assert loads == [{"weights_dir": "w"}] and ns["pipe"].adapter is None
    assert "the pretrained model, from the verified snapshot" in capsys.readouterr().out


def test_learner_text_states_the_reset_and_the_byod_starting_model():
    md = _markdown()
    assert "Sections 4, 6a, 6b, 7 and 8 begin with `reset_to_pretrained()`" in md
    assert "Section 4 first reloads the published model if the default run fine-tuned it" in md
    assert "`adapted` `False` in the printed record" in md


# --- MMT-m1: BYOD size bounds ------------------------------------------------------------------------------------------


def test_byod_size_bounds_at_the_default_split():
    assert byod_size_bounds() == (12, 10_002)
    assert byod_size_bounds(min_eval_records=samples_module.MIN_RECORDS) == (50, 10_002)


def test_split_refuses_outside_the_bounds_naming_them():
    with pytest.raises(ValueError, match="MAX_EVAL_RECORDS=2000: supply at most 10002"):
        split_dataset(_records(12_000), seed=42, min_eval_records=8)
    with pytest.raises(ValueError, match="in validation; at least 8 are required: supply at least 50"):
        split_dataset(_records(49), seed=42, min_eval_records=8)
    sizes = {k: len(v) for k, v in split_dataset(_records(50), seed=42, min_eval_records=8).items()}
    assert sizes == {"test": 10, "validation": 8, "train": 32}
    assert {k: len(v) for k, v in split_dataset(_records(10_002), seed=42, min_eval_records=8).items()} == {"test": 2000, "validation": 1500, "train": 6502}


def _section4_namespace(tmp_path, monkeypatch, *, pipe_adapter=None):
    monkeypatch.chdir(tmp_path)
    ns = {name: getattr(package, name) for name in package.__all__}
    ns.update({name: getattr(samples_module, name) for name in ("MIN_RECORDS", "MAX_RECORDS")})
    ns.update({"MAX_EVAL_RECORDS": pipeline_module.MAX_EVAL_RECORDS, "os": os, "Path": Path})
    ns["pipe"] = types.SimpleNamespace(adapter=pipe_adapter)
    ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    ns["WEIGHTS_DIR"] = "w"
    return ns


def _run_section4(ns, **fields):
    src = _cell_with("def reset_to_pretrained():")
    for name, value in fields.items():
        lines = [line for line in src.splitlines() if line.startswith(f"{name} = ")]
        assert len(lines) == 1
        src = src.replace(lines[0], f"{name} = {value!r}")
    exec(compile(src, "<section 4>", "exec"), ns)
    return ns


def _write_csv(path: Path, records: list[dict]) -> str:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "source", "target"])
        writer.writeheader()
        writer.writerows(records)
    return str(path)


def test_section4_byod_refuses_a_large_file_before_any_model_work(tmp_path, monkeypatch):
    ns = _section4_namespace(tmp_path, monkeypatch)
    path = _write_csv(tmp_path / "big.csv", _records(12_000))
    with pytest.raises(ValueError, match="MAX_EVAL_RECORDS"):
        _run_section4(ns, USE_BYOD=True, BYOD_PATH=path)


def test_section4_byod_stated_minimum_runs_and_one_less_is_refused(tmp_path, monkeypatch, capsys):
    ns = _section4_namespace(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="at least 8 are required: supply at least 50"):
        _run_section4(ns, USE_BYOD=True, BYOD_PATH=_write_csv(tmp_path / "small.csv", _records(49)))
    ns = _run_section4(_section4_namespace(tmp_path, monkeypatch), USE_BYOD=True, BYOD_PATH=_write_csv(tmp_path / "ok.csv", _records(50)))
    assert ns["disjoint"] == {"test": 10, "validation": 8, "train": 32}
    out = capsys.readouterr().out
    assert "'records_with_distinct_sources': [50, 10002]" in out and "'probe': 'too small', 'rejected'" in out


def test_section4_byod_path_and_upload_refusals_are_actionable(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="is not a file"):
        _run_section4(_section4_namespace(tmp_path, monkeypatch), USE_BYOD=True, BYOD_PATH="no/such.csv")
    monkeypatch.setitem(sys.modules, "google.colab", None)  # no Colab: importing google.colab raises ImportError
    with pytest.raises(RuntimeError, match="set BYOD_PATH"):
        _run_section4(_section4_namespace(tmp_path, monkeypatch), USE_BYOD=True, BYOD_PATH="")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {})
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError, match="upload cancelled or empty"):
        _run_section4(_section4_namespace(tmp_path, monkeypatch), USE_BYOD=True, BYOD_PATH="")


def test_section4_byod_rerun_after_default_reloads_the_pretrained_model(tmp_path, monkeypatch, capsys):
    loads = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            loads.append(kwargs)
            return types.SimpleNamespace(adapter=None)

    ns = _section4_namespace(tmp_path, monkeypatch, pipe_adapter={"best_epoch": 2})
    ns["MarianMTTranslationPipeline"] = FakePipeline
    _run_section4(ns, USE_BYOD=True, BYOD_PATH=_write_csv(tmp_path / "ok.csv", _records(60)))
    assert loads == [{"weights_dir": "w"}] and ns["pipe"].adapter is None
    assert "'reloaded': 'the pretrained model, from the verified snapshot'" in capsys.readouterr().out


def test_prerequisites_and_byod_steps_state_the_enforced_bounds():
    md = _markdown()
    assert "**at least 50 and at most 10,002 records with distinct English sources**" in md
    assert "`MAX_EVAL_RECORDS` = 2,000" in md
    assert "a dataset needs 8..20,000" not in md and "unique ids, 8..20,000 records" not in md


# --- MMT-m2: verdicts, not assertions, for quality outcomes ---------------------------------------------------------


def test_quality_outcomes_are_verdicts_and_negative_results_reach_the_end():
    section7 = _cell_with("frozen_test = pipe.evaluate(")
    section9 = _cell_with("adapted_test = pipe.evaluate(")
    assert "assert frozen_test" not in section7 and "frozen_verdict = " in section7
    assert "assert adapted_test" not in section9 and "adaptation_verdict = " in section9
    assert "comparison['verdicts'] = " in section9
    md = _markdown()
    assert "The Section 9 assertion" not in md and "a legitimate negative result" in md


# --- MMT-m3 / MMT-m8: isolated runtime, no restart, documented executor variable --------------------------------------


def test_exactly_two_kernel_cells_and_no_restart_text():
    kernel = [s for s in _code() if "# dimer: kernel cell" in s]
    assert len(kernel) == 2
    assert "--require-hashes" in kernel[0] and "--managed-python" in kernel[0] and "LOCK_SHA256" in kernel[0]
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    assert "torch==2.14.0" in lock and "--hash=sha256:" in lock
    md = _markdown()
    assert "Restart the runtime, then rerun" not in md and "installs the pinned dependencies" not in md
    assert "`DIMER_NOTEBOOK_CI_PREINSTALLED=1`" in md and "**Linux x86_64 runtimes only**" in md


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: accelerate calls importlib.util.find_spec("google.colab"), which raises on a spec-less stub."""
    router = [s for s in _code() if "# dimer: kernel cell" in s][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


# --- MMT-m4: Colab collapses the Infrastructure cells -------------------------------------------------------------------


def test_infrastructure_cells_are_titled_and_collapsed_for_colab():
    code_cells = [c for c in _cells() if c["cell_type"] == "code"]
    setup = code_cells[:7]  # install, router, runtime record, three carried modules, model
    assert all(c["metadata"].get("cellView") == "form" for c in setup)
    assert all(_source(c).startswith("# @title Infrastructure: ") for c in setup)
    assert all(c["metadata"].get("cellView") != "form" for c in code_cells[7:])


# --- MMT-m5: both hosts are named -----------------------------------------------------------------------------------


def test_opening_and_prerequisites_name_both_hosts():
    cells = [_source(c) for c in _cells() if c["cell_type"] == "markdown"]
    opening = cells[0]
    prereq = next(s for s in cells if s.startswith("## Prerequisites"))
    access = next(line for line in prereq.splitlines() if line.startswith("- **External access:**"))
    for text in (opening, access):
        assert "huggingface.co" in text and "object.pouta.csc.fi" in text
    assert "Hub only" not in _markdown()


# --- MMT-m6: device difference stated, timings match records ------------------------------------------------------


def test_cpu_gpu_difference_and_recorded_timings():
    md = _markdown()
    assert "**CPU and GPU give slightly different adapted numbers.**" in md
    assert "59.36 / BLEU 33.75 with 189 of 300" in md and "59.56 / 33.65 with 196 changed" in md
    assert "230 s" not in md and "263 s" in md
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "263.3 s" in record and "267.5 s" in record


# --- MMT-m7: BYOD "new sentences" are labelled as test rows -----------------------------------------------------------


def test_byod_new_sentences_are_labelled_as_already_scored_test_rows():
    section12 = _cell_with("new_report = evaluation_report(")
    assert "'first six BYOD test records (already scored in Section 9; not new data)'" in section12
    assert "**With BYOD there are no unused pairs**" in _markdown()
