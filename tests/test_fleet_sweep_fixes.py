"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['marianmt_translation_colab']
LOCK = ROOT / 'tutorials/requirements-colab.lock.txt'
MIN_PREDICT = {'marianmt_translation_colab': 0}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


NB = NOTEBOOKS[0]


def _markdown() -> str:
    return "\n".join(c["source"] for c in _nb(NB)["cells"] if c["cell_type"] == "markdown")


def _learner_code() -> str:
    return "\n".join(c["source"] for c in _code_cells(_nb(NB)) if not c["metadata"].get("dimer", {}).get("embedded_module"))


# --- SWP-R (prose) and SWP-G: the existing guided layer now describes the isolated runtime ------------------------


def test_swp_r_learner_text_no_longer_describes_an_in_kernel_install():
    markdown = _markdown()
    assert "Restart the runtime" not in markdown and "keeps the NumPy that Colab has already loaded" not in markdown
    assert "completes in one pass" in markdown
    assert "Section 1 stops with *This notebook needs a Linux x86_64 runtime*" in markdown


def test_swp_r_infrastructure_cells_stay_collapsed():
    cells = _code_cells(_nb(NB))
    kernel = next(c for c in cells if "# dimer: kernel cell" in c["source"])
    assert kernel["metadata"].get("cellView") == "form" and kernel["metadata"].get("jupyter", {}).get("source_hidden") is True
    record = next(c for c in cells if c["source"].startswith("# @title Infrastructure: record the runtime"))
    assert record["metadata"].get("jupyter", {}).get("source_hidden") is True


def test_swp_g_guided_layer_is_complete():
    markdown = _markdown()
    for marker in (
        "**Who this notebook is for.**",
        "The intended audience is",
        "### How to use this notebook",
        "## The task: Input → Model → Output",
        "## Roadmap",
        "## Troubleshooting",
        "## Glossary",
        "check your reasoning",
        "Sample answer",
    ):
        assert marker in markdown, marker
    assert markdown.count("**Checkpoint") >= 5


def test_swp_g_no_template_placeholders_leak():
    text = "\n".join(c["source"] for c in _nb(NB)["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:", "<<"):
        assert leftover not in text, leftover


# --- SWP-A: quality outcomes are recorded verdicts, never asserts -------------------------------------------------


def test_swp_a_no_quality_assert_remains():
    code = _learner_code()
    quality = [line for line in code.splitlines() if line.strip().startswith("assert ") and re.search(r"chrf|bleu", line)]
    assert quality == []
    assert "assert parity['identical_translations'] == parity['of']" in code  # contract integrity stays hard


def test_swp_a_negative_results_are_recorded_and_do_not_stop_the_notebook():
    s7 = _cell(_nb(NB), "frozen_test = pipe.evaluate(test_records")
    frozen_line = next(line for line in s7.splitlines() if line.startswith("frozen_verdict = "))
    s9 = _cell(_nb(NB), "adapted_test = pipe.evaluate(test_records")
    block = s9[s9.index("delta_chrf = ") : s9.index("for metric, row in comparison.items():")]
    for frozen, copy, adapted, expected in ((20.0, 30.0, 19.0, "worse"), (45.0, 20.0, 45.0, "no gain"), (45.0, 20.0, 47.5, "improved")):
        ns = {"frozen_test": {"chrf": frozen}, "baseline_copy": {"chrf": copy}, "adapted_test": {"chrf": adapted}, "comparison": {}}
        exec(frozen_line, ns)
        exec(block, ns)
        assert ns["comparison"]["verdicts"]["adapted_vs_pretrained_chrf"] == expected
        assert ns["comparison"]["verdicts"]["pretrained_vs_copy_source"] == ("above the copy-source baseline" if frozen > copy else "not above the copy-source baseline")
    assert s9.index("comparison['verdicts'] = ") < s9.index("json.dump(evaluation_report_payload")
    assert "if adaptation_verdict != 'improved':" in s9


# --- SWP-F: Sections 6-8 always use the pretrained model ---------------------------------------------------------


def test_swp_f_frozen_pipeline_reloads_an_adapted_pipeline(capsys):
    s6 = _cell(_nb(NB), "def frozen_pipeline():")
    helper = s6[s6.index("def frozen_pipeline():") : s6.index("\n\n\nfrozen_pipeline()")]
    loads = []

    class Stand:
        @staticmethod
        def from_pretrained(weights_dir):
            loads.append(weights_dir)
            return types.SimpleNamespace(adapter=None)

    ns = {"pipe": types.SimpleNamespace(adapter={"best_epoch": 2}), "MarianMTTranslationPipeline": Stand, "WEIGHTS_DIR": "w"}
    exec(helper, ns)
    ns["frozen_pipeline"]()
    assert loads == ["w"] and ns["pipe"].adapter is None
    assert "Reloaded the pretrained model" in capsys.readouterr().out
    ns["frozen_pipeline"]()
    assert loads == ["w"]


def test_swp_f_pretrained_cells_call_frozen_pipeline_first():
    for marker, first_use in (
        ("result = pipe.translate(texts", "result = pipe.translate("),
        ("pretrained_outputs = translate_all(test_records)", "pretrained_outputs = translate_all("),
        ("frozen_test = pipe.evaluate(test_records", "frozen_test = pipe.evaluate("),
        ("adapt_result = pipe.adapt(", "adapt_result = pipe.adapt("),
    ):
        source = _cell(_nb(NB), marker)
        assert "frozen_pipeline()" in source and source.index("frozen_pipeline()") < source.index(first_use), marker


# --- SWP-B: the upload fallback is guarded, refusals name the file ------------------------------------------------


def _byod(monkeypatch, path: str, tmp_path) -> dict:
    from marianmt_translation_pipeline.samples import load_byod_dataset

    source = _cell(_nb(NB), "BYOD_PATH = ''")
    block = source[source.index("if USE_BYOD:\n") : source.index("    splits = split_dataset(records, seed=SPLIT_SEED)")]
    monkeypatch.chdir(tmp_path)
    ns = {"Path": Path, "USE_BYOD": True, "BYOD_PATH": path, "load_byod_dataset": load_byod_dataset}
    exec(compile(block, "<section 4 BYOD>", "exec"), ns)
    return ns


def test_swp_b_byod_path_works_and_refusals_name_the_file(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    good = tmp_path / "pairs.csv"
    good.write_text("id,source,target\na,Hello.,Kumusta.\n", encoding="utf-8")
    ns = _byod(monkeypatch, str(good), tmp_path)
    assert ns["file_name"] == "pairs.csv" and ns["records"] == [{"id": "a", "source": "Hello.", "target": "Kumusta."}]
    with pytest.raises(FileNotFoundError, match="BYOD_PATH .*missing.csv.* is not a file"):
        _byod(monkeypatch, str(tmp_path / "missing.csv"), tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("id,text\na,Hello.\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^bad\.csv: CSV is missing columns"):
        _byod(monkeypatch, str(bad), tmp_path)
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty and this runtime has no Colab upload dialog"):
        _byod(monkeypatch, "", tmp_path)


def test_swp_b_cancelled_or_multiple_uploads_are_refused(monkeypatch, tmp_path):
    _colab(monkeypatch, lambda: {})
    with pytest.raises(RuntimeError, match="got 0 .*upload cancelled or empty"):
        _byod(monkeypatch, "", tmp_path)
    _colab(monkeypatch, lambda: {"a.csv": b"", "b.csv": b""})
    with pytest.raises(RuntimeError, match="got 2"):
        _byod(monkeypatch, "", tmp_path)
    _colab(monkeypatch, lambda: {"up.jsonl": b'{"id": "x", "source": "Hi.", "target": "Kumusta."}\n'})
    ns = _byod(monkeypatch, "", tmp_path)
    assert ns["file_name"] == "up.jsonl" and ns["records"][0]["id"] == "x"
