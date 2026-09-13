import hashlib
import json
import re
from pathlib import Path

import pytest

from marianmt_translation_pipeline import (
    DECISION_RULE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_BEAMS,
    DEFAULT_WEIGHTS_DIR,
    MAX_BATCH,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MAX_NUM_BEAMS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    WEIGHT_FILE,
    MarianMTTranslationPipeline,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _fake_count(text: str) -> int:
    return len(text.split()) + 1  # words + EOS, stands in for the SentencePiece count


def _fake_runner(texts: list[str], max_new_tokens: int, num_beams: int) -> list[tuple[str, int, str]]:
    out = []
    for text in texts:
        words = text.split()[:max_new_tokens]
        out.append(
            (
                " ".join(w.upper() for w in words),
                len(words),
                "eos" if len(words) < max_new_tokens else "max_new_tokens",
            )
        )
    return out


def _pipeline() -> MarianMTTranslationPipeline:
    return MarianMTTranslationPipeline(_fake_runner, _fake_count, "cpu", "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / WEIGHT_FILE).write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": WEIGHT_FILE, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "Helsinki-NLP/opus-mt-en-tl"
    assert MODEL_LICENSE == "apache-2.0"
    assert WEIGHT_FILE == "pytorch_model.bin"
    assert DEFAULT_WEIGHTS_DIR.name == MODEL_KEY
    assert DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert MAX_INPUT_TOKENS == 512
    assert 1 <= DEFAULT_MAX_NEW_TOKENS <= MAX_NEW_TOKENS
    assert 1 <= DEFAULT_NUM_BEAMS <= MAX_NUM_BEAMS


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY
    paths = {entry["path"] for entry in manifest["files"]}
    assert WEIGHT_FILE in paths
    assert not any(p.endswith(".safetensors") for p in paths)  # the pickle is the only weight file upstream


def test_ceilings_and_defaults_match_local_config_when_present():
    config_path = DEFAULT_WEIGHTS_DIR / "config.json"
    gen_path = DEFAULT_WEIGHTS_DIR / "generation_config.json"
    tok_path = DEFAULT_WEIGHTS_DIR / "tokenizer_config.json"
    if not (config_path.is_file() and gen_path.is_file() and tok_path.is_file()):
        pytest.skip("local snapshot config not staged")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    generation = json.loads(gen_path.read_text(encoding="utf-8"))
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    assert config["architectures"] == ["MarianMTModel"]
    assert config["max_position_embeddings"] == MAX_INPUT_TOKENS
    assert generation["max_length"] == MAX_NEW_TOKENS
    assert generation["num_beams"] == DEFAULT_NUM_BEAMS
    assert (tok["source_lang"], tok["target_lang"]) == ("en", "tl")


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    _write_snapshot(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_tampered_bytes_and_missing_file(tmp_path: Path):
    """A modified pickle never reaches torch.load: the digest check fails first."""
    _write_snapshot(tmp_path)
    (tmp_path / WEIGHT_FILE).write_bytes(b"weightz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    (tmp_path / WEIGHT_FILE).write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / WEIGHT_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        MarianMTTranslationPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path: Path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": WEIGHT_FILE, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == [WEIGHT_FILE]
    assert fetched == [WEIGHT_FILE]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path: Path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_translate_rejects_bad_inputs():
    pipe = _pipeline()
    with pytest.raises(TypeError, match="not a single string"):
        pipe.translate("a bare string")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="MAX_BATCH"):
        pipe.translate([])
    with pytest.raises(ValueError, match="MAX_BATCH"):
        pipe.translate(["x"] * (MAX_BATCH + 1))
    with pytest.raises(TypeError, match=r"texts\[1\] must be str"):
        pipe.translate(["ok", b"bytes"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match=r"texts\[0\] is empty"):
        pipe.translate(["   "])
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.translate(["x" * (MAX_TEXT_CHARS + 1)])
    with pytest.raises(ValueError, match=r"texts\[1\] is .* MAX_INPUT_TOKENS"):
        pipe.translate(
            ["short", " ".join(["w"] * MAX_INPUT_TOKENS)]
        )  # MAX_INPUT_TOKENS words + EOS > ceiling
    with pytest.raises(ValueError, match="max_new_tokens"):
        pipe.translate(["ok"], max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        pipe.translate(["ok"], max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(TypeError):
        pipe.translate(["ok"], max_new_tokens=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pipe.translate(["ok"], num_beams=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_beams"):
        pipe.translate(["ok"], num_beams=0)
    with pytest.raises(ValueError, match="num_beams"):
        pipe.translate(["ok"], num_beams=MAX_NUM_BEAMS + 1)


def test_translate_output_fields_and_order():
    pipe = _pipeline()
    result = pipe.translate(["The house is wonderful.", "Good morning to all of you"], max_new_tokens=3)
    assert result["n"] == 2
    assert result["direction"] == "en->tl"
    first, second = result["translations"]
    assert first == {
        "source": "The house is wonderful.",
        "text": "THE HOUSE IS",
        "input_tokens": 5,  # 4 words + EOS under the fake counter
        "generated_tokens": 3,
        "stopped_by": "max_new_tokens",
    }
    assert second["source"] == "Good morning to all of you"
    assert second["input_tokens"] == 7
    assert result["generation"] == {
        "max_new_tokens": 3,
        "num_beams": DEFAULT_NUM_BEAMS,
        "do_sample": False,
        "decision_rule": DECISION_RULE,
    }
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (result["device"], result["source"]) == ("cpu", "injected")
    default = pipe.translate(["Good morning."])
    assert default["translations"][0]["stopped_by"] == "eos"
    assert default["generation"]["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS


def test_translate_accepts_ceiling_boundaries():
    pipe = _pipeline()
    result = pipe.translate(
        [" ".join(["w"] * (MAX_INPUT_TOKENS - 1))] * MAX_BATCH,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=MAX_NUM_BEAMS,
    )
    assert result["n"] == MAX_BATCH
    assert all(item["input_tokens"] == MAX_INPUT_TOKENS for item in result["translations"])
    assert result["generation"]["num_beams"] == MAX_NUM_BEAMS
