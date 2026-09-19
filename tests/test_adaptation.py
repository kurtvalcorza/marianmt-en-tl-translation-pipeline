"""Offline tests for the parallel-dataset contract, the pinned corpus reader, the metrics and baseline,
BYOD loaders, CSV export, artifact-manifest rejections and adapt() argument validation. Nothing here
imports torch or transformers; the corpus is a crafted zip served through an injected fetcher."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest

from marianmt_translation_pipeline import (
    ARTIFACT_FORMAT,
    CORPUS_SHA256,
    DECODER_LAYERS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    MarianMTTranslationPipeline,
    bleu,
    build_sample_dataset,
    check_split_disjoint,
    chrf,
    copy_source_baseline,
    dataset_digest,
    fetch_corpus,
    fetch_sample_dataset,
    filter_pairs,
    load_byod_dataset,
    read_corpus_pairs,
    split_dataset,
    translation_metrics,
    validate_dataset,
    write_dataset_csv,
)
from marianmt_translation_pipeline import pipeline as pl
from marianmt_translation_pipeline import samples as sm

PAIRS = [
    ("Good morning.", "Magandang umaga."),
    ("Where is the hospital?", "Nasaan ang ospital?"),
    ("The house is big.", "Malaki ang bahay."),
    ("I am tired.", "Pagod ako."),
    ("Thank you very much.", "Maraming salamat."),
    ("She is reading a book.", "Nagbabasa siya ng libro."),
    ("We will go tomorrow.", "Pupunta tayo bukas."),
    ("He does not know.", "Hindi niya alam."),
    ("It is raining.", "Umuulan."),
    ("Please sit down.", "Maupo ka."),
    ("The children are playing.", "Naglalaro ang mga bata."),
    ("Tom is a teacher.", "Guro si Tom."),
]


def _records(pairs=PAIRS, prefix="r"):
    return [{"id": f"{prefix}{i:03d}", "source": s, "target": t} for i, (s, t) in enumerate(pairs)]


def _corpus_zip(pairs=PAIRS, *, extra_lines=0):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("README", "Corpus Name: Tatoeba\n")
        archive.writestr("Tatoeba.en-tl.en", "\n".join(s for s, _ in pairs) + "\n" * (1 + extra_lines))
        archive.writestr("Tatoeba.en-tl.tl", "\n".join(t for _, t in pairs) + "\n")
    return buffer.getvalue()


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_URL.startswith("https://object.pouta.csc.fi/OPUS-Tatoeba/")
    assert len(CORPUS_SHA256) == 64 and sm.CORPUS_BYTES == 312_459
    assert sum(SAMPLE_SPLIT.values()) == 1_700 and set(SAMPLE_SPLIT) == {"train", "validation", "test"}


def test_fetch_corpus_verifies_digest_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    payload = _corpus_zip()
    monkeypatch.setattr(sm, "CORPUS_BYTES", len(payload))
    monkeypatch.setattr(sm, "CORPUS_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def fetcher(url):
        calls.append(url)
        return payload

    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == payload
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == payload
    assert calls == [sm.CORPUS_URL]
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_pairs_requires_aligned_members(forbid_model_imports):
    pairs = read_corpus_pairs(_corpus_zip())
    assert pairs[0] == ("Good morning.", "Magandang umaga.") and len(pairs) == len(PAIRS)
    with pytest.raises(ValueError, match="not aligned"):
        read_corpus_pairs(_corpus_zip(extra_lines=2))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Tatoeba.en-tl.en", "a\n")
    with pytest.raises(ValueError, match="missing member"):
        read_corpus_pairs(buffer.getvalue())


def test_filter_and_sample_split_are_seeded_and_disjoint(forbid_model_imports):
    noisy = [*PAIRS, ("good morning.", "Magandang umaga po."), ("Hi", "Oy"), ("x" * 300, "y")]
    kept = filter_pairs(noisy)
    assert len(kept) == len(PAIRS)  # duplicate source (case-insensitive), too short and too long are dropped
    sizes = {"train": 8, "validation": 2, "test": 2}
    splits = build_sample_dataset(noisy, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == sizes
    assert splits["train"][0]["id"].startswith("train-")
    assert check_split_disjoint(splits) == sizes
    assert build_sample_dataset(noisy, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(noisy, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(PAIRS, sizes={"train": 100})


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    payload = _corpus_zip()
    monkeypatch.setattr(sm, "CORPUS_BYTES", len(payload))
    monkeypatch.setattr(sm, "CORPUS_SHA256", hashlib.sha256(payload).hexdigest())
    splits = fetch_sample_dataset(
        cache_dir=tmp_path, fetcher=lambda url: payload, sizes={"train": 8, "test": 2}
    )
    assert validate_dataset(splits["train"])["n_records"] == 8


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(forbid_model_imports):
    report = validate_dataset(_records())
    assert report["n_records"] == 12 and report["unique_sources"] == 12 and report["identical_pairs"] == 0
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    with pytest.raises(ValueError, match="8..20000"):
        validate_dataset(_records()[:3])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([{**r, "id": "same"} for r in _records()])
    with pytest.raises(ValueError, match="missing 'target'"):
        validate_dataset([{"id": r["id"], "source": r["source"]} for r in _records()])
    with pytest.raises(ValueError, match="source is empty"):
        validate_dataset([{**_records()[0], "source": "   "}, *_records()[1:]])
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_dataset([{**_records()[0], "target": "x" * 5000}, *_records()[1:]])
    with pytest.raises(ValueError, match="id must match"):
        validate_dataset([{**_records()[0], "id": "bad id!"}, *_records()[1:]])
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})


def test_split_dataset_deduplicates_and_is_seeded(forbid_model_imports):
    records = _records() + [{"id": "dup", "source": "good morning.", "target": "Magandang umaga po."}]
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=0)
    assert sum(len(v) for v in splits.values()) == 12  # the case-duplicate source is dropped
    assert check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=0) == splits
    assert split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=1) != splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.6, test_fraction=0.5)
    with pytest.raises(ValueError, match="at least 8"):
        split_dataset(_records()[:9], val_fraction=0.0, test_fraction=0.5)


# --- metrics ------------------------------------------------------------------------------------------


def test_chrf_and_bleu_have_the_expected_extremes(forbid_model_imports):
    refs = [t for _, t in PAIRS]
    assert chrf(refs, refs) == pytest.approx(100.0) and bleu(refs, refs) == pytest.approx(100.0)
    assert chrf(["zzz"] * len(refs), refs) < 5.0 and bleu(["zzz"] * len(refs), refs) == 0.0
    partial = [t.split(" ")[0] for t in refs]
    assert 0.0 < chrf(partial, refs) < 100.0
    with pytest.raises(ValueError, match="equal in length"):
        chrf(["a"], ["a", "b"])
    metrics = translation_metrics(refs, refs)
    assert metrics["n"] == 12 and "sacrebleu" in metrics["definitions"]["chrf"]


def test_copy_source_baseline_scores_the_english_input(forbid_model_imports):
    baseline = copy_source_baseline(_records())
    assert baseline["baseline"].startswith("copy source") and baseline["chrf"] < 30.0
    with pytest.raises(ValueError, match="non-empty"):
        copy_source_baseline([])


# --- BYOD I/O -----------------------------------------------------------------------------------------


def test_byod_csv_json_jsonl_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records()
    path = write_dataset_csv(records, tmp_path / "data.csv")
    assert load_byod_dataset(path) == records
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.jsonl") == records
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "obj.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="array"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match=".csv, .json or .jsonl"):
        load_byod_dataset(tmp_path / "data.txt")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.csv")


# --- adaptation and artifacts (no model) ------------------------------------------------------------------


def _pipeline_without_model():
    return MarianMTTranslationPipeline(lambda texts, m, b: [(t, 1, "eos") for t in texts], lambda text: 3)


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.adapt(_records(), trainable_decoder_layers=DECODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    # evaluate() only needs the translate path, so it works with an injected runner (which echoes the source)
    metrics = pipe.evaluate(_records())
    assert metrics["n"] == 12 and metrics["chrf"] == pytest.approx(copy_source_baseline(_records())["chrf"])
    with pytest.raises(ValueError, match="batch_size"):
        pipe.evaluate(_records(), batch_size=0)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["model.decoder.layers.5.fc1.weight"],
        "adapter": {"trainable_decoder_layers": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_and_traversal(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {"trainable_decoder_layers": 1},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "other.safetensors"}]})
    with pytest.raises(ValueError, match="must name exactly"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {}})
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
