"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a
referenced evaluation, a one-epoch adaptation of the last decoder layer on a dozen pairs, and the
artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import json

import pytest

from marianmt_translation_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, MarianMTTranslationPipeline

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

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
RECORDS = [{"id": f"p{i:02d}", "source": s, "target": t} for i, (s, t) in enumerate(PAIRS)]


@pytest.fixture(scope="module")
def pipe():
    return MarianMTTranslationPipeline.from_pretrained(device="cpu")


def test_evaluate_scores_references(pipe):
    metrics = pipe.evaluate(RECORDS[:4], num_beams=1)
    assert metrics["n"] == 4 and 0.0 < metrics["chrf"] <= 100.0 and metrics["adapted"] is False


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(RECORDS[:8], RECORDS[8:], epochs=1, trainable_decoder_layers=1, batch_size=4)
    assert result["n_trainable"] == 4_204_032 and result["history"][0]["note"] == "frozen model"
    assert all(name.startswith("model.decoder.layers.5.") for name in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"])
    reloaded = MarianMTTranslationPipeline.from_artifact(artifact, device="cpu")
    a = pipe.translate([r["source"] for r in RECORDS[:3]], num_beams=1)["translations"]
    b = reloaded.translate([r["source"] for r in RECORDS[:3]], num_beams=1)["translations"]
    assert [x["text"] for x in a] == [x["text"] for x in b]
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
