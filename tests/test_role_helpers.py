"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from marianmt_translation_pipeline import (
    DECISION_RULE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_BEAMS,
    INPUT_SCHEMA,
    MAX_BATCH,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MAX_NUM_BEAMS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

ONE = "The house is wonderful."
TWO = "Good morning to all of you."


def _result(generated: tuple[int, ...] = (6, 7), num_beams: int = DEFAULT_NUM_BEAMS) -> dict:
    return {
        "translations": [
            {"source": src, "text": "…", "input_tokens": 5, "generated_tokens": n, "stopped_by": "eos"}
            for src, n in zip((ONE, TWO), generated, strict=True)
        ],
        "n": 2,
        "direction": "en->tl",
        "generation": {
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "num_beams": num_beams,
            "do_sample": False,
            "decision_rule": DECISION_RULE,
        },
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([ONE, TWO], names=["input00", "input01"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["direction"] == "en->tl only"
    assert manifest["schema"]["batch"] == [1, MAX_BATCH]
    assert manifest["schema"]["text_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["input_tokens"] == [1, MAX_INPUT_TOKENS]
    assert manifest["schema"]["max_new_tokens"] == [1, MAX_NEW_TOKENS]
    assert manifest["schema"]["num_beams"] == [1, MAX_NUM_BEAMS]
    assert manifest["schema"]["decision_rule"] == DECISION_RULE
    assert manifest["inputs"] == [
        {"id": "input00", "chars": len(ONE), "words": 4},
        {"id": "input01", "chars": len(TWO), "words": 6},
    ]
    assert (manifest["max_new_tokens"], manifest["num_beams"]) == (DEFAULT_MAX_NEW_TOKENS, DEFAULT_NUM_BEAMS)
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_and_settings() -> None:
    manifest = validate_inputs([ONE], max_new_tokens=8, num_beams=1)
    assert [entry["id"] for entry in manifest["inputs"]] == ["input00"]
    assert (manifest["max_new_tokens"], manifest["num_beams"]) == (8, 1)


def test_validate_inputs_rejects_like_translate() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs(ONE)
    with pytest.raises(ValueError, match="MAX_BATCH"):
        validate_inputs([])
    with pytest.raises(ValueError, match="MAX_BATCH"):
        validate_inputs(["x"] * (MAX_BATCH + 1))
    with pytest.raises(TypeError, match=r"texts\[0\] must be str"):
        validate_inputs([None])  # type: ignore[list-item]
    with pytest.raises(ValueError, match=r"texts\[1\] is empty"):
        validate_inputs([ONE, "   "])
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_inputs(["x" * (MAX_TEXT_CHARS + 1)])
    with pytest.raises(TypeError, match="max_new_tokens must be an int"):
        validate_inputs([ONE], max_new_tokens=True)
    with pytest.raises(ValueError, match="num_beams must be between"):
        validate_inputs([ONE], num_beams=MAX_NUM_BEAMS + 1)
    with pytest.raises(ValueError, match="names must have one entry per text"):
        validate_inputs([ONE], names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_inputs"] == 2
    assert report["n_generated_tokens"] == 13
    assert report["sample_kind"] == "synthetic"
    assert report["task"] == "machine translation en->tl"
    assert "no reference translations" in report["reason"]
    assert "BLEU/chrF" in report["needs"]
    assert DECISION_RULE in report["score_semantics"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_stays_not_measurable_when_references_are_supplied() -> None:
    report = evaluation_report(
        _result((3, 4), num_beams=1), ["Maganda ang bahay.", "Magandang umaga."], sample_kind="BYOD upload"
    )
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["sample_kind"] == "BYOD upload"
    assert report["n_generated_tokens"] == 7
    assert "a handful of references is not a dispersion" in report["reason"]
