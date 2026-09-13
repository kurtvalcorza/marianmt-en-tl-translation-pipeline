"""English-to-Tagalog machine translation with the pinned ``Helsinki-NLP/opus-mt-en-tl`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/opus-mt-en-tl/``) or, when
explicitly allowed, from the Hugging Face Hub at the pinned revision. The upstream snapshot ships its
weights as ``pytorch_model.bin`` — a pickle, not SafeTensors — so the trust boundary is the SHA-256 in the
manifest (checked before the load) plus weights_only=True deserialization in ``transformers``. One
task method, ``translate``: a batch of English strings in, one Tagalog string per input out.
Direction is EN -> TL only; the checkpoint has no reverse direction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "Helsinki-NLP/opus-mt-en-tl"
MODEL_REVISION = "e46e1761492cb6a6fb9515a72bb55ca654815ca5"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "opus-mt-en-tl"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHT_FILE = "pytorch_model.bin"  # the only upstream weight file at this revision: a pickle, digest-pinned

SOURCE_LANG = "en"  # tokenizer_config.json source_lang
TARGET_LANG = "tl"  # tokenizer_config.json target_lang
MAX_INPUT_TOKENS = 512  # max_position_embeddings in the snapshot config.json; longer inputs rejected, not cut
MAX_NEW_TOKENS = 512  # ceiling on decoder steps per call (config.json / generation_config.json max_length)
DEFAULT_MAX_NEW_TOKENS = 128
MAX_TEXT_CHARS = 4_000  # pre-tokenisation guard per input string
MAX_BATCH = 16  # texts per translate() call
MAX_NUM_BEAMS = 8
DEFAULT_NUM_BEAMS = 4  # num_beams in the snapshot generation_config.json
DECISION_RULE = (
    "beam search over whole sequences (num_beams=4 by default, from generation_config.json); greedy argmax "
    "per step when num_beams=1; no sampling; decoding stops at </s> or max_new_tokens"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


INPUT_SCHEMA: dict[str, Any] = {
    "input": "sequence of non-empty English str (a sentence or short passage each); one Tagalog str each",
    "direction": f"{SOURCE_LANG}->{TARGET_LANG} only",
    "batch": [1, MAX_BATCH],
    "text_chars": [1, MAX_TEXT_CHARS],
    "input_tokens": [1, MAX_INPUT_TOKENS],
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "num_beams": [1, MAX_NUM_BEAMS],
    "decision_rule": DECISION_RULE,
    "preprocessing": (
        "Marian normalisation + SentencePiece encoding with source.spm (no truncation: an input over "
        "MAX_INPUT_TOKENS is rejected with a ValueError naming the count, never cut); batch padded to the "
        "longest input"
    ),
}


def _check_inputs(texts: Any, max_new_tokens: Any, num_beams: Any) -> list[str]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the texts as a list.

    The encoder-token ceiling is not checked here because it needs the loaded tokenizer;
    ``_check_input_tokens`` applies it inside the pipeline once the counts are known.
    """
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of str, not a single string")
    if not 1 <= len(texts) <= MAX_BATCH:
        raise ValueError(f"texts must hold 1..MAX_BATCH={MAX_BATCH} items, got {len(texts)}")
    clean = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{i}] must be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError(f"texts[{i}] is empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"texts[{i}] has {len(text)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}")
        clean.append(text)
    for name, value, ceiling in (
        ("max_new_tokens", max_new_tokens, MAX_NEW_TOKENS),
        ("num_beams", num_beams, MAX_NUM_BEAMS),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
        if not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be between 1 and {ceiling}, got {value}")
    return clean


def _check_input_tokens(counts: Sequence[int]) -> list[int]:
    """The encoder-token ceiling, applied once the tokenizer has counted every input."""
    for i, n_input in enumerate(counts):
        if n_input > MAX_INPUT_TOKENS:
            raise ValueError(
                f"texts[{i}] is {n_input} tokens; ceiling is MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}"
            )
    return list(counts)


def validate_inputs(
    texts: Sequence[str],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = DEFAULT_NUM_BEAMS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``translate`` would: both route through
    ``_check_inputs``. The encoder-token ceiling (``MAX_INPUT_TOKENS``) needs the loaded tokenizer and
    is enforced inside ``translate``, which reports ``input_tokens`` per item.
    """
    checked = _check_inputs(texts, max_new_tokens, num_beams)
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per text")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[i] if names else f"input{i:02d}", "chars": len(text), "words": len(text.split())}
            for i, text in enumerate(checked)
        ],
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], references: Sequence[str] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even though no metric exists here.

    The repository ships no metric helper, so the verdict is always ``not-measurable`` (EVAL9).
    ``references`` exists for interface parity with the fleet's other pipelines and is recorded in
    ``reason`` rather than scored: BLEU/chrF need a scorer and enough referenced sentences to state a
    dispersion, and manufacturing a number from a proxy such as length ratio would misrepresent a
    plumbing check as a quality measurement.
    """
    items = result.get("translations", [])
    rule = result.get("generation", {}).get("decision_rule", DECISION_RULE)
    supplied = references is not None
    return {
        "task": f"machine translation {SOURCE_LANG}->{TARGET_LANG}",
        "score_semantics": (
            "the pipeline emits no probability, confidence or score: generated_tokens, input_tokens and "
            f"stopped_by are counts and flags, and {rule} "
            "produces some token at every step with no minimum-probability cut-off and no shipped acceptance "
            "threshold"
        ),
        "sample_kind": sample_kind,
        "n_inputs": len(items),
        "n_generated_tokens": int(sum(int(item.get("generated_tokens", 0)) for item in items)),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": (
            "the repository ships no metric helper and a translation has no ground truth here"
            + (
                "; references were supplied but no metric helper exists to score them, and a handful of "
                "references is not a dispersion"
                if supplied
                else "; the evaluated sample has no reference translations"
            )
        ),
        "needs": (
            "reference Tagalog translations from the deployment domain, one or more per source sentence, "
            "over "
            "enough sentences to state a dispersion, scored with the caller's own BLEU/chrF implementation "
            "(the upstream README reports BLEU 26.6 / chrF 0.577 on Tatoeba.en.tl — an upstream claim, not "
            "measured here), excluding or re-running outputs whose stopped_by is max_new_tokens; no proxy "
            "such as length ratio substitutes for that"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@dataclass
class MarianMTTranslationPipeline:
    """``_runner(texts, max_new_tokens, num_beams)`` -> list of ``(translation, generated_tokens,
    stopped_by)``, one per input; ``_count_tokens(text)`` -> encoder token count incl. EOS. Both
    injectable so tests run offline."""

    _runner: Callable[[list[str], int, int], list[tuple[str, int, str]]]
    _count_tokens: Callable[[str], int]
    device: str = "cpu"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> MarianMTTranslationPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, kwargs, source = str(root), dict(local_files_only=True), "local-snapshot"
        elif allow_download:
            location, kwargs, source = MODEL_ID, dict(revision=MODEL_REVISION), "hf-hub"
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = MarianTokenizer.from_pretrained(location, trust_remote_code=False, **kwargs)
        # Trust boundary (MOD12): the upstream weight file is a pickle (pytorch_model.bin). Its SHA-256 was
        # checked against the manifest above; use_safetensors=False names that fact, and weights_only=True
        # makes transformers deserialise with weights_only=True, which refuses arbitrary objects.
        model = MarianMTModel.from_pretrained(
            location,
            dtype=torch.float32,
            trust_remote_code=False,
            use_safetensors=False,
            weights_only=True,
            **kwargs,
        )
        model = model.to(resolved_device).eval()
        eos_id, pad_id = model.config.eos_token_id, model.config.pad_token_id

        def count_tokens(text: str) -> int:
            return len(tokenizer(text, truncation=False)["input_ids"])

        def runner(texts: list[str], max_new_tokens: int, num_beams: int) -> list[tuple[str, int, str]]:
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(resolved_device)
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False
                )
            results = []
            for row in out.tolist():
                content = [t for t in row if t not in (eos_id, pad_id)]
                stopped_by = "eos" if eos_id in row else "max_new_tokens"
                results.append(
                    (tokenizer.decode(content, skip_special_tokens=True), len(content), stopped_by)
                )
            return results

        return cls(runner, count_tokens, resolved_device, source)

    def _validate(self, texts: Any, max_new_tokens: Any, num_beams: Any) -> tuple[list[str], list[int]]:
        clean = _check_inputs(texts, max_new_tokens, num_beams)
        return clean, _check_input_tokens([self._count_tokens(text) for text in clean])

    def translate(
        self,
        texts: Sequence[str],
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> dict[str, Any]:
        """Translate a batch of English texts to Tagalog; one output per input, in order."""
        clean, counts = self._validate(texts, max_new_tokens, num_beams)
        generated = self._runner(clean, max_new_tokens, num_beams)
        if len(generated) != len(clean):
            raise RuntimeError(f"runner returned {len(generated)} outputs for {len(clean)} inputs")
        translations = []
        for text, n_input, (output, n_generated, stopped_by) in zip(clean, counts, generated, strict=True):
            if not isinstance(output, str) or not isinstance(n_generated, int):
                raise RuntimeError("runner must return (str, int, str) per input")
            translations.append(
                {
                    "source": text,
                    "text": output,
                    "input_tokens": n_input,
                    "generated_tokens": n_generated,
                    "stopped_by": stopped_by,
                }
            )
        return {
            "translations": translations,
            "n": len(translations),
            "direction": f"{SOURCE_LANG}->{TARGET_LANG}",
            "generation": {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "do_sample": False,
                "decision_rule": DECISION_RULE,
            },
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
