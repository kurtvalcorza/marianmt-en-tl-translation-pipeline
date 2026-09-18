"""English-to-Tagalog machine translation with the pinned ``Helsinki-NLP/opus-mt-en-tl`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/opus-mt-en-tl/``) or, when
explicitly allowed, from the Hugging Face Hub at the pinned revision. The upstream snapshot ships its
weights as ``pytorch_model.bin`` — a pickle, not SafeTensors — so the trust boundary is the SHA-256 in the
manifest (checked before the load) plus weights_only=True deserialization in ``transformers``. Two
task methods: ``translate`` — a batch of English strings in, one Tagalog string per input out — and
``adapt`` — bounded supervised fine-tuning of the last decoder layers on a validated parallel dataset,
with ``evaluate`` (chrF / BLEU against references, see ``metrics.py``) and a safetensors adapter artifact
that reloads against the pinned base. Direction is EN -> TL only; the checkpoint has no reverse direction.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
WEIGHT_SHA256 = (
    "e418d573a717b2c81eaa3a80c8eb68f203c14c6b734dd36d41708c1369d0eb44"  # manifest digest of WEIGHT_FILE
)
PARAMETER_COUNT = 74_037_760
DECODER_LAYERS = 6  # config.json decoder_layers
DEFAULT_TRAINABLE_DECODER_LAYERS = 2  # the last two decoder blocks (8,408,064 parameters)
MAX_TRAIN_TOKENS = 128  # source/target truncation ceiling during adaptation (never at inference)
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_REFERENCES = 50  # below this a referenced score is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.marianmt-en-tl.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


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

    Without ``references`` the verdict is ``not-measurable`` (EVAL9): manufacturing a number from a
    proxy such as length ratio would misrepresent a plumbing check as a quality measurement. With
    references — one per translation — chrF and BLEU are computed by ``metrics.py`` and the verdict is
    ``measured`` (``measured-small-sample`` below ``MIN_SCORED_REFERENCES`` items, which carries no
    dispersion estimate).
    """
    items = result.get("translations", [])
    rule = result.get("generation", {}).get("decision_rule", DECISION_RULE)
    supplied = references is not None
    if supplied:
        from .metrics import translation_metrics

        if len(references) != len(items):
            raise ValueError("references must have one entry per translation")
        scored = translation_metrics(
            [str(item.get("text", "")) for item in items], [str(r) for r in references]
        )
        return {
            "task": f"machine translation {SOURCE_LANG}->{TARGET_LANG}",
            "score_semantics": (
                "chrF and BLEU are corpus-level agreement with the supplied references (own "
                "implementations, see "
                "metrics.py); the pipeline itself emits no probability, confidence or score, and "
                f"{rule} produces some token at every step with no acceptance threshold"
            ),
            "sample_kind": sample_kind,
            "n_inputs": len(items),
            "n_generated_tokens": int(sum(int(item.get("generated_tokens", 0)) for item in items)),
            "metrics": [{"name": "chrf", "value": scored["chrf"]}, {"name": "bleu", "value": scored["bleu"]}],
            "baselines": [],
            "verdict": "measured" if len(items) >= MIN_SCORED_REFERENCES else "measured-small-sample",
            "reason": (
                f"scored {len(items)} referenced sentences; "
                + (
                    "a sample this small carries no dispersion estimate"
                    if len(items) < MIN_SCORED_REFERENCES
                    else "one seeded holdout, no dispersion estimate"
                )
            ),
            "needs": "references from the deployment domain over enough sentences to state a dispersion",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
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
            "a translation has no ground truth here; the evaluated sample has no reference translations"
        ),
        "needs": (
            "reference Tagalog translations from the deployment domain, one or more per source sentence, "
            "over enough sentences to state a dispersion, passed as `references` (scored with the chrF/BLEU "
            "helpers in metrics.py) or through evaluate() on {id, source, target} records; the upstream "
            "README reports BLEU 26.6 / chrF 0.577 on Tatoeba.en.tl — an upstream claim reproduced only as "
            "far as this repository's own Tatoeba split allows; exclude or re-run outputs whose stopped_by "
            "is max_new_tokens; no proxy such as length ratio substitutes for references"
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
    adapter: dict[str, Any] | None = None
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

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

        return cls(runner, count_tokens, resolved_device, source, _model=model, _tokenizer=tokenizer)

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

    # ---- adaptation -----------------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._tokenizer

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = MAX_BATCH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> dict[str, Any]:
        """Translate every record's source and score the outputs against its target (chrF, BLEU)."""
        from .metrics import translation_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        if not isinstance(batch_size, int) or not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be an int in 1..{MAX_BATCH}")
        started = time.perf_counter()
        hypotheses: list[str] = []
        truncated = 0
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            result = self.translate(
                [r["source"] for r in batch], max_new_tokens=max_new_tokens, num_beams=num_beams
            )
            for item in result["translations"]:
                hypotheses.append(item["text"])
                truncated += item["stopped_by"] == "max_new_tokens"
        metrics = translation_metrics(hypotheses, [r["target"] for r in checked])
        metrics.update(
            {
                "hit_token_ceiling": truncated,
                "generation": {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": False},
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trainable_decoder_layers: int) -> list[str]:
        if (
            not isinstance(trainable_decoder_layers, int)
            or not 1 <= trainable_decoder_layers <= DECODER_LAYERS
        ):
            raise ValueError(f"trainable_decoder_layers must be an int in 1..{DECODER_LAYERS}")
        model, _ = self._require_model()
        first = DECODER_LAYERS - trainable_decoder_layers
        prefixes = tuple(f"model.decoder.layers.{k}." for k in range(first, DECODER_LAYERS))
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 2,
        lr: float = 1e-4,
        batch_size: int = 16,
        trainable_decoder_layers: int = DEFAULT_TRAINABLE_DECODER_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning on a validated parallel dataset.

        Only the last `trainable_decoder_layers` decoder blocks train (2 by default: 8,408,064 of
        74,037,760 parameters; the encoder, the shared embeddings and the earlier decoder blocks stay
        frozen). Teacher-forced cross-entropy on the Tagalog target (label smoothing 0), AdamW at a fixed
        learning rate with gradient clipping at 1.0, sources and targets truncated to MAX_TRAIN_TOKENS
        SentencePiece pieces **during training only**. Epoch 0 records the frozen model's validation chrF;
        the epoch with the highest validation chrF is kept."""
        from .samples import validate_dataset

        if not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        names = self._trainable_names(trainable_decoder_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        import torch

        torch.manual_seed(seed)
        model, tokenizer = self._require_model()
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = torch.device(self.device)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {
                k: v
                for k, v in self.evaluate(val_checked).items()
                if k in ("chrf", "bleu", "n", "hit_token_ceiling")
            }

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_chrf = entry["val"]["chrf"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_checked), generator=generator).tolist()
            losses = []
            for start in range(0, len(order), batch_size):
                batch = [train_checked[i] for i in order[start : start + batch_size]]
                encoded = tokenizer(
                    [r["source"] for r in batch],
                    text_target=[r["target"] for r in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_TRAIN_TOKENS,
                )
                labels = encoded["labels"].clone()
                labels[labels == tokenizer.pad_token_id] = -100
                out = model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    labels=labels.to(device),
                )
                optimiser.zero_grad(set_to_none=True)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                losses.append(float(out.loss.detach()))
            model.eval()
            entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
            history.append(entry)
            if progress:
                progress(entry)
            current = entry["val"]["chrf"] if entry["val"] else math.inf
            if current > best_chrf or not entry["val"]:
                best_chrf = current
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                best_epoch = epoch
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_decoder_layers": trainable_decoder_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation chrF" if val_checked else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "max_train_tokens": MAX_TRAIN_TOKENS,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted decoder tensors as safetensors with a manifest naming the pinned base."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest and digest, then overwrite exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        entry = manifest["files"][0]
        weights_path = root / entry["path"]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != manifest["tensors"]:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith("model.decoder.layers."):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable decoder tensor of the base model"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> MarianMTTranslationPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
