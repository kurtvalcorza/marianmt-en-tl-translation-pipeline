# MarianMT EN→TL Translation Pipeline

DIMER-oriented inference and fine-tuning wrapper for **Helsinki-NLP/opus-mt-en-tl** (OPUS-MT Marian transformer, English → Tagalog, Apache-2.0), pinned to an immutable Hugging Face revision. The repository exposes batched English-to-Tagalog translation — direction EN→TL only — with beam search by default (the upstream `generation_config.json` setting), a supply-chain check of the local weight snapshot that matters more than usual here because the upstream weight file is a **pickle** (`pytorch_model.bin`, no SafeTensors upstream), machine-readable provenance, and a bounded adaptation contract: a digest-pinned real parallel corpus (Tatoeba en–tl via OPUS), corpus chrF and BLEU with a copy-source baseline, supervised fine-tuning of the last decoder blocks with validation-chrF epoch selection, and a safetensors adapter that reloads onto the digest-verified base.

## Upstream alignment

- Model: `Helsinki-NLP/opus-mt-en-tl`
- Revision: `e46e1761492cb6a6fb9515a72bb55ca654815ca5`
- Upstream weight license: Apache-2.0
- Upstream task: machine translation, source `en`, target `tl` (`tokenizer_config.json`); `transformer-align` model trained on the `opus+bt` dataset, SentencePiece pre-processing (pinned README)
- Repository adaptation: bounded supervised fine-tuning of the last *k* decoder blocks (`adapt`, default 2 of 6 = 8,408,064 of 74,037,760 parameters) on caller-supplied or pinned Tatoeba pairs; the encoder, embeddings and vocabulary are never modified; the adapter carries only the trained tensors and is bound to the base pickle's SHA-256

## Quick start

```python
from marianmt_translation_pipeline import MarianMTTranslationPipeline

pipe = MarianMTTranslationPipeline.from_pretrained()   # verifies weights/opus-mt-en-tl first (incl. the pickle's SHA-256)
result = pipe.translate(["The house is wonderful.", "Where is the nearest hospital?"])
for item in result["translations"]:
    print(item["source"], "->", item["text"])   # CPU smoke: 'Ang bahay ay kahanga - hanga.' / 'Nasaan ang pinakamalapit na ospital?'
```

Adaptation on the pinned Tatoeba sample (CPU, about two minutes of model time after the snapshot is staged):

```python
from marianmt_translation_pipeline import (
    MarianMTTranslationPipeline, copy_source_baseline, fetch_sample_dataset, check_split_disjoint,
)

splits = fetch_sample_dataset()            # one pinned 312 KB zip from OPUS, digest-verified, cached under weights/tatoeba-en-tl/
check_split_disjoint(splits)               # 1,200 / 200 / 300 records, no source shared between splits
pipe = MarianMTTranslationPipeline.from_pretrained()
print(copy_source_baseline(splits['test'])['chrf'], pipe.evaluate(splits['test'])['chrf'])   # 11.28, 56.46 in the recorded run
pipe.adapt(splits['train'], splits['validation'])                                            # last 2 decoder blocks, 2 epochs, best validation chrF kept
print(pipe.evaluate(splits['test'])['chrf'])                                                 # 59.36 in the recorded run
artifact = pipe.save_artifact('outputs/adapter')                                             # adapter.safetensors (33.6 MB) + manifest.json
again = MarianMTTranslationPipeline.from_artifact(artifact)                                  # verifies base digest + artifact digest before applying
```

`translate(texts, *, max_new_tokens=128, num_beams=4)` takes 1..16 non-empty English strings (`MAX_BATCH`) of at most 4,000 characters each (`MAX_TEXT_CHARS`) that tokenise to at most 512 SentencePiece tokens including `</s>` (`MAX_INPUT_TOKENS`; longer inputs are rejected, not truncated), `max_new_tokens` in 1..512 (`MAX_NEW_TOKENS`) and `num_beams` in 1..8 (`MAX_NUM_BEAMS`; the default 4 is the snapshot's `generation_config.json` value). The result carries one `translations` entry per input, in order — `source`, `text`, `input_tokens`, `generated_tokens`, `stopped_by` (`eos` or `max_new_tokens`) — plus `n`, `direction` (`en->tl`), the `generation` settings (`max_new_tokens`, `num_beams`, `do_sample=False`, `decision_rule`), `device`, `source`, `model_id` and `model_revision`. `evaluate(records)` scores a validated `{id, source, target}` dataset with corpus chrF and BLEU (own implementations in `metrics.py`, sacrebleu-style, not sacrebleu-identical); `evaluation_report(result, references)` scores supplied references (`measured` / `measured-small-sample`) and still returns `not-measurable` without them; `adapt(train, val, *, epochs=2, lr=1e-4, batch_size=16, trainable_decoder_layers=2, seed=0)` fine-tunes the last decoder blocks and keeps the best-validation-chrF epoch; `save_artifact` / `from_artifact` export and reload the trained tensors as safetensors with a manifest bound to the base weight digest. Dataset helpers (`fetch_corpus`, `build_sample_dataset`, `validate_dataset`, `split_dataset`, `check_split_disjoint`, `load_byod_dataset`, `write_dataset_csv`) live in `samples.py`; records are 8–20,000 `{id, source, target}` mappings and every ceiling is a refusal, never a silent cut, except the 128-piece truncation applied to training pairs only.

## Weights layout

```
weights/opus-mt-en-tl/
  config.json  generation_config.json  pytorch_model.bin  source.spm  target.spm  tokenizer_config.json  vocab.json
  README.md  dimer-base-manifest.json  (no model.safetensors upstream at this revision)
```

`from_pretrained()` calls `stage_missing_files()` (fetches absent manifest entries at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (size + SHA-256 of every entry — including the 296 MB pickle — before `torch` is imported), and loads `MarianMTModel` with `use_safetensors=False`, `weights_only=True`, `local_files_only=True` and `trust_remote_code=False`, so `transformers` deserialises the pickle through `torch.load(weights_only=True)` (observed in the smoke: two calls, both `weights_only=True`). `MarianTokenizer` reads `source.spm`/`target.spm`/`vocab.json` (needs `sentencepiece`, pinned). Without a manifest it raises unless `allow_download=True`. See `docs/WEIGHTS.md`.

`sacremoses` is not a dependency of this repository: `MarianTokenizer` then warns `Recommended: pip install sacremoses.` once and skips the optional source-side Moses punctuation normalisation (the normaliser is the identity function); the smoke ran in that configuration.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake runner, token counter and corpus fetcher plus temporary manifests, never the weights (35 tests). `tests/test_model_backed.py` (2 tests: `evaluate` against references, a one-epoch adaptation of the last decoder block with an artifact round trip) runs only when `weights/opus-mt-en-tl/` is staged.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/tutorials/marianmt_translation_colab.ipynb)

`tutorials/marianmt_translation_colab.ipynb` is declared `E2E` (mode `GUIDED`) under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`; see `tutorials/README.md`). Its default path fetches one pinned Tatoeba en–tl zip from OPUS (312 KB, CC BY 2.0 FR, refused on any digest mismatch) and cuts it into 1,200 / 200 / 300 source-disjoint records with four refusal probes, stages the git-ignored `pytorch_model.bin` with `stage_missing_files(..., allow_download=True)` and digest-verifies it before the pickle is opened with `weights_only=True`, exercises the inference contract with its input manifest and sanity checks, scores the copy-source baseline and the frozen model on the test split (chrF 11.28 / 56.46 in the recorded run), fine-tunes the last two decoder blocks for two epochs with validation-chrF epoch selection (56 s on CPU), re-scores the test split (chrF 59.36, BLEU 27.13 → 33.75), translates six unseen sentences with a `measured-small-sample` report, exports a 33.6 MB safetensors adapter and reloads it with 8/8 identical translations. Every number is one seeded split with no dispersion estimate. BYOD (`{id, source, target}` CSV / JSON / JSONL) is optional and gated off by default. See `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `1adc963a` (committed at `292d4fa`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 274.3 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Licensing

This repository's code is Apache-2.0 (`LICENSE`). The packaged upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
