# Release verification

`tutorials/marianmt_translation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `MarianMTTranslationPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus_pairs` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `check_split_disjoint`, `write_dataset_csv`, `validate_inputs` with the `num_beams` refusal probe, `pipe.translate`
  with the sanity checks, `copy_source_baseline`, `pipe.evaluate` on the frozen model and on the validation and test
  splits after adaptation with the chrF assertions, `pipe.adapt` with its explicit hyperparameters, `evaluation_report`
  with references on the unseen sentences, `pipe.save_artifact`, `MarianMTTranslationPipeline.from_artifact` and the
  reload-parity assertion, and the provenance fields `weight_format`, the pickle's `weight_sha256` and the `corpus`
  block), the six expected `outputs/` paths, the learner-facing statements (EN→TL only, beam search with 4 beams as the
  default decision rule, the checkpoint is a pickle loaded with `use_safetensors=False, weights_only=True`, no
  probability or score emitted, inputs above the token ceiling rejected not truncated, the copy-source baseline,
  sacrebleu-style not sacrebleu-identical, no dispersion estimate, the `sacremoses` environment note, the CC BY 2.0 FR
  corpus licence, named exclusions) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any
  `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `MarianMTModel` / `MarianTokenizer` / `model.generate(` / `from huggingface_hub import`
  / `urllib.request` / `zipfile.` / `safetensors` / `torch.optim` / `.backward(` / `pipe._model` use **outside the
  carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`,
  `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `tokenizers`, `sentencepiece`,
`huggingface-hub`, `safetensors` and `numpy`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`,
and the offline unit suite (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner, token counter and corpus fetcher,
temporary manifests, no weights — `tests/test_model_backed.py` is skipped without the snapshot). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/opus-mt-en-tl/` or the corpus cache `weights/tatoeba-en-tl/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub, and fetches the pinned Tatoeba zip from OPUS, so neither
   directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `GEN_MAX_NEW_TOKENS = 128`, `NUM_BEAMS = 4`, `EPOCHS = 2`,
   `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 16`, `TRAINABLE_DECODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `sentencepiece==0.2.2`,
   `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `MarianMTTranslationPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `fetch_corpus`, `read_corpus_pairs`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`,
     `write_dataset_csv`, `chrf`, `bleu`, `translation_metrics`, `copy_source_baseline` and the ceilings) with no
     import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['pytorch_model.bin']` (and any other absent entry) fetched from
     `Helsinki-NLP/opus-mt-en-tl` at the immutable revision, and `verify_snapshot` returning its dict (8 files; the
     296 MB pickle's SHA-256 must match before the load proceeds); `from_pretrained(weights_dir=WEIGHTS_DIR)` loading
     from the verified directory with `source` `local-snapshot`, printing exactly one
     `Recommended: pip install sacremoses.` warning (a `torch.load` call with `weights_only=False`, if observed, is a
     blocker);
   - Section 4: `fetch_corpus` fetching the pinned 312,459-byte zip (SHA-256 `9abddc7d…`) from
     `object.pouta.csc.fi` into `weights/tatoeba-en-tl/`, 8,785 raw pairs read, 7,741 after filtering, and the seeded
     split into 1,200 / 200 / 300 records with `check_split_disjoint` reporting no shared source and the three dataset
     digests `c5ade2af…` / `07c99289…` / `95119836…`; `outputs/marianmt_translation_train.csv` written; the four
     dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MAX_BATCH` 16, `MAX_TEXT_CHARS` 4000, `MAX_INPUT_TOKENS` 512, `MAX_NEW_TOKENS` 512,
     `MAX_NUM_BEAMS` 8, `DEFAULT_MAX_NEW_TOKENS` 128, `DEFAULT_NUM_BEAMS` 4), `DECISION_RULE` and the `en->tl`
     direction surfaced; `validate_inputs` writing `outputs/marianmt_translation_input_manifest.json` (verdict
     `accepted`, one recorded rejection finding from the `num_beams` ceiling probe); `pipe.translate` returning one
     entry per input, in order, with every sanity check `True`;
   - Section 6: the copy-source baseline (chrF ≈ 11.3, BLEU 0.0 on the sample split) and the frozen model's test
     score (chrF ≈ 56.5, BLEU ≈ 27.1 on CPU float32 — read as a sanity match against upstream's 26.6, not a
     reproduction), with the cell's assertion that the frozen model beats the baseline;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 8,408,064 trainable of 74,037,760 parameters,
     1,200 training pairs, and a two-epoch history with validation chrF rising (≈ 58.5 → 60.3 → 61.5 in the
     recorded run; `best_epoch` 2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the three-way comparison and
     `outputs/marianmt_translation_evaluation_report.json` written (the cell asserts the adapted test chrF exceeds the
     frozen one — on the sample ≈ 59.4 versus ≈ 56.5, BLEU ≈ 33.8 versus ≈ 27.1);
   - Section 9: six unseen Tatoeba sentences translated with `evaluation_report` returning `measured-small-sample`
     and two metrics, `outputs/marianmt_translation_translations.csv` written; `pipe.save_artifact` writing
     `outputs/marianmt_translation_adapter/{adapter.safetensors,manifest.json}` (52 tensors, about 33.6 MB) and
     `MarianMTTranslationPipeline.from_artifact` reloading it with 8/8 identical translations (the cell asserts it);
     `outputs/marianmt_translation_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
     snapshot block (`weight_format`, the pickle's `weight_sha256`), the `corpus` block, the comparison, the artifact
     digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `marianmt_translation_colab.ipynb` (`E2E`) | `__LOCAL_ROW__` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `marianmt_translation_colab.ipynb` (`TASK-INFERENCE`, superseded) | `5df0317` / `c9155c925dfc` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-marianmt-translation` v1) | PASSED — 8/8 code cells, 308.2 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/marianmt_translation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/marianmt_translation_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `__LOCAL_ROW__` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | `__LOCAL_EXEC__` |
| 2026-09-14 | `5df0317` / `c9155c925dfc` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-marianmt-translation` v1) | Default sample path of the inference-only notebook: three synthetic sentences, `stage_missing_files` fetching the pickle from the Hub, `verify_snapshot`, `translate`, `not-measurable` report | 308.2 s | **PASSED** — 8/8 code cells, 18 files, 299 MB staged; does not cover the `E2E` blob |

## Current status

The `E2E` notebook source is complete and passes all static checks, including the generator parity checks
(`--check` OK). A local pre-flight execution of the committed blob completed the whole default path on CPU — corpus
fetch from the cache, validation and split, the inference contract, both baselines, two epochs of decoder fine-tuning,
held-out evaluation, unseen-sentence translation, adapter export and reload parity — which catches defects but is
**not** a supported runtime under REL1/REL10, and it ran with the snapshot and the Tatoeba zip pre-staged, so neither
the 296 MB Hub fetch of the pickle nor the OPUS download has been exercised by this notebook end to end; the earlier
`TASK-INFERENCE` Kaggle run did exercise the Hub fetch and digest check of the same pickle. The repository stays at
**Candidate** until a Colab or fresh-container run of the exact `E2E` release revision is recorded above.
