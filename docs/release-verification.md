# Release verification

`tutorials/marianmt_translation_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary
  path; exactly one cell tagged `embedded_module` equal to `src/marianmt_translation_pipeline/pipeline.py` after the generator's documented
  rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the inline `PINS` equal to
  the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py` output; the pinned-install
  cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `MarianMTTranslationPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `translate(texts, max_new_tokens=..., num_beams=...)`, `evaluation_report`), the ceiling print
  (`MAX_BATCH`, `MAX_TEXT_CHARS`, `MAX_INPUT_TOKENS`, `MAX_NEW_TOKENS`, `MAX_NUM_BEAMS`, `DEFAULT_MAX_NEW_TOKENS`, `DEFAULT_NUM_BEAMS`, `DECISION_RULE`, the direction), the exports (including the pickle's `weight_format`/`weight_sha256`), the learner-facing statements (EN→TL only, beam search with 4 beams as the default decision rule, the checkpoint is a pickle loaded with `use_safetensors=False, weights_only=True`, no probability or score emitted, no metric helper and the verdict always `not-measurable`, inputs above the token ceiling rejected not truncated, the `sacremoses` environment note, named exclusions) and the
  gated-off BYOD default listed in the validator; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `MarianMTModel` / `MarianTokenizer` / `from huggingface_hub import` /
  `model.generate(` use **outside the carried module cell**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `tokenizers`, `sentencepiece`, `huggingface-hub`, `safetensors` and `numpy`, runs `ruff`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner and token counter, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `GEN_MAX_NEW_TOKENS = 128`, `NUM_BEAMS = 4`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executing (defining `MarianMTTranslationPipeline`, `validate_inputs`, `evaluation_report` and the ceilings) with no import of the repository package;
   - the three synthetic English sentences authored in code with their text SHA-256 printed, and the ceilings (`MAX_BATCH` 16, `MAX_TEXT_CHARS` 4000, `MAX_INPUT_TOKENS` 512, `MAX_NEW_TOKENS` 512, `MAX_NUM_BEAMS` 8, `DEFAULT_MAX_NEW_TOKENS` 128, `DEFAULT_NUM_BEAMS` 4), `DECISION_RULE` and the `en->tl` direction surfaced, and `validate_inputs` writing `outputs/marianmt_translation_input_manifest.json` (verdict `accepted`, three inputs, one recorded rejection finding from the `num_beams` ceiling probe);
   - pinned `Helsinki-NLP/opus-mt-en-tl` acquisition at the immutable revision through the carried module, the inline `MANIFEST` asserted against the module identity and written to `weights/opus-mt-en-tl/`:
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports `['pytorch_model.bin']` on a
     clean runtime, `verify_snapshot` returns its dict (8 files; the 296 MB pickle's SHA-256 must match before the load proceeds), and `from_pretrained(weights_dir=WEIGHTS_DIR)`
     loads from the verified directory with `source` `local-snapshot`, printing exactly one `Recommended: pip install sacremoses.` warning as in the card-pass smoke (a `torch.load` call with `weights_only=False`, if observed, is a blocker);
   - `translate` returning one entry per input, in order, with `source`, `text`, `input_tokens`, `generated_tokens`, `stopped_by`, the `en->tl` direction and the echoed `generation` settings, with every sanity check `True`; record the three outputs and token counts (the card-pass CPU smoke returned `Ang bahay ay kahanga - hanga.` / `Magandang umaga sa inyong lahat.` / `Nasaan ang pinakamalapit na ospital?`, 6/8/7 → 7/7/6 tokens, all stopped by `eos`; a different output on another runtime is a finding to record, not a failure by itself, because no metric is asserted);
   - `evaluation_report` writing `outputs/marianmt_translation_evaluation_report.json` with verdict `not-measurable` and an empty
     `metrics` list, and the "No metric is reported" line printed;
   - `outputs/marianmt_translation_result.json` and `outputs/marianmt_translation_translations.csv` written with `NOTEBOOK_SOURCE`, model revision,
     model licence, the pickle's `weight_sha256`, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/marianmt_translation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/roberta_question_answering_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `5df0317` / `c9155c925dfc` | Kaggle CPU (`kurtvalcorza/dimer-nb2-marianmt-translation` v1) | Default sample path | 308.2 s | **PASSED** — 8/8 ok code cells executed cleanly, 18 files, 299 MB staged |

## Current status

No clean-runtime execution of the notebook has been recorded yet; the run is **pending** and queued
to the GPU lane. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real
`hf_hub_download` fetch into the snapshot directory has not been executed), and the card pass executed `translate` only on CPU in the Windows venv from a complete local snapshot (returns/L9-B: CUDA path and Hub path not executed), so the clean run will be the first real execution of the staging path — and the first time `stage_missing_files` fetches the pickle from the Hub, where the digest check is what makes the fetched file trustworthy; the CPU inference path against real weights was executed once locally (card Runtime: 6.0 s load + verify, 0.179 s / 0.108 s per 3-sentence batch; `torch.load` observed with `weights_only=True`).

**The standalone carrier itself — executing the carried module cell in a runtime that has no repository
checkout — has been validated statically only (parity PASS) and never run end-to-end.** A carrier probe
did exec the install, carried-module and identity-assert cells in a fresh interpreter with the repository
package blocked on `sys.meta_path`, which confirms the cells define the public API without the package;
it fetched nothing and loaded no model. The clean run will therefore be the first execution of the
standalone path.
