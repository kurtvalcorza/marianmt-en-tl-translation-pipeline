"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "marianmt_translation_pipeline",
    "repo_name": "marianmt-en-tl-translation-pipeline",
    "stem": "marianmt_translation",
    "notebook_name": "marianmt_translation_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "MarianMTTranslationPipeline",
    "weights_key": "opus-mt-en-tl",
    "runtime_imports": ["torch", "transformers"],
    "title": "OPUS-MT en-tl — DIMER English→Tagalog translation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/tutorials/marianmt_translation_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Helsinki--NLP%2Fopus--mt--en--tl-ffcc4d?style=flat",
            "https://huggingface.co/Helsinki-NLP/opus-mt-en-tl",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-Helsinki--NLP%2FOPUS--MT--train-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/Helsinki-NLP/OPUS-MT-train",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-1804.00344-b31b1b.svg", "https://arxiv.org/abs/1804.00344"),
    ],
    "capability": "batched English→Tagalog machine translation (EN→TL only) using the pinned `Helsinki-NLP/opus-mt-en-tl` weights",
    "intro": (
        "`Helsinki-NLP/opus-mt-en-tl` is a ~77 M-parameter OPUS-MT Marian transformer (6+6 layers, `d_model` 512) "
        "trained by the Helsinki-NLP group on the `opus+bt` corpus and released on 2020-02-26 — an **EN→TL only** "
        "model: English in, Tagalog out, and nothing else. At inference the encoder reads the English sentence once "
        "and the decoder emits one SentencePiece token per step until `</s>` or a step ceiling; **beam search with "
        "4 beams** (the snapshot's `generation_config.json`) is the default decision rule and greedy decoding is "
        "available on request — there is no sampling and no seed. **No adaptation occurs:** no training, "
        "fine-tuning, in-context conditioning, or preprocessing fitting — the pinned checkpoint is used as published. "
        "**The checkpoint is a pickle:** the only upstream weight file at this revision is `pytorch_model.bin` (no "
        "SafeTensors), so Section 3 re-hashes it against the inline SHA-256 manifest **before** it is opened, and "
        "the carried module then loads it with `use_safetensors=False, weights_only=True`, which makes "
        "`transformers` call `torch.load(weights_only=True)` — a restricted unpickler. What the upstream checkpoint "
        "supplies is the model and the SentencePiece tokenizer; what the carried pipeline module adds is manifest "
        "verification, input validation with named ceilings, batched translation with a fixed output contract, "
        "and the `validate_inputs` and `evaluation_report` stage helpers.\n\n"
        "**Environment note:** `sacremoses` is not pinned and not installed, so `MarianTokenizer` prints one "
        "`Recommended: pip install sacremoses.` warning and uses the identity function as its source-side "
        "punctuation normaliser; the card-pass smoke ran in exactly that state and the outputs below are from it."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author three English "
        "sentences (or upload your own), stage and digest-verify the immutable upstream snapshot — a pickle "
        "checkpoint pinned by SHA-256 and loaded with `weights_only=True` — surface the pipeline's ceilings and "
        "the decision rule and validate the batch into an input manifest before any model work, translate "
        "through the public API with explicit `max_new_tokens`/`num_beams`, read `stopped_by` and the token "
        "counts correctly, read from the machine-readable evaluation report why **no metric is reported** and "
        "what references a BLEU/chrF evaluation would need, and export every translation with its identifier "
        "plus provenance."
    ),
    "exclusions": (
        "Tagalog→English or any other direction, document-level translation with sentence splitting, "
        "instruction following or chat, sampling-based decoding, glossary or terminology control, source-side "
        "Moses punctuation normalisation (`sacremoses` is not installed), or any BLEU/chrF measurement. The "
        "repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (float32 there too). CPU is adequate: the repository's model card records, for the Windows-venv smoke on an Intel Core Ultra 9 275HX, 6.0 s to load and digest-verify the 299 MB snapshot and 0.179 s (4 beams) / 0.108 s (greedy) for a batch of three sentences. The pinned `torch==2.14.0` install and the 296 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what an encoder-decoder (seq2seq) model is; what beam search and greedy decoding do; why a translation can be fluent and wrong.",
        "- **Data:** the default sample is **synthetic** — three English sentences authored in code (the ones the card-pass smoke used) — so nothing is downloaded and no private data is needed. It carries no reference translations, so any number it produces is smoke/sanity evidence, never a quality measurement. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction; each non-empty line of the uploaded UTF-8 text file is one English input. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: three English sentences authored in this cell — the same three the "
                "repository's card-pass smoke translated (`The house is wonderful.`, `Good morning to all of you.`, "
                "`Where is the nearest hospital?`). None has a reference translation, so nothing in this notebook is a "
                "quality measurement; the card's smoke observations (`Ang bahay ay kahanga - hanga.`, `Magandang umaga sa "
                "inyong lahat.`, `Nasaan ang pinakamalapit na ospital?`) are one run on one machine, not expected values "
                "this notebook asserts. The sample identity and a SHA-256 of its text are printed so an export can be "
                "tied to exactly these inputs.\n\n"
                "Two Colab form parameters fix the generation settings for the whole batch: `GEN_MAX_NEW_TOKENS` "
                "(default 128, the package's `DEFAULT_MAX_NEW_TOKENS`) and `NUM_BEAMS` (default 4, the snapshot's "
                "`generation_config.json` value and the package's `DEFAULT_NUM_BEAMS`; set 1 for greedy). They are "
                "checked against the carried module's ceilings in the next section.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 text file in which every "
                "non-empty line is one **English** input; at most `MAX_BATCH` lines per run, each at most "
                "`MAX_TEXT_CHARS` characters and tokenising to at most `MAX_INPUT_TOKENS` SentencePiece pieces, which "
                "the pipeline enforces by rejecting, not by truncating. The upload stays inside this runtime. If you "
                "hold reference Tagalog translations for your lines, keep them outside the notebook — Section 7 "
                "explains what to compute with them."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "GEN_MAX_NEW_TOKENS = 128  # @param {{type:\"integer\"}}\n"
                "NUM_BEAMS = 4  # @param {{type:\"integer\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    texts = [line.strip() for line in io.StringIO(uploaded[sample_name].decode('utf-8')) if line.strip()]\n"
                "    if not texts:\n"
                "        raise ValueError(f'{{sample_name}}: expected at least one non-empty English line')\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    texts = [\n"
                "        'The house is wonderful.',\n"
                "        'Good morning to all of you.',\n"
                "        'Where is the nearest hospital?',\n"
                "    ]\n"
                "    sample_name = 'synthetic_english_sentences'\n"
                "    sample_kind = 'synthetic (authored in this cell)'\n"
                "item_ids = [f'input{{index:02d}}' for index in range(len(texts))]\n"
                "sample_sha256 = hashlib.sha256('\\n'.join(texts).encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'inputs': len(texts), 'text_sha256': sample_sha256, 'max_new_tokens': GEN_MAX_NEW_TOKENS, 'num_beams': NUM_BEAMS}})\n"
                "for item_id, text in zip(item_ids, texts, strict=True):\n"
                "    print(f'{{item_id}}: {{text[:110]}}' + ('...' if len(text) > 110 else ''))"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `translate` "
                "applies — both route through the same private `_check_inputs` — so the sequence type, the batch size "
                "1..`MAX_BATCH`, each item's type, non-emptiness and character ceiling `MAX_TEXT_CHARS`, "
                "`max_new_tokens` in 1..`MAX_NEW_TOKENS` and `num_beams` in 1..`MAX_NUM_BEAMS` are enforced "
                "identically. It returns one **input manifest** naming the schema and ceilings, the direction "
                "(`en->tl only`), each input's identifier, character and word counts, the settings in force, and the "
                "verdict; it is written to `outputs/{stem}_input_manifest.json`. `MAX_INPUT_TOKENS` (encoder tokens "
                "including `</s>`; the upstream `max_position_embeddings`) needs the real tokenizer and is therefore "
                "enforced inside `translate`, which **rejects with a `ValueError` naming the count, never silently "
                "cuts**; every translation reports `input_tokens`. `DECISION_RULE` states the decoding rule in force "
                "(beam search, 4 beams by default; greedy when `num_beams=1`; no sampling). To show what rejection looks "
                "like, the cell also validates an out-of-range `num_beams` and records the pipeline's own error message "
                "as a finding. Nothing here trims or alters the texts, and nothing checks that they are English — a "
                "non-English line is accepted and mistranslated."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_BATCH': MAX_BATCH, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_INPUT_TOKENS': MAX_INPUT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'MAX_NUM_BEAMS': MAX_NUM_BEAMS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DEFAULT_NUM_BEAMS': DEFAULT_NUM_BEAMS}}\n"
                "print(ceilings)\n"
                "print({{'decision_rule': DECISION_RULE, 'direction': f'{{SOURCE_LANG}}->{{TARGET_LANG}}'}})\n"
                "input_manifest = validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, names=item_ids)\n"
                "# Demonstrate rejection on a setting that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=MAX_NUM_BEAMS + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'num-beams-ceiling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "print({{'token_ceiling': 'enforced by translate() with the real tokenizer; reported as input_tokens per item'}})"
            ),
        },
        {
            "md": (
                "## 6. Translate and read the outputs correctly\n\n"
                "`translate(texts, max_new_tokens=..., num_beams=...)` runs the whole batch through the encoder-decoder "
                "in one padded pass and returns a dict with one `translations` entry per input, in order: `source` (the "
                "English input), `text` (the decoded Tagalog with special tokens removed), `input_tokens` (encoder "
                "tokens including `</s>`, the number checked against `MAX_INPUT_TOKENS`), `generated_tokens` (decoder "
                "tokens emitted, excluding `</s>`/pad) and `stopped_by` (`eos` when the model ended the sequence itself, "
                "`max_new_tokens` when it hit the ceiling — such an output is cut mid-sentence and should be re-run with "
                "a larger `GEN_MAX_NEW_TOKENS` before anyone reads it as a finished translation); plus `n`, `direction`, "
                "the `generation` settings actually used (`max_new_tokens`, `num_beams`, `do_sample=False`, "
                "`decision_rule`), `device`, `source` and the model identity. **Score semantics:** the pipeline emits "
                "**no probability, confidence or score of any kind** — the counts above are counts, not scores; beam "
                "search keeps the highest-scoring sequences with no minimum-probability cut-off, so some token is always "
                "produced, and the pipeline ships no acceptance threshold on output quality. Whoever deploys it owns any "
                "acceptance rule, judged on their own references. The run is deterministic for a given batch, settings, "
                "weights, device and library versions (no sampling, `model.eval()`, no seed needed); batch padding and "
                "float32 kernel differences between CPU and CUDA can flip a near-tied token and change the rest of the "
                "sequence from that point, and beam search can differ from greedy. The checks below are falsifiable "
                "plumbing checks — one result per input, in order, every count within its ceiling, the direction echoed "
                "— plus the batch wall time measured on the runtime identified in Section 1 (includes warm-up). Look "
                "for three short Tagalog sentences; whether they are *good* is exactly what no number here can tell you."
            ),
            "code": (
                "import time\n\n"
                "started = time.perf_counter()\n"
                "result = pipe.translate(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "elapsed = time.perf_counter() - started\n"
                "results = [{{'id': item_id, 'seconds_batch': round(elapsed, 3), **item}} for item_id, item in zip(item_ids, result['translations'], strict=True)]\n"
                "for r in results:\n"
                "    print(f\"{{r['id']}} {{r['input_tokens']}} -> {{r['generated_tokens']}} tokens, stopped_by={{r['stopped_by']}}\")\n"
                "    print(f\"    {{r['source']}}\")\n"
                "    print(f\"    {{r['text']}}\")\n"
                "print({{'batch_seconds': round(elapsed, 3), 'direction': result['direction'], 'device': result['device']}})\n"
                "checks = {{\n"
                "    'one_result_per_input_in_order': [r['source'] for r in results] == list(texts),\n"
                "    'generated_within_ceiling': all(r['generated_tokens'] <= GEN_MAX_NEW_TOKENS for r in results),\n"
                "    'input_within_ceiling': all(r['input_tokens'] <= MAX_INPUT_TOKENS for r in results),\n"
                "    'direction_is_en_to_tl': result['direction'] == f'{{SOURCE_LANG}}->{{TARGET_LANG}}',\n"
                "    'settings_echoed': result['generation']['max_new_tokens'] == GEN_MAX_NEW_TOKENS and result['generation']['num_beams'] == NUM_BEAMS and result['generation']['do_sample'] is False,\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'translate output failed a sanity check: {{checks}}')\n"
                "print({{'checks': checks, 'decision_rule': result['generation']['decision_rule'], 'hit_token_ceiling': [r['id'] for r in results if r['stopped_by'] == 'max_new_tokens']}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report — even, as "
                "here, when nothing is measurable. The repository ships **no metric helper and reports no performance "
                "measure**: machine translation is conventionally scored with BLEU and chrF (or COMET) against human "
                "**reference translations** — one or more Tagalog references per source sentence — over enough sentences "
                "to state a dispersion, and the synthetic sample has none, so the verdict is always `not-measurable` and "
                "none is manufactured from a proxy such as length ratio. Supplying references does not change the "
                "verdict, because no metric helper exists to score them and a handful of references is not a "
                "dispersion; the helper records that in `reason`. The report covers the whole batch (`n_inputs`, summed "
                "`n_generated_tokens`) and lands at `outputs/{stem}_evaluation_report.json`. The upstream Tatoeba "
                "figures in the model card (BLEU 26.6, chr-F 0.577) are upstream claims, not measured here."
            ),
            "code": (
                "report = evaluation_report(result, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: the sample has no reference translations, and the repository ships no metric helper; compute BLEU/chrF on your own referenced sentences.')"
            ),
        },
        {
            "md": (
                "## 8. Export the translations and provenance\n\n"
                "Two further files are written under `outputs/` beside the input manifest and the evaluation report: "
                "`outputs/{stem}_translations.csv` — one row per input with its identifier, the English source, the "
                "Tagalog output, both token counts, `stopped_by` and the batch wall time, so every translation maps "
                "back to its input — and `outputs/{stem}_result.json`, which carries the same items plus the generation "
                "settings in force, the direction, the ceilings, the sanity checks, the input manifest, the evaluation "
                "report, the sample identity and digest, the notebook's source (repository, revision, embedded module "
                "digest, generator), the model identifier, the immutable model revision, the model licence, the verified "
                "snapshot summary (including the pickle's manifest digest, so the export records what was loaded), and "
                "the runtime identity (Python, `torch`, `transformers`, device, dtype). No credentials are involved in "
                "any step, so none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "items = [\n"
                "    {{'id': r['id'], 'source': r['source'], 'translation': r['text'], 'input_tokens': r['input_tokens'], 'generated_tokens': r['generated_tokens'], 'stopped_by': r['stopped_by'], 'seconds_batch': r['seconds_batch']}}\n"
                "    for r in results\n"
                "]\n"
                "with open('outputs/{stem}_translations.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(items[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(items)\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "payload = {{\n"
                "    'items': items,\n"
                "    'generation': result['generation'],\n"
                "    'direction': result['direction'],\n"
                "    'ceilings': ceilings,\n"
                "    'sanity_checks': checks,\n"
                "    'translations_file': 'outputs/{stem}_translations.csv',\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'inputs': len(texts), 'text_sha256': sample_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'pytorch pickle, digest-verified, loaded with weights_only=True', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "        'source': pipe.source,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The Tagalog strings are the model's beam-search (or greedy) output for *your* English input: fluent text that "
        "can drop a negation, change a number, leave an entity in English or pick a politeness register the source "
        "never specified, and the pipeline attaches no probability, confidence or quality score to it — "
        "`generated_tokens`, `input_tokens` and `stopped_by` are counts and flags, not evidence of adequacy. On the "
        "synthetic sample the checks prove only that the input contract, the digest-verified pickle load, the batched "
        "generation path and the ordering work end to end; the evaluation report is `not-measurable` because nothing "
        "can be computed without reference translations, and a real evaluation needs referenced sentences from your "
        "own domain, a BLEU/chrF-style scorer, and enough items to state a dispersion. Inputs above `MAX_INPUT_TOKENS` "
        "are refused rather than cut; multi-sentence inputs are translated as one sequence; outputs that stop at "
        "`max_new_tokens` are truncated mid-sentence; non-English input is accepted and mistranslated. The pipeline "
        "exposes no reverse direction, sentence splitting, sampling or terminology control, and `sacremoses` "
        "punctuation normalisation is not applied. Decoding is deterministic on a fixed device, dtype and batch, but "
        "CPU and CUDA float32 kernels can diverge on a near-tied token.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot (including its pickle weight file), validate the "
        "demonstrated inputs against the enforced ceilings, execute the public pipeline path, and emit the shown "
        "machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, translation quality on any domain, a usable acceptance threshold, safety for "
        "high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: "
        "the pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the "
        "top. `FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file "
        "(most likely the 296 MB `pytorch_model.bin`) is incomplete or altered — delete it from `weights/{MODEL_KEY}/` "
        "and rerun Section 3; the load never proceeds on a digest mismatch. `UserWarning: Recommended: pip install "
        "sacremoses.` in Section 3: expected; the normaliser is the identity in this configuration. `ValueError: "
        "texts[i] is N tokens; ceiling is MAX_INPUT_TOKENS=512` in Section 6: split or shorten that BYOD line and rerun "
        "from Section 4. `stopped_by` equal to `max_new_tokens`: raise `GEN_MAX_NEW_TOKENS` (ceiling `MAX_NEW_TOKENS`) "
        "and rerun Section 6. Output that repeats or echoes the input: the line is probably not English.\n\n"
        "**Next experiments.** Set `NUM_BEAMS = 1` and compare the greedy outputs with the 4-beam ones (the card-pass "
        "smoke found the three sentences unchanged); translate the same sentence with curly quotes and with straight "
        "quotes and compare, since no punctuation normalisation runs; hand the pipeline a sentence you hold a human "
        "Tagalog reference for and score the output with a BLEU/chrF implementation of your choice — the first step "
        "towards the real evaluation the report asks for; run the same batch on a CUDA runtime and diff the outputs "
        "against the CPU run. None of these turns the sample result into evidence of production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance and the pickle trust boundary: https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream training code: https://github.com/Helsinki-NLP/OPUS-MT-train\n"
        "- OPUS-MT — Building open translation services for the World (Tiedemann & Thottingal, EAMT 2020): https://aclanthology.org/2020.eamt-1.61\n"
        "- Marian: Fast Neural Machine Translation in C++ (Junczys-Dowmunt et al., ACL 2018): https://arxiv.org/abs/1804.00344"
    ),
}
