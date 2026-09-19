"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E translation workflow: the pinned OPUS-MT en→tl snapshot is
digest-verified and loaded, a digest-pinned real parallel corpus (Tatoeba en–tl via OPUS) is fetched,
validated and split, three English sentences are translated through the inference contract, the frozen
model is scored against references beside the copy-source baseline, a bounded fine-tuning of the last
decoder layers runs in the kernel, the held-out split is scored again, and the adapter is exported and
reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "marianmt_translation_pipeline",
    "repo_name": "marianmt-en-tl-translation-pipeline",
    "stem": "marianmt_translation",
    "notebook_name": "marianmt_translation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned OPUS-MT snapshot (a pickle checkpoint pinned by SHA-256 and loaded with `weights_only=True`), fetches the "
        "digest-pinned Tatoeba English–Tagalog corpus from OPUS (312 KB, no credential), filters and splits it into "
        "1,200 / 200 / 300 disjoint training, validation and test pairs, translates three English sentences through the "
        "inference contract with an input manifest and a rejection probe, scores the frozen model on the test split with "
        "chrF and BLEU beside the copy-source baseline, runs a bounded fine-tuning of the last two decoder blocks on the "
        "training pairs with validation-chrF epoch selection, scores the held-out split again, translates new sentences "
        "with the adapted model, exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh "
        "pipeline to verify translation parity. The default path needs no repository clone, no DIMER worker or service, no "
        "credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about "
        "two minutes of model time after the downloads."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "English–Tagalog pairs as a CSV (columns `id`, `source`, `target`), a JSON array or a JSONL file of `{{id, source, "
        "target}}` records. They pass through the same validation, seeded source-disjoint split, baselines, fine-tuning, held-out "
        "evaluation, inference, artifact export and reload-parity cells as the Tatoeba sample. The expected schema and the "
        "ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is "
        "optional and never part of the default path."
    ),
    "pipeline_class": "MarianMTTranslationPipeline",
    "weights_key": "opus-mt-en-tl",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "OPUS-MT en-tl — DIMER E2E English→Tagalog translation fine-tuning tutorial (standalone)",
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
    "capability": "batched English→Tagalog machine translation (EN→TL only) and bounded supervised fine-tuning of the last decoder blocks on a parallel corpus, using the pinned `Helsinki-NLP/opus-mt-en-tl` weights",
    "intro": (
        "`Helsinki-NLP/opus-mt-en-tl` is a ~74 M-parameter OPUS-MT Marian transformer (6+6 layers, `d_model` 512) "
        "trained by the Helsinki-NLP group on the `opus+bt` corpus and released on 2020-02-26 — an **EN→TL only** "
        "model: English in, Tagalog out, and nothing else. At inference the encoder reads the English sentence once "
        "and the decoder emits one SentencePiece token per step until `</s>` or a step ceiling; **beam search with "
        "4 beams** (the snapshot's `generation_config.json`) is the default decision rule and greedy decoding is "
        "available on request — there is no sampling and no seed. **The checkpoint is a pickle:** the only upstream "
        "weight file at this revision is `pytorch_model.bin` (no SafeTensors), so Section 3 re-hashes it against the "
        "inline SHA-256 manifest **before** it is opened, and the carried module then loads it with "
        "`use_safetensors=False, weights_only=True`, which makes `transformers` call `torch.load(weights_only=True)` — a "
        "restricted unpickler.\n\n"
        "What this notebook adds to inference is **adaptation with references**. The dataset is real: the Tatoeba "
        "English–Tagalog sentence pairs as distributed by OPUS (release v2023-04-12, 8,785 pairs, CC BY 2.0 FR), fetched as "
        "one digest-pinned 312 KB zip and read member by member. It is the corpus the upstream README reports BLEU 26.6 / "
        "chrF 0.577 on; the pinned model was trained on OPUS data that predates this release, so the frozen model is "
        "already strong here — the frozen test BLEU in Section 6 lands within a point of the upstream figure — and the "
        "fine-tuning question is whether a small in-domain adaptation still moves held-out chrF and BLEU. Two metrics are "
        "implemented in the carried `metrics.py` (corpus chrF and BLEU, sacrebleu-style, not sacrebleu-identical), and the "
        "**copy-source baseline** — the English input submitted as the translation — shows where a system that does "
        "nothing sits. Nothing here is a quality claim about your domain: it is one seeded split of one corpus.\n\n"
        "**Environment note:** `sacremoses` is not pinned and not installed, so `MarianTokenizer` prints one "
        "`Recommended: pip install sacremoses.` warning and uses the identity function as its source-side "
        "punctuation normaliser; every number below was produced in exactly that state."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, dataset and metrics modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot — a pickle checkpoint pinned by SHA-256 and loaded with "
        "`weights_only=True`; fetch a digest-pinned parallel corpus and validate and split it without leakage; translate "
        "through the public API with explicit `max_new_tokens`/`num_beams` and read `stopped_by` and the token counts "
        "correctly; score the frozen model against references beside the copy-source baseline and read why chrF is the "
        "headline for Tagalog; run a bounded fine-tuning with explicit hyperparameters and validation-based epoch "
        "selection; evaluate on an independent test split; translate new sentences; and export a safetensors adapter that "
        "reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "Tagalog→English or any other direction, document-level translation with sentence splitting, instruction "
        "following or chat, sampling-based decoding, glossary or terminology control, source-side Moses punctuation "
        "normalisation (`sacremoses` is not installed), full-model or encoder fine-tuning, back-translation, COMET or any "
        "learned metric, and any claim that a Tatoeba split stands in for your domain. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate: the build record measured 5.4 s to load and digest-verify the 299 MB snapshot, 11 s to translate the 300-sentence test split with 4 beams, and 56 s for the default two-epoch fine-tuning of the last two decoder blocks on 1,200 pairs. The pinned `torch==2.14.0` install and the 296 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what an encoder-decoder (seq2seq) model is; what beam search and greedy decoding do; why a translation can be fluent and wrong; what chrF and BLEU measure and why neither is a human judgement.",
        "- **Data contract:** records are `{{id, source, target}}` — an English sentence and its Tagalog reference, each 1..4,000 characters, ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a dataset needs 8..20,000 records; sources are de-duplicated case-insensitively before splitting so the same English sentence never sits in two splits; during training only, sources and targets are truncated to 128 SentencePiece pieces (inference never truncates — it rejects). BYOD accepts CSV, JSON or JSONL in that shape.",
        "- **Validation is structural, not linguistic:** nothing checks that a source is English, that a target is Tagalog, or that a pair is a faithful translation — a misaligned corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — a proprietary translation memory is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches one pinned object (`en-tl.txt.zip`, 312,459 bytes, SHA-256 `9abddc7d…`) from OPUS at `object.pouta.csc.fi` over HTTPS, refused on any mismatch before it is read; Tatoeba sentences are CC BY 2.0 FR (attribution: the Tatoeba contributors; redistribution: OPUS).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Parallel corpus, validation and split\n\n"
                "`fetch_sample_dataset` downloads the pinned Tatoeba en–tl zip from OPUS (or reads it from the cache), "
                "refuses a byte-size or SHA-256 mismatch before the archive is opened, reads the two Moses members "
                "without extracting to disk, keeps pairs with both sides in 3..200 characters, drops repeated English "
                "sources case-insensitively, and cuts a seeded shuffle into 1,200 training, 200 validation and 300 test "
                "records. `validate_dataset` then checks every record against the contract, `check_split_disjoint` asserts "
                "no English source appears in two splits, and the training split is written to "
                "`outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 8,785 raw pairs, three digests, splits 1,200 / 200 / 300, zero identical pairs, and four refusal "
                "probes — a duplicate id, an empty target, a missing field and a dataset too small to split — each rejected "
                "before `torch` does anything."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_pairs = len(records)\n"
                "else:\n"
                "    corpus_bytes = fetch_corpus(cache_dir='weights/tatoeba-en-tl')\n"
                "    raw_pairs = len(read_corpus_pairs(corpus_bytes))\n"
                "    splits = build_sample_dataset(read_corpus_pairs(corpus_bytes), seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} en-tl {{CORPUS_RELEASE}} via OPUS ({{CORPUS_LICENSE}})'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_pairs': raw_pairs, 'splits': disjoint, 'corpus_sha256': CORPUS_SHA256[:16] + '...'}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_sources': manifest['unique_sources'], 'identical_pairs': manifest['identical_pairs'], 'source_chars': manifest['source_chars'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'example': train_records[0]}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty target': [{{**train_records[0], 'target': ' '}}, *train_records[1:8]],\n"
                "    'missing field': [{{'id': r['id'], 'source': r['source']}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Translate through the inference contract\n\n"
                "Before any adaptation, the inference contract is exercised as it always was: `validate_inputs` applies "
                "exactly the checks `translate` applies (batch size 1..`MAX_BATCH`, non-empty strings under "
                "`MAX_TEXT_CHARS`, `max_new_tokens` and `num_beams` within their ceilings) and returns an input manifest; "
                "the encoder-token ceiling `MAX_INPUT_TOKENS` needs the real tokenizer and is enforced inside `translate`, "
                "which **rejects with a `ValueError` naming the count, never silently cuts**. `translate` returns one entry "
                "per input, in order, with `source`, `text`, `input_tokens`, `generated_tokens` and `stopped_by` (`eos` "
                "when the model ended the sequence itself, `max_new_tokens` when it hit the ceiling and the output is cut). "
                "**Score semantics:** the pipeline emits **no probability, confidence or score of any kind** — the counts "
                "are counts, and beam search always produces some token. The three sentences are the ones the "
                "repository's original smoke used; whether their translations are *good* is what Section 6 measures on "
                "300 referenced sentences, not what these three can tell you."
            ),
            "code": (
                "import time\n\n"
                "GEN_MAX_NEW_TOKENS = 128  # @param {{type:\"integer\"}}\n"
                "NUM_BEAMS = 4  # @param {{type:\"integer\"}}\n\n"
                "texts = ['The house is wonderful.', 'Good morning to all of you.', 'Where is the nearest hospital?']\n"
                "item_ids = [f'input{{index:02d}}' for index in range(len(texts))]\n"
                "ceilings = {{'MAX_BATCH': MAX_BATCH, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_INPUT_TOKENS': MAX_INPUT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'MAX_NUM_BEAMS': MAX_NUM_BEAMS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DEFAULT_NUM_BEAMS': DEFAULT_NUM_BEAMS}}\n"
                "print(ceilings)\n"
                "print({{'decision_rule': DECISION_RULE, 'direction': f'{{SOURCE_LANG}}->{{TARGET_LANG}}'}})\n"
                "input_manifest = validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, names=item_ids)\n"
                "try:\n"
                "    validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=MAX_NUM_BEAMS + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'num-beams-ceiling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "result = pipe.translate(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "elapsed = time.perf_counter() - started\n"
                "results = [{{'id': item_id, **item}} for item_id, item in zip(item_ids, result['translations'], strict=True)]\n"
                "for r in results:\n"
                "    print(f\"{{r['id']}} {{r['input_tokens']}} -> {{r['generated_tokens']}} tokens, stopped_by={{r['stopped_by']}}: {{r['source']}} => {{r['text']}}\")\n"
                "checks = {{\n"
                "    'one_result_per_input_in_order': [r['source'] for r in results] == list(texts),\n"
                "    'input_within_ceiling': all(r['input_tokens'] <= MAX_INPUT_TOKENS for r in results),\n"
                "    'direction_is_en_to_tl': result['direction'] == f'{{SOURCE_LANG}}->{{TARGET_LANG}}',\n"
                "    'settings_echoed': result['generation']['max_new_tokens'] == GEN_MAX_NEW_TOKENS and result['generation']['num_beams'] == NUM_BEAMS and result['generation']['do_sample'] is False,\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'translate output failed a sanity check: {{checks}}')\n"
                "print({{'batch_seconds': round(elapsed, 3), 'checks': checks, 'hit_token_ceiling': [r['id'] for r in results if r['stopped_by'] == 'max_new_tokens'], 'findings': len(input_manifest['findings'])}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's score on the test split\n\n"
                "Two numbers frame the adaptation. The **copy-source baseline** submits every English test sentence as its "
                "own translation and scores it against the Tagalog reference: what a system that does nothing gets, and a "
                "reminder that chrF rewards shared characters (names, numbers, punctuation) even across languages. The "
                "**frozen model** translates the 300 test sentences with the settings from Section 5 and is scored with the "
                "same two metrics: corpus **chrF** (character 1..6-grams, spaces removed, beta 2 — the headline, because "
                "Tagalog's affixation makes word-level matching harsh) and corpus **BLEU-4** (regex-tokenised, with the "
                "brevity penalty; sacrebleu-style, not sacrebleu-identical). Expect the frozen BLEU to land within a point "
                "of the upstream README's 26.6 on Tatoeba — a sanity check that the pinned weights, the tokenizer and the "
                "decoding path are the upstream ones — and read `hit_token_ceiling`, the count of outputs cut at "
                "`max_new_tokens`, before trusting any score."
            ),
            "code": (
                "baseline_copy = copy_source_baseline(test_records)\n"
                "print({{'copy_source_baseline': {{'chrf': round(baseline_copy['chrf'], 2), 'bleu': round(baseline_copy['bleu'], 2), 'n': baseline_copy['n']}}}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "print({{'frozen_model_test': {{'chrf': round(frozen_test['chrf'], 2), 'bleu': round(frozen_test['bleu'], 2), 'n': frozen_test['n'], 'hit_token_ceiling': frozen_test['hit_token_ceiling']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "shown = pipe.translate([r['source'] for r in test_records[:3]], max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)['translations']\n"
                "for record, item in zip(test_records[:3], shown, strict=True):\n"
                "    print({{'source': record['source'], 'frozen': item['text'], 'reference': record['target']}})\n"
                "assert frozen_test['chrf'] > baseline_copy['chrf']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last decoder blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_DECODER_LAYERS` decoder blocks — two by default, 8,408,064 of "
                "74,037,760 parameters; the encoder, the shared embeddings and the earlier decoder blocks stay frozen — with "
                "teacher-forced cross-entropy on the Tagalog target, AdamW at a fixed learning rate, gradient clipping at "
                "1.0, seeded shuffling and no scheduler. Sources and targets are truncated to 128 SentencePiece pieces "
                "**during training only**. Epoch 0 records the frozen model's validation chrF and BLEU; every epoch is "
                "scored on the validation split, and the epoch with the highest validation chrF is kept.\n\n"
                "Watch validation chrF rise a few points over two epochs (about a minute on CPU). The build record's "
                "counter-examples: training the whole decoder (25 M parameters) or the whole model (74 M) gained nothing "
                "beyond the last two blocks on this corpus while costing more time and a far larger artifact."
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_chrf'] = round(entry['val']['chrf'], 2)\n"
                "        row['val_bleu'] = round(entry['val']['bleu'], 2)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no English source in it appears in "
                "the training split. The adapted model is scored exactly as the frozen model was in Section 6, and the "
                "three numbers are put side by side. Look for a chrF gain of a few points and a BLEU gain of several — the "
                "cell asserts the adapted chrF is above the frozen chrF — and for the same three sentences translated by "
                "the adapted model. Three hundred sentences from one seeded split of one corpus give no dispersion "
                "estimate; the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and "
                "an in-domain gain on Tatoeba says nothing about your domain until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "comparison = {{\n"
                "    'chrf': {{'copy_source': round(baseline_copy['chrf'], 2), 'frozen': round(frozen_test['chrf'], 2), 'adapted': round(adapted_test['chrf'], 2)}},\n"
                "    'bleu': {{'copy_source': round(baseline_copy['bleu'], 2), 'frozen': round(frozen_test['bleu'], 2), 'adapted': round(adapted_test['bleu'], 2)}},\n"
                "    'delta_vs_frozen': {{'chrf': round(adapted_test['chrf'] - frozen_test['chrf'], 2), 'bleu': round(adapted_test['bleu'] - frozen_test['bleu'], 2)}},\n"
                "}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "shown_adapted = pipe.translate([r['source'] for r in test_records[:3]], max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)['translations']\n"
                "for record, item in zip(test_records[:3], shown_adapted, strict=True):\n"
                "    print({{'source': record['source'], 'adapted': item['text'], 'reference': record['target']}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'generation': frozen_test['generation'],\n"
                "    'baselines': {{'copy_source': baseline_copy}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['chrf'] > frozen_test['chrf']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Translate new sentences, export the adapter and reload it\n\n"
                "Six sentences that were in none of the splits are translated by the adapted model through the same "
                "`translate` contract as Section 5, and scored with `evaluation_report` — the inference-stage helper, which "
                "now returns a `measured-small-sample` verdict when references are supplied, because six sentences carry no "
                "dispersion estimate.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two decoder blocks, about 34 MB — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, "
                "the digest of the base pickle checkpoint, the tensor names, the file size and SHA-256, the training "
                "configuration and the epoch history (OUT8). `MarianMTTranslationPipeline.from_artifact` re-verifies the base "
                "snapshot, checks the artifact manifest and digest **before** deserialising, refuses any tensor that is not "
                "an adaptable decoder tensor, and overlays the tensors onto a freshly loaded base — a new object from files, "
                "not the in-memory model (VER2). The cell asserts identical translations (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "held_out = [r for r in read_corpus_pairs(corpus_bytes) if r[0].lower() not in {{x['source'].lower() for part in splits.values() for x in part}}][:6] if not USE_BYOD else [(r['source'], r['target']) for r in test_records[:6]]\n"
                "new_records = [{{'id': f'new-{{i:02d}}', 'source': s, 'target': t}} for i, (s, t) in enumerate(held_out)]\n"
                "new_result = pipe.translate([r['source'] for r in new_records], max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "new_report = evaluation_report(new_result, [r['target'] for r in new_records], sample_kind='six unseen Tatoeba pairs' if not USE_BYOD else 'BYOD test records')\n"
                "for record, item in zip(new_records, new_result['translations'], strict=True):\n"
                "    print({{'id': record['id'], 'source': record['source'], 'adapted': item['text'], 'reference': record['target'], 'stopped_by': item['stopped_by']}})\n"
                "print({{'new_sentences': {{'verdict': new_report['verdict'], 'metrics': new_report['metrics'], 'reason': new_report['reason']}}}})\n"
                "with open('outputs/{stem}_translations.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=['id', 'source', 'translation', 'reference', 'input_tokens', 'generated_tokens', 'stopped_by'])\n"
                "    writer.writeheader()\n"
                "    for record, item in zip(new_records, new_result['translations'], strict=True):\n"
                "        writer.writerow({{'id': record['id'], 'source': record['source'], 'translation': item['text'], 'reference': record['target'], 'input_tokens': item['input_tokens'], 'generated_tokens': item['generated_tokens'], 'stopped_by': item['stopped_by']}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = MarianMTTranslationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [item['text'] for item in pipe.translate([r['source'] for r in test_records[:8]], max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)['translations']]\n"
                "after = [item['text'] for item in reloaded.translate([r['source'] for r in test_records[:8]], max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)['translations']]\n"
                "parity = {{'identical_translations': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_translations'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'pytorch pickle, digest-verified, loaded with weights_only=True', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'url': CORPUS_URL, 'sha256': CORPUS_SHA256, 'license': CORPUS_LICENSE}},\n"
                "    'comparison': comparison,\n"
                "    'new_sentences': new_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model already translates Tatoeba-style sentences well — its test BLEU sits where the upstream README puts "
        "it — and a bounded fine-tuning of the last two decoder blocks on 1,200 in-domain pairs still lifts held-out chrF by "
        "a few points and BLEU by several, in about a minute on CPU, with a 34 MB adapter that reloads to identical "
        "translations. That is the claim: the adaptation contract works end to end on a real parallel corpus, and the "
        "numbers it produces are read against a copy-source baseline and the frozen model rather than in isolation.\n\n"
        "The test split is 300 sentences from one seeded split of one corpus, the metrics are two reference-based scores "
        "(own implementations, not sacrebleu-identical, and neither a human judgement), and Tatoeba is short, "
        "conversational and already in the model's training lineage. So a gain here says the contract works, not that the "
        "adapted model is better on your domain, that it handles long or technical text, or that its fluent output is "
        "faithful — a translation can drop a negation, change a number or leave an entity in English and still score "
        "well on character n-grams. Fine-tuning on a narrow corpus can also erode the model elsewhere; nothing here "
        "measures that.\n\n"
        "Three things to carry to real data. **References first:** the copy-source baseline and the frozen model's score "
        "on *your* references are the two numbers to read before any adapted one. **Leakage:** de-duplicate sources "
        "across splits (the contract does this case-insensitively) and split by document or session when your pairs come "
        "from one, never at random over near-duplicates. **Ceilings:** inputs over `MAX_INPUT_TOKENS` are refused at "
        "inference and truncated to 128 pieces only during training — long-document translation is out of scope.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot (including its pickle weight file), fetch and "
        "digest-verify a real parallel corpus, validate the demonstrated dataset contract without leakage, execute the "
        "inference contract and a bounded fine-tuning, evaluate against a trivial baseline and the frozen model on an "
        "independent split, and emit the shown machine-readable artifacts — without the repository being reachable. It "
        "does **not** establish benchmark superiority, translation quality on any other domain, a usable acceptance "
        "threshold, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_DECODER_LAYERS = 6` to train the "
        "whole decoder and compare the artifact size and the test scores; raise `EPOCHS`; set `NUM_BEAMS = 1` and read the "
        "greedy scores; or bring your own pairs through BYOD and read the copy-source baseline before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance and the pickle trust boundary: https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream training code: https://github.com/Helsinki-NLP/OPUS-MT-train\n"
        "- OPUS-MT — Building open translation services for the World (Tiedemann & Thottingal, EAMT 2020): https://aclanthology.org/2020.eamt-1.61\n"
        "- Marian: Fast Neural Machine Translation in C++ (Junczys-Dowmunt et al., ACL 2018): https://arxiv.org/abs/1804.00344\n"
        "- Tatoeba en–tl via OPUS (Tiedemann, LREC 2012; corpus release v2023-04-12, CC BY 2.0 FR): https://opus.nlpl.eu/Tatoeba-v2023-04-12.php\n"
        "- chrF: character n-gram F-score for automatic MT evaluation (Popović, WMT 2015): https://aclanthology.org/W15-3049\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
