"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §3.5 guided layer on the §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures a GUIDED E2E translation unit, "English–Tagalog Translation — Preserving
Meaning Across Languages" (curriculum Level 2, applied tasks). Around the unchanged technical workflow —
pinned OPUS-MT en→tl snapshot, digest-pinned Tatoeba en–tl corpus, validation and source-disjoint
split, inference contract, copy-source baseline and frozen-model scores, bounded fine-tuning of the
last decoder blocks with validation-chrF epoch selection, held-out evaluation, adapter export and
reload parity — it adds the learner layer: orientation, prediction prompts, side-by-side
source/reference/output comparisons chosen by fixed rules, surface meaning checks calibrated on the
human references, a controlled minimal-pair activity on instructional probes, interpretation
checkpoints with collapsible sample answers, an evidence-based conclusion scaffold, troubleshooting
and a glossary.

String conventions: `lede`, `run_all`, `byod`, `intro`, `learning_objectives`, `exclusions`,
`prerequisites` and `orientation` are inserted verbatim. Stage `cells` and `closing` go through
`str.format` (placeholders `{stem}`, `{MODEL_ID}`, ...); the older cells escape literal braces by
hand, and the cells written with `_t` use natural braces and `<<name>>` placeholders instead.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

import re as _re


def _t(text: str) -> str:
    """Escape literal braces for str.format and turn `<<name>>` into a generator placeholder."""
    return _re.sub(r"<<(\w+)>>", r"{\1}", text.replace("{", "{{").replace("}", "}}")).strip("\n")


_SAMPLE = "<details><summary><b>Sample answer</b> (open after you have written your own)</summary>\n\n"

TEMPLATE = {
    "package": "marianmt_translation_pipeline",
    "repo_name": "marianmt-en-tl-translation-pipeline",
    "stem": "marianmt_translation",
    "notebook_name": "marianmt_translation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "notebook_spec": "2.2",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned OPUS-MT snapshot (a pickle checkpoint pinned by SHA-256 and loaded with `weights_only=True`), fetches the "
        "digest-pinned Tatoeba English–Tagalog corpus from OPUS (312 KB, no credential), filters and splits it into "
        "1,200 / 200 / 300 disjoint training, validation and test pairs, translates three English sentences through the "
        "inference contract with an input manifest and a rejection probe, shows English, reference and pretrained output "
        "side by side, scores the pretrained model on the test split with chrF and BLEU beside the copy-source baseline, runs "
        "a bounded fine-tuning of the last two decoder blocks on the training pairs with validation-chrF epoch selection, "
        "scores the held-out split again and compares baseline, pretrained and adapted outputs sentence by sentence, runs "
        "surface meaning checks and a controlled minimal-pair activity on instructional probes, translates new sentences with "
        "the adapted model, exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh "
        "pipeline to verify translation parity. The default path needs no repository clone, no DIMER worker or service, no "
        "credential, no upload dialog and no configuration edit (DIMER Notebook Specification §5)."
    ),
    "byod": (
        "After the default run completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "English–Tagalog pairs as a CSV (columns `id`, `source`, `target`), a JSON array or a JSONL file of `{id, source, "
        "target}` records — uploaded through a dialog, or read from `BYOD_PATH` when you set it. They pass through the same "
        "validation, seeded source-disjoint split, baselines, fine-tuning, held-out evaluation, meaning checks, inference, "
        "artifact export and reload-parity cells as the Tatoeba sample. The expected schema and the ceilings are stated in the "
        "Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the "
        "default path."
    ),
    "pipeline_class": "MarianMTTranslationPipeline",
    "weights_key": "opus-mt-en-tl",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "English–Tagalog Translation — Preserving Meaning Across Languages",
    "lede": (
        "> **Driving question:** *A translation sounds fluent — but did it preserve the meaning?*\n\n"
        "A DIMER guided notebook · **Curriculum placement:** Level 2 — Applied tasks · **Model:** "
        "`Helsinki-NLP/opus-mt-en-tl` (English → Tagalog only) · **You need:** basic Python and Colab; no prior machine-learning experience"
    ),
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
        "**Why this matters.** Machine translation is usually *fluent* — the Tagalog reads naturally — even when it is "
        "wrong. A dropped *not*, a changed number or a swapped name gives a sentence that sounds fine and means something "
        "else. In this notebook you run a pretrained English→Tagalog model, adapt it on example sentence pairs, measure it "
        "with two automatic scores, read its translations for meaning, and run one small controlled experiment. The "
        "habit it builds is the one that matters in practice: never accept a translation, or a score, on fluency alone.\n\n"
        "**The model.** `Helsinki-NLP/opus-mt-en-tl` is a ~74 M-parameter OPUS-MT Marian transformer (6 encoder + 6 decoder "
        "layers, `d_model` 512) trained by the Helsinki-NLP group on the `opus+bt` corpus and released on 2020-02-26 — an "
        "**EN→TL only** model: English in, Tagalog out, and nothing else. Giving it Tagalog does not translate into "
        "English. Its default decision rule is **beam search with 4 beams** (the snapshot's `generation_config.json`); "
        "greedy decoding is available on request, and there is no sampling and no seed.\n\n"
        "**Engineering notes (safe to skip on a first read).** **The checkpoint is a pickle:** the only upstream weight file "
        "at this revision is `pytorch_model.bin` (no SafeTensors), so Section 3 re-hashes it against the inline SHA-256 "
        "manifest **before** it is opened, and the carried module then loads it with `use_safetensors=False, "
        "weights_only=True`, which makes `transformers` call `torch.load(weights_only=True)` — a restricted unpickler. The "
        "corpus is real: the Tatoeba English–Tagalog pairs distributed by OPUS (release v2023-04-12, 8,785 pairs, CC BY 2.0 "
        "FR), fetched as one digest-pinned 312 KB zip. It is the corpus the upstream README reports BLEU 26.6 / chrF 0.577 "
        "on, and the model's training data predates this release, so the pretrained model is already strong here; the "
        "adaptation question is whether a small in-domain fine-tuning still moves held-out scores. Both metrics are "
        "implemented in the carried `metrics.py` (corpus chrF and BLEU, sacrebleu-style, not sacrebleu-identical).\n\n"
        "**Environment note:** `sacremoses` is not pinned and not installed, so `MarianTokenizer` prints one "
        "`Recommended: pip install sacremoses.` warning and uses the identity function as its source-side punctuation "
        "normaliser; every recorded number for this notebook was produced in exactly that state."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to:\n\n"
        "1. **Explain** the task as *English text → translation model → Tagalog text*, including what the model does not do.\n"
        "2. **Distinguish** pretrained inference (using the model as published), adaptation (updating some of its parameters on "
        "example pairs) and evaluation (scoring outputs on sentences it never trained on).\n"
        "3. **Compare** a do-nothing baseline, the pretrained model and the adapted model on the same held-out sentences.\n"
        "4. **Interpret** chrF and BLEU alongside your own reading of the translations, and say what each score cannot tell you.\n"
        "5. **Conduct** one controlled experiment — change one meaning-bearing word, hold everything else fixed — and explain its limitations.\n"
        "6. **Identify** the evidence you would still need before using these translations in practice."
    ),
    "exclusions": (
        "Tagalog→English or any other direction, document-level translation with sentence splitting, instruction "
        "following or chat, sampling-based decoding, glossary or terminology control, source-side Moses punctuation "
        "normalisation (`sacremoses` is not installed), full-model or encoder fine-tuning, back-translation, COMET or any "
        "learned metric, a robustness benchmark, bilingual human evaluation with a scoring rubric, and any claim that a "
        "Tatoeba split stands in for your domain. The repository exposes none of these."
    ),
    "orientation": [
        (
            "## Start here\n\n"
            "**Who this notebook is for.** You can open a notebook in Google Colab, run cells, and read basic Python. You do "
            "not need machine-learning experience: each new term is explained where it first matters and collected in the "
            "**Glossary** at the end. Reading Tagalog helps in two activities but is not required — where it matters, the "
            "notebook shows you how to record uncertainty instead of guessing.\n\n"
            "**Runtime and downloads.** A CPU runtime is enough; a GPU (for example Colab's T4) is used automatically when "
            "present and is faster. The run downloads the pinned PyTorch and Transformers packages, the 296 MB model "
            "checkpoint from Hugging Face and a 312 KB sentence corpus from OPUS. No account, token or upload is needed. "
            "Measured: all cells after the downloads took about four minutes on a workstation CPU (230 s, 2026-09-27); "
            "installation and downloads add time that depends on your network, and a GPU is faster.\n\n"
            "### How to use this notebook\n\n"
            "1. Open it in Colab and choose **Runtime → Run all**. The default path needs no edits. If the first cell stops "
            "with *Restart the runtime, then rerun from the top*, do exactly that once (**Runtime → Restart session**, then "
            "**Run all** again): it means Colab had different package versions loaded.\n"
            "2. Cells with a form on the right (`# @param`) are the **knobs**. Leave them at their defaults for the first run. "
            "Afterwards, change one knob and re-run from that cell downwards.\n"
            "3. Sections 1–3 are **Infrastructure**: they install packages, carry the pipeline code and verify the model "
            "download. Run them; you do not need to read their code, which is collapsed where your notebook viewer supports it.\n"
            "4. Sections 4–13 are the lesson. Each follows the same rhythm: **question → predict → run → What to notice → "
            "Checkpoint**. Checkpoints have a collapsible **Sample answer** — write your own answer first.\n"
            "5. If something fails, see **Troubleshooting** at the end."
        ),
        (
            "## The task: Input → Model → Output\n\n"
            "```text\n"
            "  INPUT                        MODEL                                  OUTPUT\n"
            "  one English sentence   ──►   opus-mt-en-tl translation model   ──►  one Tagalog sentence\n"
            "  \"Where is the nearest       encoder reads the English;             \"Nasaan ang pinakamalapit\n"
            "   hospital?\"                 decoder writes Tagalog word-pieces      na ospital?\"\n"
            "```\n\n"
            "- **Input:** English text, up to 4,000 characters and 512 word-pieces (longer input is refused, not cut).\n"
            "- **Model:** a pretrained translation model; later in this notebook you also adapt a small part of it.\n"
            "- **Output:** Tagalog text. There is **no confidence score**: the model always produces some sentence, and "
            "nothing in its output tells you whether that sentence is right.\n\n"
            "You will use the model in three different ways — keep them apart:\n\n"
            "| Activity | What happens to the model | Where |\n"
            "|---|---|---|\n"
            "| **Pretrained inference** | nothing — it translates exactly as published | Section 6 |\n"
            "| **Adaptation** (fine-tuning) | a small part of its parameters is updated on 1,200 example pairs | Section 8 |\n"
            "| **Evaluation** | nothing — its outputs on 300 held-back sentences are compared with human references | Sections 7, 9, 10 |\n\n"
            "Headings tell you what kind of material a section is: **Core concept** (what translation models do), "
            "**Evaluation practice** (how to judge them) and **Engineering** (how the run is made reproducible and safe)."
        ),
        (
            "## Roadmap\n\n"
            "| Section | Stage | The question it answers | Kind |\n"
            "|---|---|---|---|\n"
            "| 1–3 | Install, pipeline code, model verification | Is the runtime exactly the pinned one? | Infrastructure |\n"
            "| 4 | Inspect the data | What does a sentence pair and its reference look like? | Core concept |\n"
            "| 5 | Predict | What do you expect the model to get wrong? | Evaluation practice |\n"
            "| 6 | Run the pretrained model | What does the published model produce? | Core concept |\n"
            "| 7 | Baseline and scores | What do chrF and BLEU measure, and what does doing nothing score? | Evaluation practice |\n"
            "| 8 | Adapt | Which parameters change, and how is the kept version chosen? | Core concept |\n"
            "| 9 | Evaluate | Did adaptation help on sentences it never saw — and where did it hurt? | Evaluation practice |\n"
            "| 10 | Interpret | Is the meaning preserved: negation, names, numbers, omissions, additions? | Evaluation practice |\n"
            "| 11 | Controlled activity | Does changing one meaning-bearing word change the translation accordingly? | Evaluation practice |\n"
            "| 12 | Export and reload | Does the saved adapter reproduce the model? | Engineering |\n"
            "| 13 | Conclude | What does the evidence support — and not support? | Evaluation practice |\n\n"
            "**Fast path:** if you are short on time, choose **Run all**, then read Sections 4, 6, 9, 10, 11 and 13."
        ),
    ],
    "infrastructure_note": (
        "**Infrastructure — you may run this without studying it.** This section keeps the run reproducible and safe; "
        "it is not part of the lesson. Its code cell is collapsed where your notebook viewer supports it."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate: a local pre-flight of this notebook on a workstation CPU (2026-09-27, downloads pre-staged) measured 2.6 s to digest-verify and load the 299 MB snapshot, 27–41 s per pass over the 300-sentence test split with 4 beams, 92 s for the default two-epoch fine-tuning of the last two decoder blocks on 1,200 pairs, and 230 s for all cells. The pinned `torch==2.14.0` install and the 296 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python (variables, lists, dictionaries, running a cell) and basic Colab use. No machine-learning background is assumed; the notebook explains translation models, beam search, fine-tuning, chrF and BLEU where they are used.",
        "- **Data contract:** records are `{id, source, target}` — an English sentence and its Tagalog reference, each 1..4,000 characters, ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a dataset needs 8..20,000 records; sources are de-duplicated case-insensitively before splitting so the same English sentence never sits in two splits; during training only, sources and targets are truncated to 128 SentencePiece pieces (inference never truncates — it rejects). BYOD accepts CSV, JSON or JSONL in that shape.",
        "- **Validation is structural, not linguistic:** nothing checks that a source is English, that a target is Tagalog, or that a pair is a faithful translation — a misaligned corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — a proprietary translation memory is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches one pinned object (`en-tl.txt.zip`, 312,459 bytes, SHA-256 `9abddc7d…`) from OPUS at `object.pouta.csc.fi` over HTTPS, refused on any mismatch before it is read; Tatoeba sentences are CC BY 2.0 FR (attribution: the Tatoeba contributors; redistribution: OPUS).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Inspect the data\n\n"
                "**Core concept — parallel data and references.** A *parallel corpus* is a list of sentence pairs: an English "
                "**source** and a Tagalog **reference** translation written by a person. A **reference is one acceptable "
                "translation, not the only correct wording.** Tagalog word order is flexible — *Tama ang sagot mo.* and "
                "*Ang sagot mo'y tama.* both say \"Your answer is correct.\" — so any score that compares the model's output "
                "with a single reference will mark down some correct translations. Keep that in mind in every later section.\n\n"
                "**Three splits, three jobs.** The pairs are shuffled with a fixed seed and cut into **training** (1,200 pairs "
                "the model may learn from), **validation** (200 pairs used only to choose which training epoch to keep) and "
                "**test** (300 pairs held back for the final comparison). No English sentence appears in two splits, so the "
                "test score is about sentences the model never trained on. A random split assumes the pairs are roughly "
                "independent; pairs that come from one document or conversation should be split by that grouping instead.\n\n"
                "**Engineering — what the next cell does.** `fetch_corpus` downloads the pinned Tatoeba en–tl zip from OPUS "
                "(or reads it from the cache) and refuses a byte-size or SHA-256 mismatch before the archive is opened; "
                "`read_corpus_pairs` reads the two text members without extracting to disk; `build_sample_dataset` keeps pairs "
                "with both sides in 3..200 characters, drops repeated English sources case-insensitively and cuts the seeded "
                "shuffle into the three splits. `validate_dataset` checks every record against the contract, "
                "`check_split_disjoint` asserts no English source appears in two splits, and the training split is written to "
                "`outputs/{stem}_train.csv` in the shape BYOD expects. Four refusal probes then feed deliberately broken data — "
                "a duplicate id, an empty target, a missing field and a dataset too small to split — to show that bad input is "
                "rejected with a message instead of being trained on.\n\n"
                "**What to notice:** 8,785 raw pairs; splits 1,200 / 200 / 300; `identical_pairs` 0; three dataset digests "
                "(fingerprints that identify exactly which data you used); and all four probes `rejected`."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH:\n"
                "        byod_path = Path(BYOD_PATH)\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        byod_path = Path('work') / file_name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
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
            "md": _t(
                """
### 4b. Read some pairs

The next cell prints the first eight training pairs **in the seeded split order** — a fixed rule, not a hand-picked selection. If you read Tagalog, read a few pairs and ask whether you would have translated them the same way. If you do not, look at structure: which words — names, numbers, borrowed English words — appear in both languages, and is the Tagalog usually longer or shorter than the English?
"""
            ),
            "code": _t(
                """
def show_rows(rows, columns):
    # Print each row as a labelled block so long sentences stay readable.
    width = max(len(label) for label, _key in columns)
    for row in rows:
        print('-' * 96)
        for label, key in columns:
            print(f'{label:>{width}} | {row[key]}')
    print('-' * 96)


print({'data_source': data_source, 'splits': disjoint, 'used_for': {'train': 'learning (Section 8)', 'validation': 'choosing the epoch to keep (Section 8)', 'test': 'final comparison only (Sections 7 and 9)'}})
show_rows(train_records[:8], [('id', 'id'), ('English', 'source'), ('reference', 'target')])
print({'mean_characters': {'English': round(sum(len(r['source']) for r in train_records) / len(train_records), 1), 'Tagalog reference': round(sum(len(r['target']) for r in train_records) / len(train_records), 1)}})
"""
            ),
        },
        {
            "md": _t(
                """
**Checkpoint 1 — one reference, many translations.** Pick one pair above. Think of a different wording that would still be correct — in Tagalog if you can, otherwise a different English wording of the source. Would it be wrong? What does that imply for a score that compares the model's output with this one reference?

"""
                + _SAMPLE
                + """A different wording can be completely correct and still share fewer letters and words with the reference, so it would score lower. Scores against a single reference therefore **underestimate** the quality of valid paraphrases, and a small score difference between two systems can reflect wording rather than meaning. That is why every score in this notebook is read together with actual sentences.

</details>
"""
            ),
        },
        {
            "md": _t(
                """
## 5. Predict: what will be hard for the model?

Before you run the model, write down — in a new text cell or on paper — which of these you expect it to get wrong most often, and why:

- **Negation** — *not*, *never*, *no*. One short word carries the whole meaning of the sentence.
- **Names** of people and places — will they be copied, respelled, or translated (the references write *France* as *Pransiya*)?
- **Quantities and numbers** — digits can be copied; number words must be translated.
- **Ambiguity** — English *you* can be one person or several; Tagalog distinguishes them (*ka / ikaw* versus *kayo*), so the model must guess.
- **Unfamiliar terminology** — words the model rarely saw. Tagalog speakers often keep such words in English, so an untranslated term is not always an error.

"""
                + _SAMPLE
                + """There is no single right prediction; the point is to have one you can check. Many people expect **negation** and **names** to be the riskiest: dropping one short word (the usual Tagalog negator is *hindi*) flips the meaning while leaving a fluent sentence, and names may be copied, respelled or translated. **Numbers** written as digits are often copied correctly, but a model may also write them as words — correct, yet harder to check automatically. **Ambiguous** *you* and **domain terms** are hard to judge without context. Keep your prediction; Sections 10 and 11 test it.

</details>
"""
            ),
        },
        {
            "md": (
                "## 6. Run the pretrained model\n\n"
                "### 6a. The inference contract\n\n"
                "**Core concept — pretrained inference.** The model translates exactly as published; nothing is trained. The "
                "encoder reads the whole English sentence; the decoder then writes Tagalog one **token** at a time (a token is a "
                "word or word-piece from the model's vocabulary). **Beam search with 4 beams** keeps the four most promising "
                "partial sentences at every step and returns the best finished one. It is deterministic: the same sentence and "
                "settings give the same output; nothing is sampled.\n\n"
                "**Engineering — the contract.** `validate_inputs` applies exactly the checks `translate` applies (batch size "
                "1..`MAX_BATCH`, non-empty strings under `MAX_TEXT_CHARS`, `max_new_tokens` and `num_beams` within their "
                "ceilings) and returns an input manifest; the encoder-token ceiling `MAX_INPUT_TOKENS` needs the real tokenizer "
                "and is enforced inside `translate`, which **rejects with a `ValueError` naming the count, never silently "
                "cuts**. `translate` returns one entry per input, in order, with `source`, `text`, `input_tokens`, "
                "`generated_tokens` and `stopped_by` (`eos` when the model ended the sentence itself, `max_new_tokens` when it "
                "hit the ceiling and the output is cut). **Score semantics:** the pipeline emits **no probability, confidence "
                "or score of any kind** — the counts are counts, and beam search always produces some sentence. The three "
                "sentences are the repository's original smoke inputs; whether translations are *good* is what Sections 7 and "
                "9 measure on 300 referenced sentences, not what these three can tell you.\n\n"
                "**What to notice:** one line per sentence with its token counts and `stopped_by`; every sanity check `True`; "
                "and one recorded finding — the `num_beams` probe was rejected before the model ran, because invalid settings "
                "fail loudly instead of being silently corrected."
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
            "md": _t(
                """
### 6b. English, reference and pretrained output side by side

The pretrained model now translates **every** test sentence with the settings above; these exact outputs are scored in Section 7 and compared with the adapted model in Section 9. The cell prints the first `SHOW_N` test pairs **in split order** — a fixed rule, so the rows were not chosen to flatter or embarrass the model.

**Decide what you will check before you read the output.** For each row: Is the meaning of the English preserved? Is a *not* dropped or added? Are names and numbers carried over? Is anything missing, or added that the English does not say? Is English left untranslated? Compare with the reference *loosely* — a different wording can still be correct.
"""
            ),
            "code": _t(
                """
SHOW_N = 6  # @param {type:"integer"}
show_n = max(1, min(int(SHOW_N), len(test_records)))


def translate_all(records):
    # Translate every record's source with the current model and the fixed decoding settings, MAX_BATCH at a time.
    outputs = []
    for start in range(0, len(records), MAX_BATCH):
        batch = records[start:start + MAX_BATCH]
        translated = pipe.translate([r['source'] for r in batch], max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)
        outputs.extend(item['text'] for item in translated['translations'])
    return outputs


t0 = time.perf_counter()
pretrained_outputs = translate_all(test_records)
print({'translated': len(pretrained_outputs), 'test_records': len(test_records), 'seconds': round(time.perf_counter() - t0, 1)})
paired = [{'id': r['id'], 'source': r['source'], 'reference': r['target'], 'pretrained': output} for r, output in zip(test_records, pretrained_outputs, strict=True)]
show_rows(paired[:show_n], [('id', 'id'), ('English', 'source'), ('reference', 'reference'), ('pretrained', 'pretrained')])
"""
            ),
        },
        {
            "md": (
                "**What to notice:** most outputs are fluent, and many differ in wording from the reference. Mark the rows where "
                "you think the *meaning* differs — not just the wording. If you cannot read Tagalog, mark the rows where names, "
                "numbers or sentence length look different, and write *uncertain* for meaning. Some outputs write hyphenated "
                "Tagalog words with spaces around the hyphen (`kahanga - hanga`): a formatting habit of this model that costs "
                "a little in the scores but does not change the meaning."
            ),
        },
        {
            "md": (
                "## 7. Establish a baseline and score the pretrained model\n\n"
                "**Evaluation practice — two automatic scores.** Both compare an output with the reference and run from 0 to "
                "100 in this implementation:\n\n"
                "- **chrF** counts shared *character* sequences (1 to 6 characters long, spaces removed) and combines precision "
                "and recall, weighting recall twice as much. It is the headline here because Tagalog builds words with prefixes, "
                "infixes and suffixes (*kain*, *kumain*, *kakainin*): a nearly right word still earns partial credit.\n"
                "- **BLEU** counts shared *word* sequences (1 to 4 words long) with a penalty for outputs shorter than the "
                "reference. A slightly different word form earns nothing, so BLEU is harsher.\n\n"
                "**Neither score is a percentage of sentences translated correctly.** 100 means the output matches the reference's "
                "character or word sequences exactly; a correct paraphrase scores lower. Both are **corpus-level**: the matches "
                "of all 300 sentences are pooled before the score is computed, so the denominator is the whole test split, not "
                "one sentence. Both are the carried `metrics.py` implementations — sacrebleu-style, not sacrebleu-identical.\n\n"
                "**The copy-source baseline** submits every English test sentence unchanged as its own \"translation\". It is "
                "not a translator — it is a **diagnostic reference**: it shows what a system that does nothing scores. It is "
                "above zero because names, numbers, punctuation and borrowed English words are shared across the two "
                "languages. The **pretrained model** is then scored on the same 300 sentences with the same settings. Its BLEU "
                "should land within about a point of the upstream README's 26.6 on Tatoeba — a sanity check that the weights, "
                "tokenizer and decoding are the upstream ones, not a reproduction of their evaluation.\n\n"
                "**What to notice:** the pretrained model far above the copy-source baseline (the cell asserts it); "
                "`hit_token_ceiling` 0 (no output was cut); the denominators; and `section_6b_outputs_reproduce_this_score` "
                "`True` — the outputs you read in 6b are exactly the ones being scored."
            ),
            "code": (
                "references = [r['target'] for r in test_records]\n"
                "baseline_copy = copy_source_baseline(test_records)\n"
                "print({{'copy_source_baseline': {{'chrf': round(baseline_copy['chrf'], 2), 'bleu': round(baseline_copy['bleu'], 2), 'n': baseline_copy['n']}}}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "print({{'frozen_model_test': {{'chrf': round(frozen_test['chrf'], 2), 'bleu': round(frozen_test['bleu'], 2), 'n': frozen_test['n'], 'hit_token_ceiling': frozen_test['hit_token_ceiling']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "recomputed = translation_metrics(pretrained_outputs, references)\n"
                "print({{'denominators': {{'sentences': frozen_test['n'], 'output_characters': frozen_test['hypothesis_chars'], 'reference_characters': frozen_test['reference_chars']}}, 'section_6b_outputs_reproduce_this_score': abs(recomputed['chrf'] - frozen_test['chrf']) < 0.01}})\n"
                "assert frozen_test['chrf'] > baseline_copy['chrf']"
            ),
        },
        {
            "md": (
                _t(
                    """
**Checkpoint 2 — reading a score.** The copy-source baseline translates nothing, yet its chrF is clearly above zero. Why? And would it be fair to describe the pretrained model's chrF as "the model is that many percent correct"?

"""
                    + _SAMPLE
                    + """The copy baseline shares characters with the reference: names such as *Tom* and *Mary*, digits, punctuation and English words that Tagalog borrows, so its chrF is above zero; its BLEU is usually 0 because whole four-word sequences almost never match. A chrF score is **not** "percent correct": it is an F-score of overlapping character sequences pooled over all 300 sentences. A perfect translation worded differently from the reference can score well below 100, and a fluent sentence with one fatal error (a dropped *not*) can score high. The score tells you *how much the output overlaps with one reference*, not *whether the meaning is right*.

</details>
"""
                )
            ),
        },
        {
            "md": (
                "## 8. Adapt: bounded fine-tuning of the last decoder blocks\n\n"
                "**Core concept — adaptation.** Fine-tuning continues the model's training on example pairs from the task. "
                "For each training pair the model is shown the English source and the reference so far, and its parameters are "
                "nudged so that the next reference token becomes more likely (*teacher forcing*; the error being reduced is the "
                "**training loss**).\n\n"
                "**What changes and what stays frozen.** Only the last `TRAINABLE_DECODER_LAYERS` decoder blocks train — two by "
                "default, 8,408,064 of 74,037,760 parameters (about 11 %). The encoder (which reads English), the shared "
                "vocabulary embeddings and the first four decoder blocks stay **frozen**, so the model cannot forget how to read "
                "English and the vocabulary cannot change.\n\n"
                "**Why the budget is bounded.** Two epochs over 1,200 pairs take about a minute on a CPU, produce an adapter of "
                "about 34 MB instead of a whole-model copy, and limit how far a small dataset can pull the model away from what "
                "it already knows. The build record's counter-examples: training the whole decoder (25 M parameters) or the whole "
                "model (74 M) gained nothing beyond the last two blocks on this corpus while costing more time and a far larger "
                "artifact.\n\n"
                "**Engineering — the recipe.** Teacher-forced cross-entropy on the Tagalog target, AdamW at a fixed learning "
                "rate, gradient clipping at 1.0, seeded shuffling, no scheduler; sources and targets are truncated to 128 "
                "SentencePiece pieces **during training only**.\n\n"
                "**Validation-based selection.** Epoch 0 records the *pretrained* model's validation chrF and BLEU; after every "
                "epoch the validation split is scored again, and the epoch with the highest validation chrF is kept. If training "
                "made the model worse, epoch 0 — the original model — would be kept. The test split is not touched.\n\n"
                "**What to notice:** training loss falling and validation chrF rising a few points over two epochs, "
                "`best_epoch`, and `trainable_parameters` against `total_parameters`. A falling training loss alone would not "
                "prove better translations — that is what the validation and test scores are for."
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
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'trainable_share': f\"{{adapt_result['n_trainable'] / adapt_result['n_total']:.1%}}\", 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": _t(
                """
**Checkpoint 3 — why validation, not test?** The kept epoch was chosen with the validation split. Why not simply choose the epoch with the best *test* score?

"""
                + _SAMPLE
                + """Choosing uses the data you choose with. If the test split picked the epoch, the reported test score would be optimistic — partly a measure of how well the choice happened to fit those 300 sentences — and no untouched data would be left for an honest final comparison. Keeping validation for decisions and test for the final report is what makes the test score a held-out estimate.

</details>
"""
            ),
        },
        {
            "md": _t(
                """
## 9. Evaluate on held-out sentences

**Evaluation practice — same sentences, same settings, three systems.** The test split was never used for training or for choosing the epoch, and no English source in it appears in the training split. The adapted model is scored exactly as the pretrained model was in Section 7, and the three systems — copy-source baseline, pretrained, adapted — are compared on the **same 300 sentences**. Three hundred sentences from one seeded split of one corpus give no dispersion estimate: the differences are sample evidence that the adaptation workflow works, not a benchmark, and a gain on Tatoeba says nothing about your domain until you measure it there.

The cell then translates the test split with the adapted model and pairs each sentence's three outputs. It reports how many outputs adaptation **changed**, and among those how many sentence-level chrF scores went up or down — **negative results are kept, not hidden**. Sentence chrF on a short sentence is noisy (one word can move it by twenty points), so use it to find sentences worth reading, not to rank them. The same first `SHOW_N` test rows as Section 6b are printed with all three outputs, and every row is written to `outputs/<<stem>>_test_predictions.csv`.

On the default sample the cell asserts that adapted chrF exceeds pretrained chrF: a regression guard that this pinned run still behaves as recorded, **not** a promise that adaptation always helps. With BYOD, the cell reports a non-improvement instead of stopping.

**What to notice:** the three corpus scores and their shared denominator; how many outputs changed; and that some sentence scores went *down* even if the corpus score went up.
"""
            ),
            "code": (
                "import csv\n\n"
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "comparison = {{\n"
                "    'chrf': {{'copy_source': round(baseline_copy['chrf'], 2), 'frozen': round(frozen_test['chrf'], 2), 'adapted': round(adapted_test['chrf'], 2)}},\n"
                "    'bleu': {{'copy_source': round(baseline_copy['bleu'], 2), 'frozen': round(frozen_test['bleu'], 2), 'adapted': round(adapted_test['bleu'], 2)}},\n"
                "    'delta_vs_frozen': {{'chrf': round(adapted_test['chrf'] - frozen_test['chrf'], 2), 'bleu': round(adapted_test['bleu'] - frozen_test['bleu'], 2)}},\n"
                "    'n_test_sentences': adapted_test['n'],\n"
                "}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n\n"
                "adapted_outputs = translate_all(test_records)\n"
                "for row, output in zip(paired, adapted_outputs, strict=True):\n"
                "    row['adapted'] = output\n"
                "    row['copy_source'] = row['source']\n"
                "    for system in ('copy_source', 'pretrained', 'adapted'):\n"
                "        row[f'chrf_{{system}}'] = round(chrf([row[system]], [row['reference']]), 1)\n"
                "    row['sentence_chrf'] = f\"copy {{row['chrf_copy_source']}} | pretrained {{row['chrf_pretrained']}} | adapted {{row['chrf_adapted']}}\"\n"
                "changed = [row for row in paired if row['adapted'] != row['pretrained']]\n"
                "n_up = sum(row['chrf_adapted'] > row['chrf_pretrained'] for row in changed)\n"
                "n_down = sum(row['chrf_adapted'] < row['chrf_pretrained'] for row in changed)\n"
                "changed_by_adaptation = {{'test_sentences': len(paired), 'output_changed': len(changed), 'sentence_chrf_up': n_up, 'sentence_chrf_down': n_down, 'sentence_chrf_equal': len(changed) - n_up - n_down, 'output_unchanged': len(paired) - len(changed)}}\n"
                "print({{'changed_by_adaptation': changed_by_adaptation}})\n"
                "show_rows(paired[:show_n], [('id', 'id'), ('English', 'source'), ('reference', 'reference'), ('pretrained', 'pretrained'), ('adapted', 'adapted'), ('sentence chrF', 'sentence_chrf')])\n"
                "with open('outputs/{stem}_test_predictions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=['id', 'source', 'reference', 'pretrained', 'adapted', 'chrf_copy_source', 'chrf_pretrained', 'chrf_adapted'])\n"
                "    writer.writeheader()\n"
                "    for row in paired:\n"
                "        writer.writerow({{key: row[key] for key in writer.fieldnames}})\n\n"
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
                "    'changed_by_adaptation': changed_by_adaptation,\n"
                "    'test_predictions': 'outputs/{stem}_test_predictions.csv',\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "if USE_BYOD:\n"
                "    if adapted_test['chrf'] <= frozen_test['chrf']:\n"
                "        print({{'note': 'adaptation did not raise held-out chrF on your data; report this result as it is'}})\n"
                "else:\n"
                "    assert adapted_test['chrf'] > frozen_test['chrf'], 'the pinned sample run no longer reproduces the recorded gain; report the comparison above'\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'predictions': 'outputs/{stem}_test_predictions.csv'}})"
            ),
        },
        {
            "md": _t(
                """
## 10. Interpret: was the meaning preserved?

**Evaluation practice — read for meaning.** A score says how much an output overlaps with one reference. Meaning is judged by reading, with a checklist:

| Check | Question |
|---|---|
| **Meaning** | Would a reader of the Tagalog understand what the English says? |
| **Negation** | Is every *not / never / no* carried over — and none added? |
| **Names** | Are people and places kept (or correctly rendered, e.g. *France → Pransiya*)? |
| **Numbers** | Are quantities the same, as digits or as words? |
| **Omissions** | Is any part of the English missing? |
| **Unsupported additions** | Does the Tagalog say something the English does not? |

**Automatic surface flags** help you find sentences worth reading. They are deliberately crude string checks, written for this notebook:

- `negation`: the English has a negation word and the output has none of the Tagalog negators *hindi, di, wala, walang, huwag, ayaw, ayoko*;
- `number`: a digit string in the English is missing from the output (a number written as a word is flagged too — it may be correct);
- `name`: a capitalised word after the first word of the English is missing from the output (translated names such as *Pransiya* are flagged too);
- `length`: the output is under half or over two and a half times the English length (a possible omission or addition);
- `untranslated`: the output equals the English.

**A flag is not an error, and no flag is not correctness.** To see how noisy each check is, the cell also runs it on the *human references*: flags that fire on a correct human translation show you the false-alarm rate. Then it lists the three changed sentences whose sentence chrF **fell** most after adaptation, and the three whose score rose most — a fixed rule — with their flags.

Your reading here is **qualitative**. This notebook ships no scoring rubric and no reviewed annotations, so do not turn your judgements into a percentage. If you cannot read Tagalog, record *uncertain* and rely only on the checks that do not need it (names, digits, length).
"""
            ),
            "code": _t(
                """
import re

EN_NEGATION = re.compile(r"\\b(?:not|never|no|nobody|nothing|none|nowhere|neither|nor|cannot)\\b|n't\\b", re.IGNORECASE)
TL_NEGATION = re.compile(r"\\b(?:hindi|di|wala|walang|huwag|ayaw|ayoko|ayokong)\\b", re.IGNORECASE)
DIGITS = re.compile(r"\\d+")
NAME = re.compile(r"[A-Z][a-z]+")
FLAG_NAMES = ('negation', 'number', 'name', 'length', 'untranslated')
SYSTEMS = ('reference', 'pretrained', 'adapted')


def surface_flags(source, output):
    # Cheap pointers to sentences worth reading: a flag is not an error, and no flag is not correctness.
    flags = []
    if EN_NEGATION.search(source) and not TL_NEGATION.search(output):
        flags.append('negation')
    if any(number not in output for number in DIGITS.findall(source)):
        flags.append('number')
    later_words = [word.strip('.,!?;:"()') for word in source.split()[1:]]
    if any(NAME.fullmatch(word) and word not in output for word in later_words):
        flags.append('name')
    ratio = len(output) / max(len(source), 1)
    if ratio < 0.5 or ratio > 2.5:
        flags.append('length')
    if output.strip().lower() == source.strip().lower():
        flags.append('untranslated')
    return flags


for row in paired:
    for system in SYSTEMS:
        row[f'flags_{system}'] = surface_flags(row['source'], row[system])
flag_counts = {name: {system: sum(name in row[f'flags_{system}'] for row in paired) for system in SYSTEMS} for name in FLAG_NAMES}
print(f'surface flags over {len(paired)} test sentences (the reference column is the false-alarm check on human translations):')
for name, counts in flag_counts.items():
    print({name: counts})

for row in changed:
    row['delta'] = round(row['chrf_adapted'] - row['chrf_pretrained'], 1)
    row['flags'] = f"pretrained {row['flags_pretrained'] or '-'} | adapted {row['flags_adapted'] or '-'}"
fell = sorted((row for row in changed if row['delta'] < 0), key=lambda row: row['delta'])[:3]
rose = sorted((row for row in changed if row['delta'] > 0), key=lambda row: row['delta'], reverse=True)[:3]
columns = [('id', 'id'), ('English', 'source'), ('reference', 'reference'), ('pretrained', 'pretrained'), ('adapted', 'adapted'), ('chrF change', 'delta'), ('flags', 'flags')]
print(f'\\nLargest sentence-chrF drops after adaptation ({len(fell)} shown):')
show_rows(fell, columns)
print(f'\\nLargest sentence-chrF gains after adaptation ({len(rose)} shown):')
show_rows(rose, columns)
"""
            ),
        },
        {
            "md": _t(
                """
**Checkpoint 4 — did adaptation introduce an error?** Look at the largest drops above. For each: is the adapted output actually *wrong* (a lost negation, a changed name, a missing clause, an addition), or is it a valid rewording that happens to share less with the reference? Then look at the flag table: which check fires most often on the human references, and what does that tell you about trusting that check?

"""
                + _SAMPLE
                + """A falling sentence score can be a real error or a harmless rewording; only reading tells them apart, ideally by someone who reads Tagalog. That is why corpus scores are paired with examples: a net gain in corpus chrF can hide individual sentences the adaptation made worse, and those are exactly the ones a user would be hurt by. A check that fires often on the human references (typically `name`, because references translate names such as *France*) has many false alarms — treat its flags as prompts to look, never as a count of errors. If you could not judge a sentence, the honest record is *uncertain*.

</details>
"""
            ),
        },
        {
            "md": _t(
                """
## 11. Controlled activity: change one meaning-bearing element

**Question:** *Does changing one meaning-bearing element produce the corresponding change in the translation?*

**Design.** Each pair below is two English sentences that differ in **exactly one element** — affirmative versus negative, one number, or one name. The model checkpoint (the adapted model, at its selected epoch) and the decoding settings (`NUM_BEAMS`, `GEN_MAX_NEW_TOKENS`) are held **fixed**, so a difference between the two translations can only come from the one change in the input.

**These are instructional probes**, written for this notebook — they are not Tatoeba data, the cell checks they appear in no split (and, on the default path, nowhere in the corpus), and they play no part in the test evaluation. They have **no reference translations and no bilingual review**, so the cell reports **no quality score** for them. Instead it applies one automatic cue per pair that does not require judging the whole sentence:

- **negation pairs:** a Tagalog negator (*hindi, di, wala, huwag, ayaw, ...*) is absent from the first output and present in the second;
- **number pairs:** each digit string appears in its own output (a number written as a Tagalog word would fail this cue yet may be correct — read it);
- **name pairs:** each name appears in its own output (a name correctly written the Tagalog way — *Manila* is *Maynila* — fails this cue yet is right; read it).

It also lists every other word that differs between the two outputs, so you can see whether **unrelated meaning** changed too.

**Predict first:** which category do you expect to fail most often? Write it down.

This is a small **exploratory** activity — seven hand-written, short sentences — **not a robustness benchmark**. It can reveal a failure; passing it says little about longer, ambiguous or domain-specific text.
"""
            ),
            "code": _t(
                """
import difflib

PROBE_PAIRS = [
    {'id': 'neg-1', 'category': 'negation', 'original': 'I like this song.', 'changed': 'I do not like this song.', 'cue': None},
    {'id': 'neg-2', 'category': 'negation', 'original': 'She is at home.', 'changed': 'She is not at home.', 'cue': None},
    {'id': 'neg-3', 'category': 'negation', 'original': 'The store is open today.', 'changed': 'The store is not open today.', 'cue': None},
    {'id': 'num-1', 'category': 'number', 'original': 'I paid 250 pesos for the ticket.', 'changed': 'I paid 400 pesos for the ticket.', 'cue': ('250', '400')},
    {'id': 'num-2', 'category': 'number', 'original': 'The meeting is in room 12.', 'changed': 'The meeting is in room 31.', 'cue': ('12', '31')},
    {'id': 'name-1', 'category': 'person name', 'original': 'Maria is my teacher.', 'changed': 'Jose is my teacher.', 'cue': ('Maria', 'Jose')},
    {'id': 'place-1', 'category': 'place name', 'original': 'We flew to Manila yesterday.', 'changed': 'We flew to Cebu yesterday.', 'cue': ('Manila', 'Cebu')},
]
PROBE_PROVENANCE = 'instructional probe written for this notebook; not from Tatoeba; no reference translation; no bilingual review'

seen_sources = {r['source'].strip().lower() for part in splits.values() for r in part}
if not USE_BYOD:
    seen_sources |= {source.strip().lower() for source, _target in read_corpus_pairs(corpus_bytes)}
probe_overlap = sorted({p['id'] for p in PROBE_PAIRS for text in (p['original'], p['changed']) if text.strip().lower() in seen_sources})
assert not probe_overlap, f'instructional probes must stay outside the evaluated data: {probe_overlap}'

probe_settings = {'model': f"adapted, selected epoch {adapt_result['best_epoch']}", 'num_beams': NUM_BEAMS, 'max_new_tokens': GEN_MAX_NEW_TOKENS}
probe_outputs = translate_all([{'source': text} for probe in PROBE_PAIRS for text in (probe['original'], probe['changed'])])
probe_rows = []
for index, probe in enumerate(PROBE_PAIRS):
    out_a, out_b = probe_outputs[2 * index], probe_outputs[2 * index + 1]
    if probe['cue'] is None:
        carried = not TL_NEGATION.search(out_a) and bool(TL_NEGATION.search(out_b))
        rule = 'no Tagalog negator in the first output, one in the second'
    else:
        expected_a, expected_b = probe['cue']
        carried = expected_a in out_a and expected_b in out_b
        rule = f'{expected_a!r} in the first output and {expected_b!r} in the second'
    tokens_a, tokens_b = tokenize(out_a), tokenize(out_b)
    differences = [
        f"{' '.join(tokens_a[i1:i2]) or '(nothing)'} -> {' '.join(tokens_b[j1:j2]) or '(nothing)'}"
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=tokens_a, b=tokens_b, autojunk=False).get_opcodes()
        if tag != 'equal'
    ]
    probe_rows.append({
        'id': probe['id'], 'category': probe['category'], 'original': probe['original'], 'changed': probe['changed'],
        'original_output': out_a, 'changed_output': out_b, 'cue_rule': rule, 'cue_carried_through': carried,
        'output_differences': '; '.join(differences) or 'none',
        'learner_intended_change_survived': 'not assessed', 'learner_unrelated_meaning_changed': 'not assessed',
        'provenance': PROBE_PROVENANCE,
    })

print({'settings_held_fixed': probe_settings})
for row in probe_rows:
    print('=' * 96)
    print(f"{row['id']} [{row['category']}]  automatic cue carried through: {row['cue_carried_through']}  ({row['cue_rule']})")
    print(f"   original: {row['original']}")
    print(f"          -> {row['original_output']}")
    print(f"    changed: {row['changed']}")
    print(f"          -> {row['changed_output']}")
    print(f"   words that differ between the two outputs: {row['output_differences']}")
print('=' * 96)
with open('outputs/<<stem>>_probes.csv', 'w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(probe_rows[0]))
    writer.writeheader()
    writer.writerows(probe_rows)
print({'probe_pairs': len(probe_rows), 'cue_carried_through': sum(row['cue_carried_through'] for row in probe_rows), 'quality_score': 'none (no references)', 'file': 'outputs/<<stem>>_probes.csv'})
"""
            ),
        },
        {
            "md": _t(
                """
**What to notice:** for each pair, whether the automatic cue held, and the list of *other* words that differ between the two outputs. A negation pair where the second output has no negator is a silent meaning flip. A long list of other differences means the model rewrote more than the one element — read it to decide whether the rest of the meaning survived.

**Record your judgement, honestly.** Open `outputs/<<stem>>_probes.csv` (Colab: the folder icon on the left) and fill in `learner_intended_change_survived` and `learner_unrelated_meaning_changed` with *yes*, *no* or *uncertain*. If you cannot read Tagalog, write *uncertain* for meaning and base any statement only on the cues — a guess is not an evaluation. The automatic cue is a string check, not a judgement: it can pass when the sentence is wrong and fail when a number is correctly written as a word.

### 11b. Change one thing yourself

**Predict → change one thing → run → observe → explain.** Type an English sentence and a copy of it with **one** meaning-bearing change — a negation, a number, a name, a place — into the two form fields, predict what should change in the Tagalog, then run the cell. Checkpoint and decoding stay fixed. Leave the fields empty and the cell does nothing, so **Run all** is unaffected.
"""
            ),
            "code": _t(
                """
MY_ORIGINAL = ''  # @param {type:"string"}
MY_CHANGED = ''  # @param {type:"string"}

if MY_ORIGINAL.strip() and MY_CHANGED.strip():
    mine = [MY_ORIGINAL.strip(), MY_CHANGED.strip()]
    if {text.lower() for text in mine} & {r['source'].strip().lower() for r in test_records}:
        print('Note: one of your sentences is in the test split, so it is not outside the evaluation data.')
    my_outputs = translate_all([{'source': text} for text in mine])
    print({'settings_held_fixed': probe_settings})
    for text, output in zip(mine, my_outputs, strict=True):
        print(f'{text}\\n    -> {output}')
    tokens_a, tokens_b = tokenize(my_outputs[0]), tokenize(my_outputs[1])
    differences = [f"{' '.join(tokens_a[i1:i2]) or '(nothing)'} -> {' '.join(tokens_b[j1:j2]) or '(nothing)'}" for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=tokens_a, b=tokens_b, autojunk=False).get_opcodes() if tag != 'equal']
    print({'words_that_differ': differences or 'none'})
else:
    print('Skipped: type an original and a changed English sentence into the form fields, then run this cell again.')
"""
            ),
        },
        {
            "md": _t(
                """
**Checkpoint 5 — what did the controlled activity show?** Did any change fail to carry through? Did any pair change more than the one element? What can seven hand-written pairs tell you, and what can they not?

"""
                + _SAMPLE
                + """Where the cue holds and little else differs, the change probably carried through; a missing negator in the negated sentence, or a number that disappears, is a silent meaning change — the most dangerous kind of error, because the output still reads fluently. A failed cue is not automatically an error: a name written the Tagalog way (*Maynila*) is correct. And a passed cue is not automatically success: if other words changed too, read whether the rest of the meaning survived. But seven short, simple, hand-written pairs cannot estimate how often such errors occur, and they say nothing about long, ambiguous or technical text. The honest conclusion is exploratory: *"in these probes, the change carried through in N of 7 pairs by the automatic cue; I could / could not confirm the meaning myself."*

</details>
"""
            ),
        },
        {
            "md": _t(
                """
## 12. New sentences, export and reload

**Engineering — the artifact.** Six Tatoeba sentences that were in none of the splits are translated by the adapted model through the same `translate` contract as Section 6, and scored with `evaluation_report` — which returns a `measured-small-sample` verdict when references are supplied, because six sentences carry no dispersion estimate.

`pipe.save_artifact` writes the trained tensors — the last two decoder blocks, about 34 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, the digest of the base pickle checkpoint, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `MarianMTTranslationPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest and digest **before** deserialising, refuses any tensor that is not an adaptable decoder tensor, and overlays the tensors onto a freshly loaded base — a new object built from files, not the model in memory. The cell asserts identical translations before and after.

**Matching reloaded outputs verifies artifact fidelity, not translation quality:** it proves the file on disk reproduces the adapted model exactly. Whether that model translates well is the evidence of Sections 7–11.

**What to notice:** `reload_parity` equal to 8 of 8, the adapter size and digest, and the list of files in `outputs/`.
"""
            ),
            "code": (
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
                "    'changed_by_adaptation': changed_by_adaptation,\n"
                "    'surface_flags': {{'denominator': len(paired), 'counts': flag_counts}},\n"
                "    'probes': {{'pairs': len(probe_rows), 'cue_carried_through': sum(row['cue_carried_through'] for row in probe_rows), 'settings': probe_settings, 'file': 'outputs/{stem}_probes.csv', 'provenance': PROBE_PROVENANCE}},\n"
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
        {
            "md": _t(
                """
## 13. Conclude from the evidence

**Evaluation practice — say what the evidence supports, and no more.** The next cell fills in the measured part of a conclusion from *this run's* numbers. The bracketed parts are yours: name one specific improvement or failure you actually read (quote the sentence), your observation from the controlled activity, and one broader claim these results do **not** establish. If you could not judge meaning in Tagalog, say so in the sentence rather than guessing.
"""
            ),
            "code": _t(
                """
import textwrap

split_label = f'the {CORPUS_NAME} en-tl {CORPUS_RELEASE} test split' if not USE_BYOD else f'the test split of {data_source}'
conclusion_draft = (
    f"On {split_label} ({adapted_test['n']} sentences), the adapted model achieved chrF {adapted_test['chrf']:.1f} and BLEU {adapted_test['bleu']:.1f}, "
    f"compared with chrF {frozen_test['chrf']:.1f} / BLEU {frozen_test['bleu']:.1f} for the pretrained model and chrF {baseline_copy['chrf']:.1f} / BLEU {baseline_copy['bleu']:.1f} for the copy-source baseline. "
    f"Adaptation changed {changed_by_adaptation['output_changed']} of {changed_by_adaptation['test_sentences']} test translations; sentence chrF rose for {changed_by_adaptation['sentence_chrf_up']} and fell for {changed_by_adaptation['sentence_chrf_down']}. "
    "Inspection showed [one specific improvement or failure, quoting the sentence]. "
    f"The controlled activity ({len(probe_rows)} instructional probe pairs; automatic cue carried through in {sum(row['cue_carried_through'] for row in probe_rows)}) suggested [your observation about negation, numbers or names], "
    "but these results do not establish [a broader claim the evidence does not support]."
)
print(textwrap.fill(conclusion_draft, width=100, break_on_hyphens=False))
"""
            ),
        },
    ],
    "closing": _t(
        """
The scaffold you are completing:

> On **[dataset and split]**, the adapted model achieved **[results]** compared with **[baseline and pretrained model]**. Inspection showed **[specific improvement or failure]**. The controlled activity suggested **[observation]**, but these results do not establish **[broader claim]**.

**Completion records are optional.** If a course asks you for one, the printed draft with your brackets filled in, plus your annotated `outputs/<<stem>>_probes.csv`, is enough; this notebook does not require any submission.

## Interpretation and limits

The pretrained model already translates Tatoeba-style sentences well — its test BLEU sits near where the upstream README puts it — and a bounded fine-tuning of the last two decoder blocks on 1,200 in-domain pairs lifted held-out chrF by a few points and BLEU by several in the recorded runs, with a 34 MB adapter that reloads to identical translations. Read those gains together with the sentence-level view from Sections 9 and 10: adaptation changes many individual translations, and some of them get worse.

The test split is 300 sentences from one seeded split of one corpus, with no dispersion estimate; the metrics are two reference-based overlap scores (own implementations, not sacrebleu-identical, and neither a human judgement); Tatoeba is short, conversational and already in the model's training lineage; and the controlled activity is seven unreviewed, hand-written probe pairs. So a gain here says the workflow works, not that the adapted model is better on your domain, that it handles long or technical text, or that its fluent output is faithful — a translation can drop a negation, change a number or leave an entity in English and still score well. Fine-tuning on a narrow corpus can also erode the model elsewhere; nothing here measures that.

Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone notebook, can acquire and digest-verify the pinned model snapshot (including its pickle weight file), fetch and digest-verify a real parallel corpus, validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded fine-tuning, evaluate against a trivial baseline and the pretrained model on an independent split, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark superiority, translation quality on any other domain, a usable acceptance threshold, or production fitness.

### Before using these translations in practice

Evidence you would still need — none of it is produced by this notebook:

- **References from your own domain**, with the copy-source baseline and the pretrained model scored on them before any adapted number is trusted.
- **Bilingual human review** by qualified Tagalog readers with a defined rubric (for example adequacy, fluency and error categories), so judgements can be counted, not just described.
- **Targeted checks on your content** for negation, numbers, names, dates and terminology, at a scale large enough to estimate error rates.
- **Evidence on realistic inputs**: long sentences, documents (out of scope here), and text unlike Tatoeba.
- **Variability**: several splits or seeds, or confidence intervals, so a gain can be told apart from noise.
- **A use policy**: where translations need human post-editing (health, legal, safety and public-facing text almost always do), and the licences of any data you adapt on.

Three habits to carry to real data. **References first:** read the copy-source baseline and the pretrained score on *your* references before any adapted one. **Leakage:** de-duplicate sources across splits (the contract does this case-insensitively) and split by document or session when your pairs come from one. **Ceilings:** inputs over `MAX_INPUT_TOKENS` are refused at inference and truncated to 128 pieces only during training — long-document translation is out of scope.

## Optional experiments

None of these affects the default path. Change **one** knob, re-run from its cell, and compare with the recorded run:

- set `TRAINABLE_DECODER_LAYERS = 6` to train the whole decoder, and compare the adapter size and the test scores;
- raise `EPOCHS` and watch whether validation chrF keeps rising or turns down — and which epoch is kept;
- set `NUM_BEAMS = 1` (greedy decoding) and compare the scores and the probe outputs;
- add your own minimal pairs in Section 11b — try a date, a plural, or *you* singular versus plural.

## Bring your own data (optional)

1. Prepare a CSV with columns `id`, `source` (English) and `target` (Tagalog reference), or a JSON / JSONL list of such records: unique ids, 8..20,000 records, each text 1..4,000 characters.
2. In Section 4 set `USE_BYOD = True`, and either leave `BYOD_PATH` empty to get an upload dialog or set it to the file's path in the runtime.
3. Run from Section 4 downwards. Your pairs pass through the same validation, split, baselines, adaptation, evaluation, meaning checks, probes, export and reload.
4. Do not upload confidential or restricted text to a hosted runtime unless you are authorised to process it there. Read the copy-source baseline and the pretrained score on your data before the adapted one.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| Section 1 stops with *Restart the runtime, then rerun from the top* | the runtime had different package versions loaded | **Runtime → Restart session**, then **Run all** again; this is expected once on some runtimes |
| Download error or timeout in Section 3 or 4 | network access to huggingface.co or object.pouta.csc.fi | re-run the cell; only missing files are fetched again. The default path needs both hosts |
| `verify_snapshot` or `fetch_corpus` reports a size or SHA-256 mismatch | a partial or altered download | delete `weights/opus-mt-en-tl/pytorch_model.bin` (or `weights/tatoeba-en-tl/`) and re-run the cell; never bypass the check |
| One `Recommended: pip install sacremoses.` warning | expected in this environment | nothing to do; see the Environment note |
| Very slow, or out of memory | CPU runtime, or a large BYOD file | CPU is fine for the default run; for speed choose **Runtime → Change runtime type → T4 GPU**; lower `BATCH_SIZE` to 8 if memory runs out |
| `ValueError` naming a record, field or ceiling | BYOD data or a form value outside the contract | fix the named record or value (see Prerequisites) and re-run from that cell |
| The Section 9 assertion fails on the default path | the pinned run no longer reproduces the recorded gain | do not edit the cell; report the printed comparison and your runtime |
| A form change has no effect | later cells still hold the old values | re-run from the changed cell downwards |

## Glossary

- **Adapter** — the file holding only the fine-tuned parameters (here the last two decoder blocks); it is applied on top of the unchanged base model.
- **Baseline** — a simple reference system every real result is compared with; here, the **copy-source baseline** that returns the English unchanged.
- **Beam search** — decoding that keeps the few most promising partial translations at each step and returns the best finished one.
- **BLEU** — overlap of word sequences (1–4 words) between output and reference, with a brevity penalty; 0–100, not a percentage correct.
- **Checkpoint** — a saved set of model parameters; "the selected checkpoint" is the epoch kept by validation.
- **chrF** — overlap of character sequences (1–6 characters) between output and reference, recall-weighted; 0–100, not a percentage correct.
- **Corpus-level score** — a score computed from matches pooled over all sentences, not an average of per-sentence scores.
- **Decoder / encoder** — the two halves of the model: the encoder reads the English, the decoder writes the Tagalog.
- **Epoch** — one pass over all training pairs.
- **Fine-tuning / adaptation** — continuing training on task examples; here only a small part of the model changes.
- **Frozen parameters** — parameters that are not updated during adaptation.
- **Held-out (test) split** — sentences kept out of training and selection, used only for the final comparison.
- **Instructional probe** — a sentence written for teaching, not drawn from the evaluated data, and never scored for quality here.
- **Parallel corpus** — a collection of sentences paired with their translations.
- **Pretrained model** — the model as published, before any adaptation in this notebook.
- **Reference** — one human translation of a source sentence; one acceptable answer, not the only one.
- **Token** — a word or word-piece from the model's vocabulary; the model writes one token at a time.
- **Validation split** — sentences used to make choices during training (which epoch to keep), separate from the test split.

## References

- Repository README: https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/README.md
- Repository model card: https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/MODEL_CARD.md
- Weight provenance and the pickle trust boundary: https://github.com/kurtvalcorza/marianmt-en-tl-translation-pipeline/blob/main/docs/WEIGHTS.md
- Upstream model: https://huggingface.co/<<MODEL_ID>>
- Upstream training code: https://github.com/Helsinki-NLP/OPUS-MT-train
- OPUS-MT — Building open translation services for the World (Tiedemann & Thottingal, EAMT 2020): https://aclanthology.org/2020.eamt-1.61
- Marian: Fast Neural Machine Translation in C++ (Junczys-Dowmunt et al., ACL 2018): https://arxiv.org/abs/1804.00344
- Tatoeba en–tl via OPUS (Tiedemann, LREC 2012; corpus release v2023-04-12, CC BY 2.0 FR): https://opus.nlpl.eu/Tatoeba-v2023-04-12.php
- chrF: character n-gram F-score for automatic MT evaluation (Popović, WMT 2015): https://aclanthology.org/W15-3049
- BLEU: a method for automatic evaluation of machine translation (Papineni et al., ACL 2002): https://aclanthology.org/P02-1040
- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
"""
    ),
}
