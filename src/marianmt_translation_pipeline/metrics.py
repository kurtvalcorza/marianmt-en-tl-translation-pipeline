"""Translation quality metrics and the trivial baseline every adapted number is read against.

Two corpus-level scores are implemented here in plain Python so the notebook needs no scorer
dependency: **chrF** (character n-gram F-score, n = 1..6 with spaces removed, beta = 2, precision and
recall averaged over the orders — the sacrebleu ``chrF2`` recipe) and **BLEU** (corpus BLEU-4 with a
simple regex tokeniser and the brevity penalty; sacrebleu-*style*, not sacrebleu-identical, because the
``13a`` tokeniser is not reproduced). chrF is the headline: Tagalog is morphologically rich and
character n-grams reward partially correct affixed words that word-level BLEU counts as wrong. The
**copy-source baseline** submits the English input as the translation; it scores what a system that
does nothing scores, and it is where any translation model must start from.
"""

from __future__ import annotations

import collections
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

CHRF_MAX_ORDER = 6
CHRF_BETA = 2.0
BLEU_MAX_ORDER = 4
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _char_ngrams(text: str, order: int) -> collections.Counter:
    compact = text.replace(" ", "")
    return collections.Counter(compact[i : i + order] for i in range(len(compact) - order + 1))


def chrf(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus chrF (0..100) over aligned hypothesis/reference lists."""
    if len(hypotheses) != len(references) or not references:
        raise ValueError("hypotheses and references must be non-empty and equal in length")
    precisions, recalls = [], []
    for order in range(1, CHRF_MAX_ORDER + 1):
        matched = total_h = total_r = 0
        for hyp, ref in zip(hypotheses, references, strict=True):
            grams_h, grams_r = _char_ngrams(hyp, order), _char_ngrams(ref, order)
            matched += sum((grams_h & grams_r).values())
            total_h += sum(grams_h.values())
            total_r += sum(grams_r.values())
        precisions.append(matched / total_h if total_h else 0.0)
        recalls.append(matched / total_r if total_r else 0.0)
    precision = sum(precisions) / CHRF_MAX_ORDER
    recall = sum(recalls) / CHRF_MAX_ORDER
    if precision + recall == 0.0:
        return 0.0
    beta2 = CHRF_BETA**2
    return 100.0 * (1 + beta2) * precision * recall / (beta2 * precision + recall)


def tokenize(text: str) -> list[str]:
    """Lower-cased word/punctuation tokens for BLEU (a regex tokeniser, not Moses/13a)."""
    return _TOKEN.findall(text.lower())


def bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus BLEU-4 (0..100) with uniform n-gram weights and the standard brevity penalty."""
    if len(hypotheses) != len(references) or not references:
        raise ValueError("hypotheses and references must be non-empty and equal in length")
    matched = [0] * BLEU_MAX_ORDER
    total = [0] * BLEU_MAX_ORDER
    hyp_len = ref_len = 0
    for hyp, ref in zip(hypotheses, references, strict=True):
        toks_h, toks_r = tokenize(hyp), tokenize(ref)
        hyp_len += len(toks_h)
        ref_len += len(toks_r)
        for order in range(1, BLEU_MAX_ORDER + 1):
            grams_h = collections.Counter(
                tuple(toks_h[i : i + order]) for i in range(len(toks_h) - order + 1)
            )
            grams_r = collections.Counter(
                tuple(toks_r[i : i + order]) for i in range(len(toks_r) - order + 1)
            )
            matched[order - 1] += sum((grams_h & grams_r).values())
            total[order - 1] += max(len(toks_h) - order + 1, 0)
    if min(matched) == 0 or min(total) == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matched, total, strict=True)) / BLEU_MAX_ORDER
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100.0 * brevity * math.exp(log_precision)


def translation_metrics(hypotheses: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    """chrF and BLEU plus the counts they were computed over."""
    return {
        "n": len(references),
        "chrf": chrf(hypotheses, references),
        "bleu": bleu(hypotheses, references),
        "hypothesis_chars": sum(len(h) for h in hypotheses),
        "reference_chars": sum(len(r) for r in references),
        "definitions": {
            "chrf": (
                "corpus chrF, character 1..6-grams, spaces removed, beta=2 "
                "(sacrebleu chrF2 recipe, own implementation)"
            ),
            "bleu": (
                "corpus BLEU-4, regex word/punctuation tokens lower-cased, brevity penalty "
                "(sacrebleu-style, not 13a)"
            ),
        },
    }


def copy_source_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score the English source as if it were the translation: the do-nothing baseline."""
    if not records:
        raise ValueError("records must be non-empty")
    sources = [str(r["source"]) for r in records]
    targets = [str(r["target"]) for r in records]
    metrics = translation_metrics(sources, targets)
    metrics["baseline"] = "copy source (the English input submitted as the Tagalog output)"
    return metrics
