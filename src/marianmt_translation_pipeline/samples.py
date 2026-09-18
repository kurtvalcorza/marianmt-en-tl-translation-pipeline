"""Parallel-corpus dataset contract for fine-tuning: the pinned Tatoeba en–tl sample, validation,
seeded splitting, BYOD loaders and CSV export.

The default dataset is **real**: the English–Tagalog sentence pairs of the Tatoeba corpus as
distributed by OPUS (release v2023-04-12, Moses format, 8,785 pairs, CC BY 2.0 FR with attribution to
the Tatoeba contributors). One 312 KB zip is fetched from OPUS's object store, refused on any byte-size
or SHA-256 mismatch, and read member by member (no `extractall`). Pairs are filtered to 3..200
characters a side and de-duplicated on the lower-cased English side, so a source sentence can appear in
only one split; the split is a seeded shuffle with fixed train/validation/test sizes.

Why Tatoeba: it is the corpus the upstream README reports BLEU 26.6 / chrF 0.577 on — the pinned model
was trained on OPUS data that predates this 2023 release, so the frozen model is strong here and the
fine-tuning question is whether a small in-domain adaptation still moves held-out chrF and BLEU (the
build record says it does). A record is ``{id, source, target}``: an English sentence and its Tagalog
reference.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_TEXT_CHARS, MODEL_ID

CORPUS_NAME = "Tatoeba"
CORPUS_RELEASE = "v2023-04-12"
CORPUS_URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-tl.txt.zip"
CORPUS_BYTES = 312_459
CORPUS_SHA256 = "9abddc7d2fcea307e8582285617d1a10246fdb9050b6c6345243cb4b9991f40b"
CORPUS_MEMBERS = {"source": "Tatoeba.en-tl.en", "target": "Tatoeba.en-tl.tl"}
CORPUS_LICENSE = "CC BY 2.0 FR (Tatoeba contributors; OPUS redistribution)"
CORPUS_PAIRS = 8_785
DEFAULT_CACHE_DIR = Path("weights") / "tatoeba-en-tl"
MIN_PAIR_CHARS = 3
MAX_PAIR_CHARS = 200
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 1_200, "validation": 200, "test": 300}
MIN_RECORDS = 8
MAX_RECORDS = 20_000
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> bytes:
    """Return the pinned corpus zip bytes from the cache or the OPUS object store, digest-verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "en-tl.txt.zip"
    if local.is_file():
        data = local.read_bytes()
        if len(data) == CORPUS_BYTES and _sha256_bytes(data) == CORPUS_SHA256:
            return data
    if fetcher is not None:
        data = fetcher(CORPUS_URL)
    else:
        with urllib.request.urlopen(CORPUS_URL, timeout=120) as response:  # noqa: S310 (pinned https URL)
            data = response.read()
    if len(data) != CORPUS_BYTES or _sha256_bytes(data) != CORPUS_SHA256:
        raise ValueError(
            f"corpus zip: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
            f"pinned {CORPUS_BYTES} / {CORPUS_SHA256[:16]}…"
        )
    local.write_bytes(data)
    return data


def read_corpus_pairs(data: bytes) -> list[tuple[str, str]]:
    """Aligned (English, Tagalog) lines from the Moses members, read without extracting to disk."""
    archive = zipfile.ZipFile(io.BytesIO(data))
    names = set(archive.namelist())
    for member in CORPUS_MEMBERS.values():
        if member not in names:
            raise ValueError(f"corpus zip is missing member {member}")
    source = archive.read(CORPUS_MEMBERS["source"]).decode("utf-8").splitlines()
    target = archive.read(CORPUS_MEMBERS["target"]).decode("utf-8").splitlines()
    if len(source) != len(target):
        raise ValueError(f"corpus members are not aligned: {len(source)} vs {len(target)} lines")
    return [(s.strip(), t.strip()) for s, t in zip(source, target, strict=True)]


def filter_pairs(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep pairs with both sides in MIN_PAIR_CHARS..MAX_PAIR_CHARS, one pair per lower-cased source."""
    seen: set[str] = set()
    kept = []
    for source, target in pairs:
        if not (
            MIN_PAIR_CHARS <= len(source) <= MAX_PAIR_CHARS
            and MIN_PAIR_CHARS <= len(target) <= MAX_PAIR_CHARS
        ):
            continue
        key = source.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append((source, target))
    return kept


def build_sample_dataset(
    pairs: Sequence[tuple[str, str]], *, seed: int = SAMPLE_SEED, sizes: Mapping[str, int] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of the filtered pairs cut into the named split sizes; ids carry the split name."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    kept = filter_pairs(pairs)
    total = sum(sizes.values())
    if total > len(kept):
        raise ValueError(f"requested {total} records but only {len(kept)} filtered pairs are available")
    order = list(range(len(kept)))
    random.Random(seed).shuffle(order)
    splits: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name, size in sizes.items():
        splits[name] = [
            {"id": f"{name}-{i:04d}", "source": kept[j][0], "target": kept[j][1]}
            for i, j in enumerate(order[cursor : cursor + size])
        ]
        cursor += size
    return splits


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus_pairs(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/source/target")
    for key in ("id", "source", "target"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid, source, target = record["id"], record["source"], record["target"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    for key, value in (("source", source), ("target", target)):
        if not isinstance(value, str):
            raise ValueError(f"{label}: {key} must be a string")
        if not value.strip():
            raise ValueError(f"{label}: {key} is empty")
        if len(value) > MAX_TEXT_CHARS:
            raise ValueError(
                f"{label}: {key} has {len(value)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}"
            )
    return {"id": rid, "source": source.strip(), "target": target.strip()}


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a parallel dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, source, target} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    sources: set[str] = set()
    identical = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        sources.add(item["source"].lower())
        identical += item["source"] == item["target"]
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_sources": len(sources),
        "identical_pairs": identical,
        "source_chars": {
            "min": min(len(r["source"]) for r in checked),
            "max": max(len(r["source"]) for r in checked),
        },
        "target_chars": {
            "min": min(len(r["target"]) for r in checked),
            "max": max(len(r["target"]) for r in checked),
        },
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], r["source"], r["target"]] for r in records]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no lower-cased English source appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record["source"]).lower()
            if key in seen and seen[key] != name:
                raise ValueError(f"source {record['source']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test after de-duplicating sources."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = record["source"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(record)
    random.Random(seed).shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = round(len(unique) * val_fraction)
    splits = {
        "test": unique[:n_test],
        "validation": unique[n_test : n_test + n_val],
        "train": unique[n_test + n_val :],
    }
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, source, target}` records from CSV (columns id, source, target), a JSON array or JSONL."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        missing = {"id", "source", "target"} - set(rows[0].keys() if rows else set())
        if missing:
            raise ValueError(f"CSV is missing columns {sorted(missing)}")
        return [{"id": r["id"], "source": r["source"], "target": r["target"]} for r in rows]
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .csv, .json or .jsonl")


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "source", "target"])
        writer.writeheader()
        for record in records:
            writer.writerow({"id": record["id"], "source": record["source"], "target": record["target"]})
    return out
