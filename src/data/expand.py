"""Добирает train до заданного размера, не трогая тексты val и test."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from src.data.collect import REVIEW_SOURCES, iter_zip_rows, prepare_rows
from src.data.prepare import _usable, repair_text, tokenize
from src.data.validate import VALID_LABELS, normalized_text

LABEL_FIELDS = ("review_id", "text", "label", "source", "label_origin")
RAW_FIELDS = ("text", "label", "source")


def equal_target(
    train_counts: dict[tuple[str, str], int],
    available: dict[tuple[str, str], int],
) -> int:
    """Сколько строк может быть в каждой ячейке train, если ячейки равны."""
    keys = [(source, label) for source in REVIEW_SOURCES for label in VALID_LABELS]
    return min(train_counts.get(key, 0) + available.get(key, 0) for key in keys)


def fill_to_total(
    candidates: dict[tuple[str, str], list[dict[str, str]]],
    train_counts: dict[tuple[str, str], int],
    target_total: int,
    seed: int,
) -> list[dict[str, str]]:
    """Поднимает ячейки по кругу, пока train не достигнет target_total.

    Пустая ячейка остаётся как есть. Остальные добираются из своих запасов.
    """
    keys = [(source, label) for source in REVIEW_SOURCES for label in VALID_LABELS]
    rng = random.Random(seed)
    pools: dict[tuple[str, str], list[dict[str, str]]] = {}
    for key in keys:
        bucket = list(candidates.get(key, []))
        rng.shuffle(bucket)
        pools[key] = bucket
    counts = {key: train_counts.get(key, 0) for key in keys}
    chosen: list[dict[str, str]] = []
    need = target_total - sum(counts.values())
    while need > 0:
        open_keys = [key for key in keys if pools[key]]
        if not open_keys:
            break
        level = min(counts[key] for key in open_keys)
        for key in open_keys:
            if need <= 0 or counts[key] != level:
                continue
            chosen.append(pools[key].pop())
            counts[key] += 1
            need -= 1
    return chosen


def choose_additions(
    candidates: dict[tuple[str, str], list[dict[str, str]]],
    train_counts: dict[tuple[str, str], int],
    seed: int,
) -> list[dict[str, str]]:
    """Добирает каждую ячейку до общего потолка. Уже занятые тексты в список не входят."""
    available = {key: len(rows) for key, rows in candidates.items()}
    target = equal_target(train_counts, available)
    rng = random.Random(seed)
    chosen: list[dict[str, str]] = []
    for key in sorted(candidates):
        bucket = list(candidates[key])
        rng.shuffle(bucket)
        need = target - train_counts.get(key, 0)
        if need < 0:
            raise ValueError(f"В train уже больше потолка: {key}")
        chosen.extend(bucket[:need])
    return chosen


def usable_additions(
    rows: list[dict[str, str]],
    used_texts: set[str],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Оставляет новые пригодные отзывы, которых нет в текущих частях выборки."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        repaired = repair_text(row["text"])
        key = normalized_text(repaired)
        if key in used_texts or not _usable(tokenize(repaired)):
            continue
        used_texts.add(key)
        grouped[(row["source"], row["label"])].append({**row, "text": repaired})
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Увеличить train, не трогая val и test")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--target", type=int, default=20000)
    parser.add_argument("--labels", type=Path, default=Path("data/interim/labels.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/reviews.csv"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--provenance", type=Path, default=Path("data/raw/provenance.json"))
    args = parser.parse_args()

    from src.data.prepare import frozen_splits, prepare_dataset, write_processed

    labels = _read_csv(args.labels)
    used = {normalized_text(row["text"]) for row in labels}
    for name in ("train", "val", "test"):
        for row in _read_csv(args.processed / f"{name}.csv"):
            used.add(normalized_text(row["text"]))
    train_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in _read_csv(args.processed / "train.csv"):
        train_counts[(row["source"], row["label"])] += 1
    corpus, _skipped = prepare_rows(iter_zip_rows(args.zip))
    candidates = usable_additions(corpus, used)
    additions = fill_to_total(candidates, train_counts, args.target, seed=42)
    next_id = max(int(row["review_id"]) for row in labels) + 1
    new_rows = []
    for offset, row in enumerate(additions):
        new_rows.append(
            {
                "review_id": str(next_id + offset),
                "text": row["text"],
                "label": row["label"],
                "source": row["source"],
                "label_origin": "source",
            }
        )
    _append_csv(args.labels, LABEL_FIELDS, new_rows)
    _append_csv(args.raw, RAW_FIELDS, new_rows)
    _update_provenance(args.provenance, labels + new_rows, args.target)
    frozen = frozen_splits(args.processed)
    prepared = prepare_dataset(_read_csv(args.labels), frozen=frozen)
    prepared["notes"] = [
        "Train добран до заданного размера: короткая ячейка не останавливает остальные.",
        "Val и test не менялись.",
    ]
    write_processed(args.processed, prepared)
    print(f"Добавлено в train: {len(new_rows)}")
    splits = prepared["splits"]
    assert isinstance(splits, dict)
    print(f"train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")
    vocab = prepared["vocab"]
    assert isinstance(vocab, dict)
    print(f"Словарь: {len(vocab)}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerows(rows)


def _update_provenance(path: Path, labels: list[dict[str, str]], target: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_source: dict[str, int] = defaultdict(int)
    by_label: dict[str, int] = defaultdict(int)
    for row in labels:
        by_source[row["source"]] += 1
        by_label[row["label"]] += 1
    payload["sample_rows"] = len(labels)
    payload["counts_by_source"] = dict(sorted(by_source.items()))
    payload["counts_by_label"] = dict(sorted(by_label.items()))
    payload["train_target_rows"] = target
    payload.pop("train_extended_to_equal_ceiling", None)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
