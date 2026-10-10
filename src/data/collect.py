"""Сбор сырой выборки отзывов из архива Kaggle."""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

LABEL_NAMES = {
    "0": "neutral",
    "1": "positive",
    "2": "negative",
}

REVIEW_SOURCES = (
    "anime",
    "bank",
    "geo",
    "kinopoisk",
    "perekrestok",
    "ru-reviews-classification",
    "rureviews",
    "sber",
)

SOURCE_NOTES = {
    "anime": "отзывы об аниме",
    "bank": "отзывы о банках",
    "geo": "отзывы о местах",
    "kinopoisk": "отзывы о фильмах",
    "perekrestok": "отзывы на продукты",
    "ru-reviews-classification": "отзывы на товары",
    "rureviews": "отзывы на одежду и аксессуары",
    "sber": "отзывы на банковское приложение",
}

EXCLUDED_SOURCES = {
    "linis": "комментарии Linis Crowd, это не отзывы",
    "news": "новости, это не отзывы",
    "rusentiment": "посты соцсетей, это не отзывы",
}

DATASET_URL = "https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset"
DEFAULT_PER_CELL = 200
DEFAULT_SEED = 42


def iter_zip_rows(zip_path: Path) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"В архиве ожидается один CSV, найдено: {names}")
        with archive.open(names[0]) as raw:
            reader = csv.DictReader(line.decode("utf-8-sig") for line in raw)
            yield from reader


def prepare_rows(records: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Оставляет отзывы выбранных сфер и убирает повторы текста."""
    seen_texts: set[str] = set()
    prepared: list[dict[str, str]] = []
    skipped_duplicates = 0
    for record in records:
        source = record.get("src", "").strip()
        if source not in REVIEW_SOURCES:
            continue
        label_code = record.get("label", "").strip()
        if label_code not in LABEL_NAMES:
            continue
        text = record.get("text", "").strip()
        if not text or text in seen_texts:
            skipped_duplicates += int(bool(text))
            continue
        seen_texts.add(text)
        prepared.append(
            {
                "text": text,
                "label": LABEL_NAMES[label_code],
                "source": source,
            }
        )
    return prepared, skipped_duplicates


def sample_rows(
    rows: list[dict[str, str]],
    per_cell: int,
    seed: int,
) -> list[dict[str, str]]:
    """Берёт одинаковое число отзывов из каждой пары сфера-класс."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["label"])].append(row)

    rng = random.Random(seed)
    chosen: list[dict[str, str]] = []
    for key in sorted(groups):
        bucket = list(groups[key])
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_cell])
    rng.shuffle(chosen)
    return chosen


def write_reviews(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def write_provenance(
    path: Path,
    rows: list[dict[str, str]],
    source_rows: int,
    skipped_duplicates: int,
    per_cell: int,
    seed: int,
) -> None:
    by_source: dict[str, int] = defaultdict(int)
    by_label: dict[str, int] = defaultdict(int)
    for row in rows:
        by_source[row["source"]] += 1
        by_label[row["label"]] += 1
    payload = {
        "title": "Выборка отзывов на русском для классификации тональности",
        "upstream": DATASET_URL,
        "upstream_rows": source_rows,
        "label_map": LABEL_NAMES,
        "included_sources": SOURCE_NOTES,
        "excluded_sources": EXCLUDED_SOURCES,
        "sample_rows": len(rows),
        "per_cell": per_cell,
        "seed": seed,
        "skipped_duplicate_texts": skipped_duplicates,
        "counts_by_source": dict(sorted(by_source.items())),
        "counts_by_label": dict(sorted(by_label.items())),
        "notes": [
            "Метки корпуса: 0 neutral, 1 positive, 2 negative.",
            "Метки автоматические и иногда расходятся с текстом.",
            "Ручная проверка пилота относится к задаче T1.2.",
            "Отдельных полей с именами нет, но текст может их содержать.",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(zip_path: Path, output_dir: Path, per_cell: int, seed: int) -> int:
    records = list(iter_zip_rows(zip_path))
    prepared, skipped_duplicates = prepare_rows(records)
    sampled = sample_rows(prepared, per_cell, seed)
    write_reviews(output_dir / "reviews.csv", sampled)
    write_provenance(
        output_dir / "provenance.json",
        sampled,
        source_rows=len(records),
        skipped_duplicates=skipped_duplicates,
        per_cell=per_cell,
        seed=seed,
    )
    return len(sampled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собрать сырую выборку отзывов")
    parser.add_argument("--zip", type=Path, required=True, dest="zip_path")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--per-cell", type=int, default=DEFAULT_PER_CELL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = collect(args.zip_path, args.output_dir, args.per_cell, args.seed)
    print(f"Сохранено отзывов: {count}")


if __name__ == "__main__":
    main()
