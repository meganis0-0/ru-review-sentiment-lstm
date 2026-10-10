"""Пилотная разметка и перевод выгрузки Label Studio в CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from src.data.collect import SOURCE_NOTES

CHOICE_TO_LABEL = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}
PILOT_PER_CELL = 2
PILOT_EXTRA = 2
PILOT_SEED = 42


def read_reviews(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows, start=1):
        if not row.get("review_id"):
            row["review_id"] = str(index)
    return rows


def write_labels(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["review_id", "text", "label", "source", "label_origin"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "review_id": row["review_id"],
                    "text": row["text"],
                    "label": row["label"],
                    "source": row["source"],
                    "label_origin": row.get("label_origin", "source"),
                }
            )


def select_pilot(
    rows: list[dict[str, str]],
    per_cell: int = PILOT_PER_CELL,
    extra: int = PILOT_EXTRA,
    seed: int = PILOT_SEED,
) -> list[dict[str, str]]:
    """Берёт отзывы из каждой пары сфера-класс, затем добавляет ещё несколько."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["label"])].append(row)

    rng = random.Random(seed)
    chosen: list[dict[str, str]] = []
    leftovers: list[dict[str, str]] = []
    for key in sorted(groups):
        bucket = list(groups[key])
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_cell])
        leftovers.extend(bucket[per_cell:])
    rng.shuffle(leftovers)
    chosen.extend(leftovers[:extra])
    return chosen


def pilot_tasks(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    tasks = []
    for row in rows:
        tasks.append(
            {
                "data": {
                    "review_id": row["review_id"],
                    "text": row["text"],
                    "source": row["source"],
                    "source_name": SOURCE_NOTES.get(row["source"], row["source"]),
                }
            }
        )
    return tasks


def write_pilot(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(pilot_tasks(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def annotation_label(task: dict[str, object]) -> str | None:
    annotations = task.get("annotations") or []
    if not annotations or not isinstance(annotations, list):
        return None
    result = annotations[-1].get("result") or []
    for item in result:
        if item.get("from_name") != "sentiment":
            continue
        choices = item.get("value", {}).get("choices") or []
        if not choices:
            continue
        label = CHOICE_TO_LABEL.get(str(choices[0]))
        if label is not None:
            return label
    return None


def apply_pilot_export(
    labels: list[dict[str, str]],
    tasks: list[dict[str, object]],
) -> dict[str, int]:
    """Подменяет метки пилота ручной разметкой и считает совпадения."""
    by_id = {row["review_id"]: row for row in labels}
    compared = 0
    matched = 0
    updated = 0
    for task in tasks:
        data = task.get("data") or {}
        review_id = str(data.get("review_id", ""))
        manual = annotation_label(task)
        row = by_id.get(review_id)
        if row is None or manual is None:
            continue
        compared += 1
        matched += int(row["label"] == manual)
        if row["label"] != manual or row.get("label_origin") != "manual":
            row["label"] = manual
            row["label_origin"] = "manual"
            updated += 1
    return {"compared": compared, "matched": matched, "updated": updated}


def prepare(reviews_path: Path, interim_dir: Path) -> tuple[int, int]:
    rows = read_reviews(reviews_path)
    for row in rows:
        row["label_origin"] = "source"
    write_labels(interim_dir / "labels.csv", rows)
    pilot = select_pilot(rows)
    write_pilot(interim_dir / "pilot_tasks.json", pilot)
    return len(rows), len(pilot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подготовить разметку и пилот")
    parser.add_argument("--reviews", type=Path, default=Path("data/raw/reviews.csv"))
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--export", type=Path, help="JSON-выгрузка Label Studio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export is None:
        total, pilot = prepare(args.reviews, args.interim)
        print(f"Меток: {total}. Пилот: {pilot}")
        return

    labels = read_reviews(args.interim / "labels.csv")
    for row in labels:
        row["label_origin"] = row.get("label_origin") or "source"
    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    stats = apply_pilot_export(labels, tasks)
    write_labels(args.interim / "labels.csv", labels)
    print(
        "Сверено: {compared}. Совпало с исходной меткой: {matched}. Обновлено: {updated}".format(
            **stats
        )
    )


if __name__ == "__main__":
    main()
