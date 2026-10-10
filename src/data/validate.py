"""Проверка выборки отзывов: битые строки, дубли, баланс и длина.

Одинаковыми считаются тексты, которые совпадают после обрезки пробелов
и приведения перевода строки к одному виду.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.data.collect import REVIEW_SOURCES, SOURCE_NOTES

VALID_LABELS = ("negative", "neutral", "positive")
LABEL_TITLES = {
    "negative": "негатив",
    "neutral": "нейтрально",
    "positive": "позитив",
}
SHORT_TEXT = 10
IMBALANCE_RATIO = 1.5
EXAMPLE_LIMIT = 8


def validate_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    """Сводит проблемы таблицы отзывов в один отчёт."""
    broken_examples: list[dict[str, str]] = []
    empty_text = 0
    invalid_label = 0
    missing_source = 0
    seen_ids: dict[str, int] = {}
    duplicate_ids = 0
    text_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: list[int] = []
    short_examples: list[dict[str, object]] = []

    for row in rows:
        reasons = _row_reasons(row)
        review_id = row.get("review_id", "").strip()
        if review_id:
            if review_id in seen_ids:
                duplicate_ids += 1
                reasons.append("повтор id")
            else:
                seen_ids[review_id] = 1
        if "пустой текст" in reasons:
            empty_text += 1
        if "чужая метка" in reasons:
            invalid_label += 1
        if "нет сферы" in reasons:
            missing_source += 1
        if reasons and len(broken_examples) < EXAMPLE_LIMIT:
            broken_examples.append({"review_id": review_id, "reasons": ", ".join(reasons)})

        label = row.get("label", "").strip()
        source = row.get("source", "").strip()
        if label in VALID_LABELS:
            class_counts[label] += 1
            if source:
                by_source[source][label] += 1

        text = _duplicate_key(row.get("text", ""))
        if not text:
            continue
        text_groups[text].append(row)
        length = len(text)
        lengths.append(length)
        if length < SHORT_TEXT:
            short_examples.append(
                {
                    "review_id": review_id,
                    "text": text,
                    "label": label,
                    "source": source,
                    "length": length,
                }
            )

    length_stats = _length_stats(lengths)
    duplicates = _duplicate_report(text_groups)
    counts = {label: class_counts[label] for label in VALID_LABELS}
    ratio = _imbalance_ratio(counts)
    ordered_sources = [source for source in REVIEW_SOURCES if source in by_source]
    ordered_sources.extend(sorted(set(by_source) - set(ordered_sources)))
    report: dict[str, object] = {
        "rows": len(rows),
        "broken": {
            "empty_text": empty_text,
            "invalid_label": invalid_label,
            "missing_source": missing_source,
            "duplicate_ids": duplicate_ids,
            "examples": broken_examples,
        },
        "duplicates": duplicates,
        "classes": counts,
        "imbalance_ratio": ratio,
        "imbalanced": ratio is None or ratio > IMBALANCE_RATIO,
        "by_source": {
            source: {label: by_source[source][label] for label in VALID_LABELS}
            for source in ordered_sources
        },
        "length": length_stats,
        "short_examples": _shortest(short_examples),
        "label_origin": dict(
            Counter(row.get("label_origin", "") for row in rows if row.get("label_origin"))
        ),
    }
    report["problems"] = _problems(report)
    return report


def load_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = {"text", "label", "source"} - fieldnames
        if missing:
            raise ValueError(f"В {path} нет колонок: {', '.join(sorted(missing))}")
        return list(reader)


def format_report(report: dict[str, object]) -> str:
    classes = report["classes"]
    assert isinstance(classes, dict)
    length = report["length"]
    assert isinstance(length, dict)
    duplicates = report["duplicates"]
    assert isinstance(duplicates, dict)
    lines = [
        f"Строк: {report['rows']}",
        "Классы: " + ", ".join(f"{LABEL_TITLES[label]} {classes[label]}" for label in VALID_LABELS),
        (
            f"Длина, символы: min {length['min']}, медиана {length['median']}, "
            f"max {length['max']}"
        ),
        (
            f"Коротких (<{SHORT_TEXT}): {length['short']}. "
            f"Выбросы длины: снизу {length['low_outliers']}, сверху {length['high_outliers']}"
        ),
        (
            f"Групп одинакового текста: {duplicates['groups']}, "
            f"лишних строк: {duplicates['extra_rows']}"
        ),
        "Проблемы:",
    ]
    problems = report["problems"]
    assert isinstance(problems, list)
    lines.extend(f"- {problem}" for problem in problems)
    return "\n".join(lines)


def _shortest(examples: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(examples, key=lambda item: (int(item["length"]), str(item["review_id"])))
    return ordered[:EXAMPLE_LIMIT]


def normalized_text(text: str) -> str:
    """Один вид перевода строки и без пробелов по краям."""
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def _duplicate_key(text: str) -> str:
    return normalized_text(text)


def _row_reasons(row: dict[str, str]) -> list[str]:
    reasons = []
    if not row.get("text", "").strip():
        reasons.append("пустой текст")
    if row.get("label", "").strip() not in VALID_LABELS:
        reasons.append("чужая метка")
    if not row.get("source", "").strip():
        reasons.append("нет сферы")
    return reasons


def _imbalance_ratio(counts: dict[str, int]) -> float | None:
    low = min(counts.values())
    if low == 0:
        return None
    return max(counts.values()) / low


def _length_stats(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {
            "min": 0,
            "q1": 0,
            "median": 0,
            "q3": 0,
            "max": 0,
            "low_fence": 0,
            "high_fence": 0,
            "short": 0,
            "low_outliers": 0,
            "high_outliers": 0,
        }
    ordered = sorted(lengths)
    if len(ordered) < 4:
        q1 = median = q3 = float(statistics.median(ordered))
    else:
        q1, median, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    return {
        "min": ordered[0],
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": ordered[-1],
        "low_fence": low_fence,
        "high_fence": high_fence,
        "short": sum(length < SHORT_TEXT for length in ordered),
        "low_outliers": sum(length < low_fence for length in ordered),
        "high_outliers": sum(length > high_fence for length in ordered),
    }


def _duplicate_report(text_groups: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    examples = []
    extra_rows = 0
    groups = 0
    for text, grouped in text_groups.items():
        if len(grouped) < 2:
            continue
        groups += 1
        extra_rows += len(grouped) - 1
        if len(examples) >= EXAMPLE_LIMIT:
            continue
        examples.append(
            {
                "text": text[:120],
                "review_ids": [row.get("review_id", "") for row in grouped],
                "sources": [row.get("source", "") for row in grouped],
                "labels": [row.get("label", "") for row in grouped],
            }
        )
    return {"groups": groups, "extra_rows": extra_rows, "examples": examples}


def _problems(report: dict[str, object]) -> list[str]:
    problems: list[str] = []
    if report["rows"] == 0:
        return ["Выборка пустая."]

    broken = report["broken"]
    assert isinstance(broken, dict)
    broken_total = sum(
        (
            broken["empty_text"],
            broken["invalid_label"],
            broken["missing_source"],
            broken["duplicate_ids"],
        )
    )
    if broken_total:
        problems.append(
            "Битые строки: "
            f"пустой текст {broken['empty_text']}, чужая метка {broken['invalid_label']}, "
            f"нет сферы {broken['missing_source']}, повтор id {broken['duplicate_ids']}."
        )

    duplicates = report["duplicates"]
    assert isinstance(duplicates, dict)
    if duplicates["groups"]:
        problems.append(
            "Одинаковый текст в разных строках: "
            f"{duplicates['groups']} групп, лишних строк {duplicates['extra_rows']}."
        )

    classes = report["classes"]
    assert isinstance(classes, dict)
    counts_text = ", ".join(f"{LABEL_TITLES[label]} {classes[label]}" for label in VALID_LABELS)
    ratio = report["imbalance_ratio"]
    if report["imbalanced"]:
        ratio_text = "пустой класс" if ratio is None else f"отношение {ratio:.2f}"
        problems.append(f"Дисбаланс классов ({counts_text}), {ratio_text}.")
    else:
        assert isinstance(ratio, float)
        problems.append(f"Классы близки ({counts_text}), отношение {ratio:.2f}.")

    length = report["length"]
    assert isinstance(length, dict)
    if length["short"]:
        problems.append(
            f"Слишком коротких текстов (меньше {SHORT_TEXT} символов): {length['short']}."
        )
    if length["low_outliers"] or length["high_outliers"]:
        problems.append(
            "Выбросы длины по правилу 1.5 IQR: "
            f"снизу {length['low_outliers']}, сверху {length['high_outliers']}."
        )
    if not problems:
        problems.append("Пустых строк, чужих меток и перекоса классов нет.")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверить выборку отзывов")
    parser.add_argument("--labels", type=Path, default=Path("data/interim/labels.csv"))
    return parser.parse_args()


def main() -> None:
    report = validate_rows(load_labels(parse_args().labels))
    print(format_report(report))
    by_source = report["by_source"]
    assert isinstance(by_source, dict)
    for source, counts in by_source.items():
        assert isinstance(counts, dict)
        titled = ", ".join(f"{LABEL_TITLES[label]} {counts[label]}" for label in VALID_LABELS)
        print(f"{SOURCE_NOTES.get(str(source), str(source))}: {titled}")


if __name__ == "__main__":
    main()
