"""Оценка чекпоинта на отложенной части: отчёт, матрица и разбор ошибок."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch import nn

from src.data.collect import SOURCE_NOTES
from src.data.dataset import LABEL_TO_INDEX, make_loader
from src.data.prepare import MAX_LEN
from src.data.validate import LABEL_TITLES, VALID_LABELS
from src.train import METRIC_LABELS, compute_metrics

INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
EXAMPLE_LIMIT = 2
TEXT_LIMIT = 180


def predict_file(model: nn.Module, path: Path, batch_size: int = 32) -> list[dict[str, object]]:
    """Предсказывает классы в порядке строк файла. Dropout на оценке выключен."""
    rows = _read_rows(path)
    loader = make_loader(path, batch_size=batch_size, shuffle=False)
    predicted: list[int] = []
    confidence: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            probabilities = torch.softmax(model(batch["token_ids"], batch["lengths"]), dim=-1)
            chosen = probabilities.argmax(dim=-1)
            predicted.extend(int(item) for item in chosen.tolist())
            confidence.extend(float(item) for item in probabilities.max(dim=-1).values.tolist())
    if len(predicted) != len(rows):
        raise ValueError(f"Предсказаний {len(predicted)}, строк {len(rows)}")
    records: list[dict[str, object]] = []
    for row, pred, score in zip(rows, predicted, confidence, strict=True):
        label = row["label"].strip()
        records.append(
            {
                "review_id": row["review_id"],
                "text": row["text"],
                "label": label,
                "predicted": INDEX_TO_LABEL[pred],
                "source": row["source"],
                "label_origin": row.get("label_origin", ""),
                "token_count": int(row["token_count"]),
                "confidence": score,
                "correct": label == INDEX_TO_LABEL[pred],
            }
        )
    return records


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    """Сводит accuracy, отчёт по классам, сферы и типы ошибок."""
    if not records:
        raise ValueError("Нет строк для оценки")
    y_true = [LABEL_TO_INDEX[str(row["label"])] for row in records]
    y_pred = [LABEL_TO_INDEX[str(row["predicted"])] for row in records]
    report = classification_report(
        y_true,
        y_pred,
        labels=METRIC_LABELS,
        target_names=list(VALID_LABELS),
        output_dict=True,
        zero_division=0,
    )
    mistakes = [row for row in records if not row["correct"]]
    adjacent = 0
    polarity = 0
    for row in mistakes:
        if _gap(str(row["label"]), str(row["predicted"])) == 1:
            adjacent += 1
        else:
            polarity += 1
    return {
        "rows": len(records),
        "errors": len(mistakes),
        "adjacent_errors": adjacent,
        "polarity_errors": polarity,
        "metrics": compute_metrics(y_true, y_pred),
        "report": report,
        "by_source": _by_source(records),
        "truncated": _slice(records, lambda row: int(row["token_count"]) > MAX_LEN),
        "short": _slice(records, lambda row: len(str(row["text"]).strip()) < 10),
        "manual": _slice(records, lambda row: row["label_origin"] == "manual"),
        "examples": _examples(mistakes),
        "confident_hits": _confident_hits(records),
    }


def format_summary(
    summary: dict[str, object],
    *,
    baseline_f1: float | None = None,
    val_f1: float | None = None,
) -> str:
    """Текст для ноутбука: метрики, слабый класс и где модель путает соседние метки."""
    metrics = summary["metrics"]
    lines = [
        f"Строк: {summary['rows']}. Ошибок: {summary['errors']}.",
        (
            f"Accuracy {float(metrics['accuracy']):.3f}, "
            f"precision {float(metrics['precision']):.3f}, "
            f"recall {float(metrics['recall']):.3f}, "
            f"F1-macro {float(metrics['f1_macro']):.3f}."
        ),
    ]
    if baseline_f1 is not None:
        lines.append(f"Baseline на test, логистическая регрессия: F1-macro {baseline_f1:.3f}.")
    if val_f1 is not None:
        lines.append(f"F1-macro на val у этого чекпоинта: {val_f1:.3f}.")
    lines.append("По классам:")
    report = summary["report"]
    for label in VALID_LABELS:
        cell = report[label]
        lines.append(
            f"- {LABEL_TITLES[label]}: precision {cell['precision']:.3f}, "
            f"recall {cell['recall']:.3f}, f1 {cell['f1-score']:.3f}, "
            f"support {int(cell['support'])}"
        )
    lines.append(f"Матрица ошибок: {metrics['confusion_matrix']}")
    lines.append(
        "Соседний класс: "
        f"{summary['adjacent_errors']}. Перескок негатив↔позитив: {summary['polarity_errors']}."
    )
    strongest, weakest = _class_ends(report)
    lines.append(f"Сильнее всего {LABEL_TITLES[strongest]}. Слабее всего {LABEL_TITLES[weakest]}.")
    lines.append("По сферам:")
    for item in summary["by_source"]:
        mistake = item["top_mistake"]
        extra = f", чаще всего {mistake}" if mistake else ""
        lines.append(
            f"- {item['title']}: accuracy {item['accuracy']:.3f}, "
            f"ошибок {item['errors']} из {item['support']}{extra}"
        )
    lines.append(_slice_line("Длиннее окна в 200 токенов", summary["truncated"]))
    lines.append(_slice_line("Короче 10 символов", summary["short"]))
    lines.append(_slice_line("Ручная метка пилота", summary["manual"]))
    return "\n".join(lines)


def format_examples(summary: dict[str, object]) -> str:
    """Уверенные ошибки и по одному уверенному попаданию на класс."""
    lines = ["Ошибки, в которых модель уверена:"]
    examples = summary["examples"]
    if not examples:
        lines.append("- нет")
    for row in examples:
        lines.append(_example_line(row))
    lines.append("Уверенные верные ответы:")
    for row in summary["confident_hits"]:
        lines.append(_example_line(row))
    return "\n".join(lines)


def _example_line(row: dict[str, object]) -> str:
    true_title = LABEL_TITLES[str(row["label"])]
    pred_title = LABEL_TITLES[str(row["predicted"])]
    return (
        f"- [{true_title} → {pred_title}] {float(row['confidence']):.2f}, "
        f"{row['source']}, {row['token_count']} токенов\n  «{_clip(str(row['text']))}»"
    )


def _class_ends(report: dict) -> tuple[str, str]:
    scores = {label: float(report[label]["f1-score"]) for label in VALID_LABELS}
    strongest = max(VALID_LABELS, key=lambda label: (scores[label], -VALID_LABELS.index(label)))
    weakest = min(VALID_LABELS, key=lambda label: (scores[label], VALID_LABELS.index(label)))
    return strongest, weakest


def _gap(true_label: str, predicted: str) -> int:
    return abs(LABEL_TO_INDEX[true_label] - LABEL_TO_INDEX[predicted])


def _by_source(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source"])].append(row)
    items = []
    for source, rows in grouped.items():
        errors = [row for row in rows if not row["correct"]]
        pairs = Counter((str(row["label"]), str(row["predicted"])) for row in errors)
        top = None
        if pairs:
            (true_label, predicted), count = pairs.most_common(1)[0]
            top = f"{LABEL_TITLES[true_label]} → {LABEL_TITLES[predicted]} ({count})"
        items.append(
            {
                "source": source,
                "title": _source_title(source),
                "support": len(rows),
                "errors": len(errors),
                "accuracy": (len(rows) - len(errors)) / len(rows),
                "top_mistake": top,
            }
        )
    return sorted(items, key=lambda item: (float(item["accuracy"]), str(item["source"])))


def _source_title(source: str) -> str:
    return SOURCE_NOTES.get(source, source)


def _slice(records: list[dict[str, object]], predicate) -> dict[str, object]:
    chosen = [row for row in records if predicate(row)]
    if not chosen:
        return {"support": 0, "errors": 0, "accuracy": None}
    errors = sum(1 for row in chosen if not row["correct"])
    return {
        "support": len(chosen),
        "errors": errors,
        "accuracy": (len(chosen) - errors) / len(chosen),
    }


def _slice_line(title: str, payload: dict[str, object]) -> str:
    support = int(payload["support"])
    if support == 0 or payload["accuracy"] is None:
        return f"{title}: 0."
    return f"{title}: accuracy {float(payload['accuracy']):.3f} на {support}."


def _examples(mistakes: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in mistakes:
        grouped[(str(row["label"]), str(row["predicted"]))].append(row)
    chosen: list[dict[str, object]] = []
    for pair in sorted(grouped, key=lambda item: (-len(grouped[item]), item)):
        ordered = sorted(grouped[pair], key=lambda row: float(row["confidence"]), reverse=True)
        chosen.extend(ordered[:EXAMPLE_LIMIT])
    return chosen


def _confident_hits(records: list[dict[str, object]]) -> list[dict[str, object]]:
    hits = []
    for label in VALID_LABELS:
        same = [row for row in records if row["correct"] and row["label"] == label]
        if not same:
            continue
        hits.append(max(same, key=lambda row: float(row["confidence"])))
    return hits


def _clip(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= TEXT_LIMIT:
        return compact
    return compact[: TEXT_LIMIT - 1] + "…"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
