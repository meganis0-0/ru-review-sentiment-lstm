"""Baseline до LSTM: частотный класс и логистическая регрессия по мешку слов."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from src.data.prepare import PAD_ID, SEED
from src.data.validate import VALID_LABELS

DEFAULT_TRAIN = Path("data/processed/train.csv")
DEFAULT_VAL = Path("data/processed/val.csv")
DEFAULT_TEST = Path("data/processed/test.csv")
DEFAULT_VOCAB = Path("data/processed/vocab.json")
DEFAULT_OUTPUT = Path("reports/baseline.json")


def load_split(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def vocab_size(path: Path) -> int:
    vocab = json.loads(path.read_text(encoding="utf-8"))
    return len(vocab)


def majority_label(labels: list[str]) -> str:
    """Самая частая метка train. При равенстве берётся более ранний класс."""
    counts = Counter(labels)
    return max(VALID_LABELS, key=lambda label: counts[label])


def bag_of_words(rows: list[dict[str, str]], size: int) -> csr_matrix:
    """Считает токены отзыва. Номер pad в признаки не входит."""
    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    for row in rows:
        counts: Counter[int] = Counter(
            token_id
            for token_id in (int(part) for part in row["token_ids"].split())
            if token_id != PAD_ID
        )
        for token_id, count in counts.items():
            indices.append(token_id)
            data.append(count)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), shape=(len(rows), size))


def fit_logistic(rows: list[dict[str, str]], size: int, seed: int = SEED) -> LogisticRegression:
    model = LogisticRegression(solver="saga", max_iter=4000, random_state=seed)
    model.fit(bag_of_words(rows, size), [row["label"] for row in rows])
    return model


def classification_scores(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    per_class = {label: _f1(y_true, y_pred, label) for label in VALID_LABELS}
    correct = sum(true == pred for true, pred in zip(y_true, y_pred, strict=True))
    return {
        "accuracy": correct / len(y_true),
        "f1_macro": sum(per_class.values()) / len(VALID_LABELS),
        "f1": per_class,
    }


def evaluate(
    train: list[dict[str, str]],
    val: list[dict[str, str]],
    test: list[dict[str, str]],
    size: int,
    seed: int = SEED,
) -> dict[str, object]:
    """Учит baseline только на train и считает метрики на val и test."""
    label = majority_label([row["label"] for row in train])
    model = fit_logistic(train, size, seed)
    report: dict[str, object] = {
        "seed": seed,
        "vocab_size": size,
        "majority_label": label,
    }
    for name, rows in (("val", val), ("test", test)):
        truth = [row["label"] for row in rows]
        majority_pred = [label] * len(rows)
        logistic_pred = [str(item) for item in model.predict(bag_of_words(rows, size))]
        report[name] = {
            "majority": classification_scores(truth, majority_pred),
            "logistic_regression": classification_scores(truth, logistic_pred),
        }
    return report


def _f1(y_true: list[str], y_pred: list[str], label: str) -> float:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for true, pred in zip(y_true, y_pred, strict=True):
        if pred == label and true == label:
            true_positive += 1
        elif pred == label:
            false_positive += 1
        elif true == label:
            false_negative += 1
    precision_denom = true_positive + false_positive
    recall_denom = true_positive + false_negative
    precision = true_positive / precision_denom if precision_denom else 0.0
    recall = true_positive / recall_denom if recall_denom else 0.0
    denom = precision + recall
    if not denom:
        return 0.0
    return 2 * precision * recall / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Посчитать baseline до LSTM")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        load_split(args.train),
        load_split(args.val),
        load_split(args.test),
        vocab_size(args.vocab),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("val", "test"):
        part = report[name]
        assert isinstance(part, dict)
        majority = part["majority"]
        logistic = part["logistic_regression"]
        assert isinstance(majority, dict)
        assert isinstance(logistic, dict)
        print(
            f"{name}: частотный класс F1-macro {majority['f1_macro']:.3f}, "
            f"логистическая регрессия F1-macro {logistic['f1_macro']:.3f}"
        )


if __name__ == "__main__":
    main()
