import json
from pathlib import Path

from sklearn.metrics import f1_score

from src.baseline import bag_of_words, classification_scores, evaluate, majority_label
from src.data.validate import VALID_LABELS


def _row(label: str, token_ids: str) -> dict[str, str]:
    return {"label": label, "token_ids": token_ids}


def _separable() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = []
    test = []
    for index, label in enumerate(VALID_LABELS, start=2):
        token_ids = f"{index} 0 0"
        train.extend(_row(label, token_ids) for _copy in range(8))
        test.extend(_row(label, token_ids) for _copy in range(2))
    return train, test


def test_majority_label_breaks_ties_by_class_order() -> None:
    labels = ["positive", "negative", "neutral", "positive"]

    assert majority_label(labels) == "positive"
    assert majority_label(["neutral", "negative"]) == "negative"


def test_bag_of_words_ignores_padding() -> None:
    rows = [_row("neutral", "2 2 0 0"), _row("neutral", "2 2")]
    matrix = bag_of_words(rows, size=4)

    assert matrix.shape == (2, 4)
    assert matrix[0].toarray().tolist() == matrix[1].toarray().tolist()
    assert matrix[0, 0] == 0
    assert matrix[0, 2] == 2


def test_f1_macro_matches_sklearn() -> None:
    y_true = ["negative", "neutral", "positive", "negative"]
    y_pred = ["negative", "negative", "positive", "neutral"]
    expected = f1_score(y_true, y_pred, labels=list(VALID_LABELS), average="macro", zero_division=0)

    assert classification_scores(y_true, y_pred)["f1_macro"] == expected


def test_logistic_regression_beats_majority_on_separable_tokens() -> None:
    train, test = _separable()
    report = evaluate(train, test, test, size=5, seed=42)
    majority = report["test"]["majority"]["f1_macro"]
    logistic = report["test"]["logistic_regression"]["f1_macro"]

    assert logistic == 1
    assert majority < logistic


def test_readme_records_saved_baseline() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/baseline.json").read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    logistic = report["test"]["logistic_regression"]["f1_macro"]
    majority = report["test"]["majority"]["f1_macro"]

    assert f"{logistic:.3f}" in readme
    assert f"{majority:.3f}" in readme
