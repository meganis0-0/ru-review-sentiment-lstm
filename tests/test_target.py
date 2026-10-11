import json
from pathlib import Path

import pytest

from src.target import summarize_run


def test_summarize_run_keeps_the_best_epoch() -> None:
    summary = summarize_run(
        "lstm",
        "LSTM",
        {
            "stopped_early": True,
            "history": [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "val_loss": 1.0,
                    "f1_macro": 0.4,
                    "accuracy": 0.4,
                    "confusion_matrix": [[1]],
                },
                {
                    "epoch": 2,
                    "train_loss": 0.2,
                    "val_loss": 1.4,
                    "f1_macro": 0.55,
                    "accuracy": 0.5,
                    "confusion_matrix": [[2]],
                },
            ],
        },
    )

    assert summary["best_epoch"] == 2
    assert summary["best_f1_macro"] == 0.55
    assert summary["last_train_loss"] == 0.2
    assert summary["confusion_matrix"] == [[2]]
    assert len(summary["history"]) == 2


def test_summarize_run_rejects_an_empty_history() -> None:
    with pytest.raises(ValueError, match="lstm"):
        summarize_run("lstm", "LSTM", {"history": [], "stopped_early": False})


def test_target_report_matches_saved_numbers() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "reports" / "target_metric.json").read_text(encoding="utf-8"))
    text = (root / "reports" / "target_metric.md").read_text(encoding="utf-8")
    best = float(payload["best"]["f1_macro"])
    baseline = float(payload["baseline"]["logistic_regression_f1_macro"])

    assert payload["target_reached"] is False
    assert payload["best"]["name"] == "combo"
    assert payload["extra_data"]["added"] is False
    assert f"{best:.3f}" in text
    assert f"{baseline:.3f}" in text
    assert "0.80" in text
