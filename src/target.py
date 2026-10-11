"""Прогоны к целевой F1-macro: веса, аугментация, Navec и другая голова LSTM."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.data.dataset import make_loader
from src.models.config import ModelConfig
from src.models.embeddings import vectors_from_navec
from src.train import (
    build_model,
    class_weight_tensor,
    load_train_config,
    run_training,
    set_seed,
)
from src.utils import get_logger

NAVEC_PATH = "data/external/navec_hudlit_v1_12B_500K_300d_100q.tar"
REPORT_PATH = "reports/target_metric.json"


@dataclass(frozen=True)
class Experiment:
    name: str
    title: str
    model: ModelConfig


EXPERIMENTS = (
    Experiment("lstm", "LSTM, последнее состояние", ModelConfig()),
    Experiment("class-weights", "LSTM и веса классов", ModelConfig(class_weights=True)),
    Experiment("token-dropout", "LSTM и выкидывание слов", ModelConfig(token_dropout=0.1)),
    Experiment(
        "bilstm-mean",
        "BiLSTM, среднее по словам",
        ModelConfig(bidirectional=True, pooling="mean"),
    ),
    Experiment(
        "pretrained",
        "LSTM и замороженный Navec",
        ModelConfig(
            embedding_dim=300,
            pretrained_embeddings=NAVEC_PATH,
            freeze_embeddings=True,
        ),
    ),
    Experiment(
        "combo",
        "BiLSTM, Navec и выкидывание слов",
        ModelConfig(
            embedding_dim=300,
            bidirectional=True,
            pooling="mean",
            pretrained_embeddings=NAVEC_PATH,
            freeze_embeddings=True,
            token_dropout=0.1,
        ),
    ),
)


def summarize_run(name: str, title: str, result: dict[str, object]) -> dict[str, object]:
    """Оставляет лучшую эпоху и короткую кривую, без оптимизатора и весов."""
    history = result["history"]
    if not isinstance(history, list) or not history:
        raise ValueError(f"У прогона {name} пустая история")
    best = max(history, key=lambda row: float(row["f1_macro"]))
    last = history[-1]
    return {
        "name": name,
        "title": title,
        "best_epoch": best["epoch"],
        "best_f1_macro": best["f1_macro"],
        "best_accuracy": best["accuracy"],
        "best_train_loss": best["train_loss"],
        "best_val_loss": best["val_loss"],
        "last_epoch": last["epoch"],
        "last_train_loss": last["train_loss"],
        "last_val_loss": last["val_loss"],
        "stopped_early": result["stopped_early"],
        "confusion_matrix": best["confusion_matrix"],
        "history": [
            {
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "val_loss": row["val_loss"],
                "f1_macro": row["f1_macro"],
            }
            for row in history
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    logger = get_logger("target")
    train_config = load_train_config(root / "configs" / "train.yaml")
    raw = yaml.safe_load((root / "configs" / "train.yaml").read_text(encoding="utf-8"))
    target_metric = float(raw["target_metric"])
    baseline = json.loads((root / "reports" / "baseline.json").read_text(encoding="utf-8"))
    pilot = json.loads(
        (root / "data" / "interim" / "pilot_agreement.json").read_text(encoding="utf-8")
    )
    train_path = root / "data" / "processed" / "train.csv"
    val_path = root / "data" / "processed" / "val.csv"
    vocab_path = root / "data" / "processed" / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    labels = _labels(train_path)
    weights = class_weight_tensor(labels)
    navec_file = root / NAVEC_PATH
    vectors = vectors_from_navec(navec_file, vocab) if navec_file.is_file() else None
    runs: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        if experiment.model.pretrained_embeddings and vectors is None:
            logger.info("Пропуск %s: нет архива Navec", experiment.name)
            runs.append(
                {
                    "name": experiment.name,
                    "title": experiment.title,
                    "skipped": True,
                    "reason": f"Нет файла {NAVEC_PATH}",
                }
            )
            continue
        logger.info("Старт %s: %s", experiment.name, experiment.title)
        used_vectors = vectors if experiment.model.pretrained_embeddings else None
        set_seed(train_config.seed)
        model, model_args = build_model(vocab, experiment.model, used_vectors)
        train_loader = make_loader(
            train_path,
            batch_size=train_config.batch_size,
            shuffle=True,
            seed=train_config.seed,
            token_dropout=experiment.model.token_dropout,
        )
        val_loader = make_loader(
            val_path,
            batch_size=train_config.batch_size,
            shuffle=False,
            seed=train_config.seed,
        )
        result = run_training(
            train_loader,
            val_loader,
            model,
            epochs=train_config.epochs,
            learning_rate=train_config.learning_rate,
            seed=train_config.seed,
            checkpoint_dir=root / "checkpoints" / "experiments" / experiment.name,
            log_dir=root / "logs" / "tensorboard" / f"target-{experiment.name}",
            early_stopping_patience=train_config.early_stopping_patience,
            early_stopping_min_delta=train_config.early_stopping_min_delta,
            class_weights=weights if experiment.model.class_weights else None,
            model_args=model_args,
        )
        summary = summarize_run(experiment.name, experiment.title, result)
        runs.append(summary)
        logger.info(
            "%s: лучший f1-macro %.3f на эпохе %s",
            experiment.name,
            summary["best_f1_macro"],
            summary["best_epoch"],
        )
    finished = [run for run in runs if "best_f1_macro" in run]
    best = max(finished, key=lambda run: float(run["best_f1_macro"]))
    _keep_best_checkpoint(root, best)
    coverage = None if vectors is None else {"hits": len(vectors), "vocab": len(vocab) - 2}
    payload = {
        "split": "val",
        "target_metric": target_metric,
        "target_reached": float(best["best_f1_macro"]) >= target_metric,
        "baseline": {
            "majority_f1_macro": baseline["val"]["majority"]["f1_macro"],
            "logistic_regression_f1_macro": baseline["val"]["logistic_regression"]["f1_macro"],
            "logistic_regression_test_f1_macro": baseline["test"]["logistic_regression"][
                "f1_macro"
            ],
        },
        "class_weights": [float(item) for item in weights.tolist()],
        "pretrained": {
            "pack": "navec_hudlit_v1_12B_500K_300d_100q",
            "frozen": True,
            "coverage": coverage,
            "available": vectors is not None,
        },
        "extra_data": {
            "added": False,
            "reason": (
                "В data/raw только отобранные 4788 отзывов. "
                "Полного архива Kaggle в проекте нет, val и test не расширялись."
            ),
        },
        "pilot_agreement": pilot["agreement"],
        "pilot_matched": pilot["matched"],
        "pilot_size": pilot["pilot_size"],
        "best": {"name": best["name"], "f1_macro": best["best_f1_macro"]},
        "experiments": runs,
    }
    report = root / REPORT_PATH
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Отчёт: %s. Лучший %s %.3f", report, best["name"], best["best_f1_macro"])


def _keep_best_checkpoint(root: Path, best: dict[str, object]) -> None:
    source = root / "checkpoints" / "experiments" / str(best["name"]) / "best.pt"
    target = root / "checkpoints" / "best.pt"
    if source.is_file():
        shutil.copyfile(source, target)


def _labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["label"] for row in csv.DictReader(handle)]


if __name__ == "__main__":
    main()
