"""Учит ту же BiLSTM на разметке по правилу. Чекпоинт корпуса не затирает."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from src.data.dataset import make_loader
from src.models.config import load_model_config
from src.models.embeddings import vectors_from_navec
from src.train import (
    build_model,
    class_weight_tensor,
    load_train_config,
    prepare_runtime,
    run_training,
    set_seed,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["label"] for row in csv.DictReader(handle)]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_train_config(prepare_runtime(root))
    model_config = replace(load_model_config(root / "configs" / "model.yaml"), class_weights=True)
    data_dir = root / "data" / "rule"
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    vocab_path = data_dir / "vocab.json"
    if not train_path.is_file() or not val_path.is_file() or not vocab_path.is_file():
        raise SystemExit("Нет data/rule. Сначала запустите python -m src.data.rule_label")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vectors = None
    if model_config.pretrained_embeddings:
        vectors = vectors_from_navec(root / model_config.pretrained_embeddings, vocab)
    set_seed(config.seed)
    model, model_args = build_model(vocab, model_config, vectors)
    result = run_training(
        make_loader(
            train_path,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
            token_dropout=model_config.token_dropout,
        ),
        make_loader(val_path, batch_size=config.batch_size, shuffle=False),
        model,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
        checkpoint_dir=root / "checkpoints" / "rule",
        log_dir=root / config.log_dir / "rule-labels",
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        class_weights=class_weight_tensor(_labels(train_path)),
        model_args=model_args,
    )
    report = {
        "labeling": "rule",
        "sources": ["bank", "geo", "kinopoisk"],
        "class_weights": True,
        "best_f1_macro": result["best_f1_macro"],
        "stopped_early": result["stopped_early"],
        "history": result["history"],
        "checkpoint": "checkpoints/rule/best.pt",
    }
    path = root / "reports" / "rule_metric.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Val F1-macro %.3f. Отчёт: %s", result["best_f1_macro"], path)


if __name__ == "__main__":
    main()
