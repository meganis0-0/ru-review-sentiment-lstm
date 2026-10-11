"""Цикл обучения LSTM: эпоха, метрики, чекпоинт и TensorBoard."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import make_loader
from src.data.validate import VALID_LABELS
from src.models.config import ModelConfig, load_model_config
from src.models.embeddings import build_embedding_matrix, vectors_from_navec
from src.models.rnn import NUM_CLASSES, SentimentLSTM
from src.utils import get_logger

METRIC_LABELS = list(range(NUM_CLASSES))


@dataclass(frozen=True)
class TrainConfig:
    experiment_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    checkpoint_dir: str
    log_dir: str
    early_stopping_patience: int
    early_stopping_min_delta: float


class EarlyStopping:
    """Останавливает обучение, если метрика не выросла на min_delta за patience эпох."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.waits = 0

    def step(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.waits = 0
            return False
        self.waits += 1
        return self.waits >= self.patience


def load_train_config(path: Path) -> TrainConfig:
    """Читает параметры запуска из YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"В {path} ожидается словарь параметров")
    early = payload.get("early_stopping")
    if not isinstance(early, dict):
        raise ValueError(f"В {path} нет блока early_stopping")
    required = (
        "experiment_name",
        "epochs",
        "batch_size",
        "learning_rate",
        "seed",
        "checkpoint_dir",
        "log_dir",
    )
    missing = [key for key in required if key not in payload]
    if "patience" not in early:
        missing.append("early_stopping.patience")
    if "min_delta" not in early:
        missing.append("early_stopping.min_delta")
    if missing:
        raise ValueError(f"В {path} не хватает полей: {', '.join(missing)}")
    return TrainConfig(
        experiment_name=str(payload["experiment_name"]),
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        learning_rate=float(payload["learning_rate"]),
        seed=int(payload["seed"]),
        checkpoint_dir=str(payload["checkpoint_dir"]),
        log_dir=str(payload["log_dir"]),
        early_stopping_patience=int(early["patience"]),
        early_stopping_min_delta=float(early["min_delta"]),
    )


def prepare_runtime(root: Path) -> Path:
    """Создаёт каталоги артефактов и проверяет конфиг обучения."""
    logger = get_logger("train")
    config_path = root / "configs" / "train.yaml"
    if not config_path.is_file():
        logger.error("Не найден конфиг %s", config_path)
        raise SystemExit(1)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logger.info("Конфиг обучения: %s", config_path)
    logger.info("Каталоги checkpoints и logs готовы.")
    return config_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def train_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Один шаг по батчу. Возвращает loss этого батча."""
    model.train()
    optimizer.zero_grad()
    logits = model(batch["token_ids"], batch["lengths"])
    loss = criterion(logits, batch["labels"])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Средний loss по обучающим батчам одной эпохи."""
    total = 0.0
    seen = 0
    for batch in loader:
        loss = train_batch(model, batch, criterion, optimizer)
        count = int(batch["labels"].shape[0])
        total += loss * count
        seen += count
    if seen == 0:
        return 0.0
    return total / seen


def validate_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    criterion: nn.Module,
) -> tuple[float, dict[str, object]]:
    """Loss и метрики на отложенной части без обновления весов."""
    model.eval()
    total = 0.0
    seen = 0
    truth: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["token_ids"], batch["lengths"])
            labels = batch["labels"]
            loss = criterion(logits, labels)
            count = int(labels.shape[0])
            total += float(loss.item()) * count
            seen += count
            truth.extend(int(item) for item in labels.tolist())
            predicted.extend(int(item) for item in logits.argmax(dim=-1).tolist())
    mean_loss = total / seen if seen else 0.0
    return mean_loss, compute_metrics(truth, predicted)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    """Accuracy, macro Precision/Recall/F1 и матрица ошибок 3×3."""
    matrix = confusion_matrix(y_true, y_pred, labels=METRIC_LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", labels=METRIC_LABELS, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="macro", labels=METRIC_LABELS, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=METRIC_LABELS, zero_division=0)
        ),
        "confusion_matrix": matrix.tolist(),
    }


def class_weight_tensor(labels: Iterable[str]) -> torch.Tensor:
    """Вес класса обратно пропорционален частоте. Сбалансированная выборка даёт единицы."""
    names = tuple(labels)
    counts = [sum(label == name for label in names) for name in VALID_LABELS]
    if any(count == 0 for count in counts):
        raise ValueError("В обучающей части нет одного из классов")
    total = float(sum(counts))
    n_classes = len(counts)
    return torch.tensor(
        [total / (n_classes * count) for count in counts],
        dtype=torch.float32,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, object],
    model_args: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if model_args is not None:
        payload["model_args"] = model_args
    torch.save(payload, path)


def run_training(
    train_loader: Iterable[dict[str, torch.Tensor]],
    val_loader: Iterable[dict[str, torch.Tensor]],
    model: nn.Module,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    checkpoint_dir: Path,
    log_dir: Path,
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 0.001,
    class_weights: torch.Tensor | None = None,
    model_args: dict[str, object] | None = None,
    on_epoch: Callable[[int, float], None] | None = None,
) -> dict[str, object]:
    """Кросс-энтропия, AdamW и ReduceLROnPlateau по F1-macro на val."""
    set_seed(seed)
    logger = get_logger("train")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )
    stopper = EarlyStopping(early_stopping_patience, early_stopping_min_delta)
    writer = SummaryWriter(log_dir=log_dir)
    best_f1 = -1.0
    stopped_early = False
    history: list[dict[str, object]] = []
    try:
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, metrics = validate_epoch(model, val_loader, criterion)
            f1_macro = float(metrics["f1_macro"])
            scheduler.step(f1_macro)
            _log_epoch(writer, epoch, train_loss, val_loss, metrics)
            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                model_args,
            )
            if f1_macro > best_f1:
                best_f1 = f1_macro
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    metrics,
                    model_args,
                )
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
            )
            logger.info(
                "Эпоха %s: train loss %.3f, val loss %.3f, accuracy %.3f, "
                "precision %.3f, recall %.3f, f1-macro %.3f",
                epoch,
                train_loss,
                val_loss,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                f1_macro,
            )
            logger.info("Матрица ошибок: %s", metrics["confusion_matrix"])
            if on_epoch is not None:
                on_epoch(epoch, f1_macro)
            if stopper.step(f1_macro):
                stopped_early = True
                logger.info(
                    "Ранняя остановка на эпохе %s: f1-macro не растёт %s эпох.",
                    epoch,
                    stopper.patience,
                )
                break
    finally:
        writer.close()
    return {"best_f1_macro": best_f1, "stopped_early": stopped_early, "history": history}


def _log_epoch(
    writer: SummaryWriter,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, object],
) -> None:
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
    writer.add_scalar("val/precision", metrics["precision"], epoch)
    writer.add_scalar("val/recall", metrics["recall"], epoch)
    writer.add_scalar("val/f1_macro", metrics["f1_macro"], epoch)
    writer.add_text("val/confusion_matrix", json.dumps(metrics["confusion_matrix"]), epoch)


def build_model(
    vocab: dict[str, int],
    model_config: ModelConfig,
    vectors: dict[str, list[float]] | None = None,
) -> tuple[SentimentLSTM, dict[str, object]]:
    """Собирает LSTM и словарь аргументов, который потом пишется в чекпоинт."""
    embedding_weight = None
    embedding_dim = model_config.embedding_dim
    if vectors is not None:
        if not vectors:
            raise ValueError("Пустой набор предобученных векторов")
        embedding_dim = len(next(iter(vectors.values())))
        if embedding_dim != model_config.embedding_dim:
            raise ValueError(
                f"В конфиге embedding_dim {model_config.embedding_dim}, у векторов {embedding_dim}"
            )
        embedding_weight, _hits = build_embedding_matrix(vocab, vectors, embedding_dim)
    model_args: dict[str, object] = {
        "embedding_dim": embedding_dim,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_layers,
        "dropout": model_config.dropout,
        "num_classes": model_config.num_classes,
        "padding_idx": model_config.padding_idx,
        "bidirectional": model_config.bidirectional,
        "pooling": model_config.pooling,
    }
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        num_classes=model_config.num_classes,
        padding_idx=model_config.padding_idx,
        bidirectional=model_config.bidirectional,
        pooling=model_config.pooling,
        embedding_weight=embedding_weight,
        freeze_embeddings=model_config.freeze_embeddings and embedding_weight is not None,
    )
    return model, model_args


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_train_config(prepare_runtime(root))
    model_config = load_model_config(root / "configs" / "model.yaml")
    train_path = root / "data" / "processed" / "train.csv"
    val_path = root / "data" / "processed" / "val.csv"
    vocab_path = root / "data" / "processed" / "vocab.json"
    if not train_path.is_file() or not val_path.is_file() or not vocab_path.is_file():
        raise SystemExit("Нет data/processed. Сначала соберите выборку.")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vectors = None
    if model_config.pretrained_embeddings:
        vectors = vectors_from_navec(root / model_config.pretrained_embeddings, vocab)
    set_seed(config.seed)
    model, model_args = build_model(vocab, model_config, vectors)
    train_loader = make_loader(
        train_path,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        token_dropout=model_config.token_dropout,
    )
    val_loader = make_loader(val_path, batch_size=config.batch_size, shuffle=False)
    weights = None
    if model_config.class_weights:
        weights = class_weight_tensor(_labels(train_path))
    run_training(
        train_loader,
        val_loader,
        model,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
        checkpoint_dir=root / config.checkpoint_dir,
        log_dir=root / config.log_dir / config.experiment_name,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        class_weights=weights,
        model_args=model_args,
    )


def _labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["label"] for row in csv.DictReader(handle)]


if __name__ == "__main__":
    main()
