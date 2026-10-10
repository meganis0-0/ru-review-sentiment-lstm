"""Цикл обучения LSTM: эпоха, метрики, чекпоинт и TensorBoard."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

import torch
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
from src.models.rnn import NUM_CLASSES, SentimentLSTM
from src.utils import get_logger

EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 42
METRIC_LABELS = list(range(NUM_CLASSES))


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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


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
) -> dict[str, object]:
    """Кросс-энтропия, AdamW и ReduceLROnPlateau по F1-macro на val."""
    set_seed(seed)
    logger = get_logger("train")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )
    writer = SummaryWriter(log_dir=log_dir)
    best_f1 = -1.0
    history: list[dict[str, object]] = []
    try:
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, metrics = validate_epoch(model, val_loader, criterion)
            f1_macro = float(metrics["f1_macro"])
            scheduler.step(f1_macro)
            _log_epoch(writer, epoch, train_loss, val_loss, metrics)
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, metrics)
            if f1_macro > best_f1:
                best_f1 = f1_macro
                save_checkpoint(
                    checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, metrics
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
    finally:
        writer.close()
    return {"best_f1_macro": best_f1, "history": history}


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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    prepare_runtime(root)
    train_path = root / "data" / "processed" / "train.csv"
    val_path = root / "data" / "processed" / "val.csv"
    vocab_path = root / "data" / "processed" / "vocab.json"
    if not train_path.is_file() or not val_path.is_file() or not vocab_path.is_file():
        raise SystemExit("Нет data/processed. Сначала соберите выборку.")
    vocab_size = len(json.loads(vocab_path.read_text(encoding="utf-8")))
    set_seed(SEED)
    model = SentimentLSTM(vocab_size)
    train_loader = make_loader(train_path, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    val_loader = make_loader(val_path, batch_size=BATCH_SIZE, shuffle=False)
    run_training(
        train_loader,
        val_loader,
        model,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        checkpoint_dir=root / "checkpoints",
        log_dir=root / "logs" / "tensorboard",
    )


if __name__ == "__main__":
    main()
