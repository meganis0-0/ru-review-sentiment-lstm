from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.rnn import SentimentLSTM
from src.train import compute_metrics, prepare_runtime, run_training, train_batch, train_epoch


def _batch(labels: list[int]) -> dict[str, torch.Tensor]:
    count = len(labels)
    return {
        "token_ids": torch.randint(1, 12, (count, 5)),
        "lengths": torch.full((count,), 5),
        "labels": torch.tensor(labels),
    }


def _model() -> SentimentLSTM:
    return SentimentLSTM(vocab_size=12, embedding_dim=8, hidden_size=8, dropout=0.0)


def test_prepare_runtime_creates_directories(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "train.yaml").write_text("experiment_name: test\n", encoding="utf-8")

    config_path = prepare_runtime(tmp_path)

    assert config_path == tmp_path / "configs" / "train.yaml"
    assert (tmp_path / "checkpoints").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_prepare_runtime_fails_without_config(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        prepare_runtime(tmp_path)


def test_one_batch_loss_decreases() -> None:
    model = _model()
    batch = _batch([0, 1, 2, 0, 1, 2, 0, 1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    first = train_batch(model, batch, criterion, optimizer)
    last = first
    for _step in range(40):
        last = train_batch(model, batch, criterion, optimizer)

    assert last < first


def test_epoch_saves_checkpoint_and_tensorboard(tmp_path: Path) -> None:
    model = _model()
    train_loader = [_batch([0, 1, 2, 1]), _batch([2, 0, 1, 2])]
    val_loader = [_batch([0, 1, 2, 0])]
    result = run_training(
        train_loader,
        val_loader,
        model,
        epochs=1,
        learning_rate=1e-3,
        seed=42,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
    )
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "best.pt", map_location="cpu", weights_only=False
    )
    metrics = result["history"][0]
    events = list((tmp_path / "logs").glob("events.out.tfevents.*"))

    assert (tmp_path / "checkpoints" / "last.pt").is_file()
    assert checkpoint["epoch"] == 1
    assert metrics["confusion_matrix"]
    assert len(metrics["confusion_matrix"]) == 3
    assert {"accuracy", "precision", "recall", "f1_macro"} <= set(metrics)
    assert events
    restored = _model()
    restored.load_state_dict(checkpoint["model"])
    restored.eval()
    batch = _batch([0, 1])
    with torch.no_grad():
        original = model(batch["token_ids"], batch["lengths"])
        copied = restored(batch["token_ids"], batch["lengths"])
    assert torch.allclose(original, copied)


def test_train_epoch_averages_batch_losses() -> None:
    model = _model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_epoch(model, [_batch([0, 1, 2, 0])], criterion, optimizer)

    assert loss > 0


def test_compute_metrics_uses_all_three_classes() -> None:
    metrics = compute_metrics([0, 1], [0, 1])

    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["f1_macro"] < 1
