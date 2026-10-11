from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.config import load_model_config
from src.models.rnn import SentimentLSTM
from src.train import (
    EarlyStopping,
    class_weight_tensor,
    compute_metrics,
    load_train_config,
    prepare_runtime,
    run_training,
    train_batch,
    train_epoch,
)


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


def test_train_yaml_has_launch_parameters() -> None:
    config = load_train_config(Path(__file__).resolve().parents[1] / "configs" / "train.yaml")

    assert config.experiment_name == "ru-review-sentiment-lstm"
    assert config.epochs == 15
    assert config.batch_size == 32
    assert config.learning_rate == 0.001
    assert config.seed == 42
    assert config.checkpoint_dir == "checkpoints"
    assert config.early_stopping_patience == 3
    assert config.early_stopping_min_delta == 0.001


def test_load_train_config_rejects_missing_early_stopping(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("experiment_name: test\nepochs: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="early_stopping"):
        load_train_config(path)


def test_early_stopping_waits_for_patience() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.01)

    assert stopper.step(0.40) is False
    assert stopper.step(0.405) is False
    assert stopper.step(0.42) is False
    assert stopper.step(0.42) is False
    assert stopper.step(0.42) is True


def test_run_training_stops_when_metric_stalls(tmp_path: Path) -> None:
    result = run_training(
        [_batch([0, 1, 2, 0])],
        [_batch([0, 1, 2, 0])],
        _model(),
        epochs=5,
        learning_rate=0.0,
        seed=42,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
    )

    assert result["stopped_early"] is True
    assert len(result["history"]) == 2


def test_class_weights_grow_for_the_rare_label() -> None:
    balanced = class_weight_tensor(["negative", "neutral", "positive"])
    skewed = class_weight_tensor(["negative", "negative", "neutral", "positive"])

    assert torch.allclose(balanced, torch.ones(3))
    assert float(skewed[0]) < float(skewed[1])


def test_model_yaml_uses_the_best_recorded_setup() -> None:
    config = load_model_config(Path(__file__).resolve().parents[1] / "configs" / "model.yaml")

    assert config.embedding_dim == 300
    assert config.hidden_size == 128
    assert config.bidirectional is True
    assert config.pooling == "mean"
    assert config.freeze_embeddings is True
    assert config.token_dropout == 0.1
    assert config.class_weights is False
    assert config.pretrained_embeddings is not None
    assert config.pretrained_embeddings.endswith("navec_hudlit_v1_12B_500K_300d_100q.tar")
