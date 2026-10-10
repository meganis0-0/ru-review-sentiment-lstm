from pathlib import Path

import torch
from torch import nn

from src.models.rnn import SentimentLSTM
from src.train import save_checkpoint, set_seed, train_batch


def _separable_batch() -> dict[str, torch.Tensor]:
    return {
        "token_ids": torch.tensor(
            [
                [2, 2, 2, 0],
                [3, 3, 3, 0],
                [4, 4, 4, 0],
                [2, 2, 2, 0],
            ]
        ),
        "lengths": torch.tensor([3, 3, 3, 3]),
        "labels": torch.tensor([0, 1, 2, 0]),
    }


def _model() -> SentimentLSTM:
    return SentimentLSTM(vocab_size=8, embedding_dim=16, hidden_size=16, dropout=0.0)


def test_one_batch_can_be_overfit() -> None:
    model = _model()
    batch = _separable_batch()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    for _step in range(80):
        train_batch(model, batch, criterion, optimizer)

    model.eval()
    with torch.no_grad():
        logits = model(batch["token_ids"], batch["lengths"])
        predicted = logits.argmax(dim=-1)

    assert logits.shape == (4, 3)
    assert torch.equal(predicted, batch["labels"])


def test_backward_reaches_embedding_and_lstm() -> None:
    model = _model()
    batch = _separable_batch()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(batch["token_ids"], batch["lengths"]), batch["labels"])
    loss.backward()

    embedding_grad = model.embedding.weight.grad
    lstm_grad = model.lstm.weight_ih_l0.grad
    assert embedding_grad is not None
    assert lstm_grad is not None
    assert torch.count_nonzero(embedding_grad[0]) == 0
    assert torch.count_nonzero(embedding_grad[2]) > 0
    assert torch.count_nonzero(lstm_grad) > 0
    assert model.classifier.weight.grad is not None


def test_same_seed_builds_the_same_model() -> None:
    set_seed(42)
    first = _model()
    set_seed(42)
    second = _model()
    set_seed(7)
    other = _model()

    assert torch.equal(first.classifier.weight, second.classifier.weight)
    assert not torch.equal(first.classifier.weight, other.classifier.weight)

    batch = _separable_batch()
    criterion = nn.CrossEntropyLoss()
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=0.01)
    set_seed(42)
    third = _model()
    third_optimizer = torch.optim.AdamW(third.parameters(), lr=0.01)
    first_loss = train_batch(first, batch, criterion, first_optimizer)
    third_loss = train_batch(third, batch, criterion, third_optimizer)

    assert first_loss == third_loss


def test_checkpoint_restores_weights_and_optimizer(tmp_path: Path) -> None:
    model = _model()
    batch = _separable_batch()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    train_batch(model, batch, criterion, optimizer)
    path = tmp_path / "sanity.pt"
    save_checkpoint(path, model, optimizer, scheduler, epoch=1, metrics={"f1_macro": 0.5})

    restored = _model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])

    continued = train_batch(model, batch, criterion, optimizer)
    resumed = train_batch(restored, batch, criterion, restored_optimizer)

    assert checkpoint["epoch"] == 1
    assert continued == resumed
