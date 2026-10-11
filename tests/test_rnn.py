from pathlib import Path

import pytest
import torch

from src.models.embeddings import build_embedding_matrix
from src.models.rnn import SentimentLSTM, model_from_checkpoint


def test_forward_returns_three_logits() -> None:
    model = SentimentLSTM(vocab_size=10, embedding_dim=8, hidden_size=8, dropout=0.0)
    model.eval()
    token_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
    lengths = torch.tensor([3, 2])
    logits = model(token_ids, lengths)

    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32
    assert not torch.allclose(torch.softmax(logits, dim=-1), logits)


def test_padding_does_not_change_last_state() -> None:
    model = SentimentLSTM(vocab_size=10, embedding_dim=8, hidden_size=8, dropout=0.0)
    model.eval()
    padded = torch.tensor([[2, 3, 4, 0, 0]])
    short = torch.tensor([[2, 3, 4]])

    padded_logits = model(padded, torch.tensor([3]))
    short_logits = model(short, torch.tensor([3]))

    assert torch.allclose(padded_logits, short_logits)


def test_pad_embedding_stays_zero_and_gradient_reaches_classifier() -> None:
    model = SentimentLSTM(vocab_size=10, embedding_dim=8, hidden_size=8)
    token_ids = torch.tensor([[2, 3, 0, 0]])
    logits = model(token_ids, torch.tensor([2]))
    logits.sum().backward()

    assert torch.count_nonzero(model.embedding.weight[0]) == 0
    assert model.classifier.weight.grad is not None


def test_empty_review_is_rejected() -> None:
    model = SentimentLSTM(vocab_size=10, embedding_dim=4, hidden_size=4, dropout=0.0)

    with pytest.raises(ValueError, match="1"):
        model(torch.tensor([[0, 0]]), torch.tensor([0]))


def test_padding_does_not_change_mean_pool_or_bidirectional_state() -> None:
    model = SentimentLSTM(
        vocab_size=10,
        embedding_dim=8,
        hidden_size=8,
        dropout=0.0,
        bidirectional=True,
        pooling="mean",
    )
    model.eval()
    padded = torch.tensor([[2, 3, 4, 0, 0]])
    short = torch.tensor([[2, 3, 4]])

    padded_logits = model(padded, torch.tensor([3]))
    short_logits = model(short, torch.tensor([3]))

    assert padded_logits.shape == (1, 3)
    assert model.classifier.in_features == 16
    assert torch.allclose(padded_logits, short_logits, atol=1e-5)


def test_unknown_pooling_is_rejected() -> None:
    with pytest.raises(ValueError, match="last"):
        SentimentLSTM(vocab_size=10, embedding_dim=4, hidden_size=4, pooling="sum")


def test_two_layers_still_return_class_logits() -> None:
    model = SentimentLSTM(
        vocab_size=10,
        embedding_dim=8,
        hidden_size=8,
        num_layers=2,
        dropout=0.2,
    )
    model.eval()
    logits = model(torch.tensor([[2, 3, 4]]), torch.tensor([3]))

    assert logits.shape == (1, 3)


def test_pretrained_row_is_kept_and_pad_stays_zero() -> None:
    vocab = {"<pad>": 0, "<unk>": 1, "банк": 2, "фильм": 3}
    weight, hits = build_embedding_matrix(vocab, {"банк": [1.0, 0.0]}, dim=2)

    assert hits == 1
    assert torch.count_nonzero(weight[0]) == 0
    assert torch.allclose(weight[2], torch.tensor([1.0, 0.0]))
    assert torch.allclose(weight[1], weight[2])


def test_checkpoint_restores_bidirectional_mean_pool(tmp_path: Path) -> None:
    model = SentimentLSTM(
        vocab_size=10,
        embedding_dim=8,
        hidden_size=8,
        dropout=0.0,
        bidirectional=True,
        pooling="mean",
    )
    model.eval()
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": {
                "bidirectional": True,
                "pooling": "mean",
                "hidden_size": 8,
                "dropout": 0.0,
            },
        },
        path,
    )
    restored = model_from_checkpoint(path)
    restored.eval()
    tokens = torch.tensor([[2, 3, 4, 0]])
    lengths = torch.tensor([3])

    with torch.no_grad():
        assert torch.allclose(model(tokens, lengths), restored(tokens, lengths))
