import pytest
import torch

from src.models.rnn import SentimentLSTM


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
