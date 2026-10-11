"""LSTM-классификатор тональности: embedding, последнее состояние, логиты трёх классов."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from src.data.prepare import PAD_ID
from src.data.validate import VALID_LABELS

EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.3
NUM_CLASSES = len(VALID_LABELS)


class SentimentLSTM(nn.Module):
    """Отзыв -> логиты negative / neutral / positive.

    В LSTM попадает только начало отзыва до длины `lengths`. Хвост из pad
    не сдвигает последнее состояние.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        padding_idx: int = PAD_ID,
        bidirectional: bool = False,
        pooling: str = "last",
        embedding_weight: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        if pooling not in {"last", "mean"}:
            raise ValueError("Пулинг должен быть last или mean")
        super().__init__()
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.embedding = _embedding(
            vocab_size,
            embedding_dim,
            padding_idx,
            embedding_weight,
            freeze_embeddings,
        )
        directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            self.embedding.embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if int(lengths.min()) < 1:
            raise ValueError("Длина отзыва должна быть не меньше 1")
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output, (hidden, _cell) = self.lstm(packed)
        if self.pooling == "mean":
            pooled = _mean_pool(output, lengths, token_ids.size(1))
        elif self.bidirectional:
            pooled = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            pooled = hidden[-1]
        return self.classifier(self.dropout(pooled))


def _embedding(
    vocab_size: int,
    embedding_dim: int,
    padding_idx: int,
    embedding_weight: torch.Tensor | None,
    freeze_embeddings: bool,
) -> nn.Embedding:
    if embedding_weight is None:
        return nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
    weight = embedding_weight.detach().to(dtype=torch.float32).clone()
    if weight.ndim != 2 or weight.shape[0] != vocab_size:
        raise ValueError("Матрица эмбеддингов не совпадает со словарём")
    weight[padding_idx].zero_()
    return nn.Embedding.from_pretrained(weight, freeze=freeze_embeddings, padding_idx=padding_idx)


def _mean_pool(output: PackedSequence, lengths: torch.Tensor, width: int) -> torch.Tensor:
    unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=width)
    positions = torch.arange(unpacked.size(1), device=unpacked.device)
    mask = positions.unsqueeze(0) < lengths.to(unpacked.device).unsqueeze(1)
    weights = mask.unsqueeze(-1).to(dtype=unpacked.dtype)
    denom = lengths.to(device=unpacked.device, dtype=unpacked.dtype).clamp(min=1).unsqueeze(1)
    return (unpacked * weights).sum(dim=1) / denom


def model_from_checkpoint(path: Path) -> SentimentLSTM:
    """Собирает ту же архитектуру, с которой сохраняли чекпоинт, и загружает веса."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    args = payload.get("model_args") or {}
    vocab_size, embedding_dim = state["embedding.weight"].shape
    model = SentimentLSTM(
        vocab_size=int(vocab_size),
        embedding_dim=int(embedding_dim),
        hidden_size=int(args.get("hidden_size", HIDDEN_SIZE)),
        num_layers=int(args.get("num_layers", NUM_LAYERS)),
        dropout=float(args.get("dropout", DROPOUT)),
        num_classes=int(args.get("num_classes", NUM_CLASSES)),
        padding_idx=int(args.get("padding_idx", PAD_ID)),
        bidirectional=bool(args.get("bidirectional", False)),
        pooling=str(args.get("pooling", "last")),
    )
    model.load_state_dict(state)
    return model
