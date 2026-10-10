"""LSTM-классификатор тональности: embedding, последнее состояние, логиты трёх классов."""

from __future__ import annotations

import torch
from torch import nn

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
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

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
        _, (hidden, _cell) = self.lstm(packed)
        last_state = hidden[-1]
        return self.classifier(self.dropout(last_state))
