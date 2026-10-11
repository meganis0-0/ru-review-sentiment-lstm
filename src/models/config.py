"""Архитектура LSTM из configs/model.yaml. Незаданные поля остаются как в T2.3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.data.prepare import PAD_ID
from src.models.rnn import DROPOUT, EMBEDDING_DIM, HIDDEN_SIZE, NUM_CLASSES, NUM_LAYERS


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = EMBEDDING_DIM
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    num_classes: int = NUM_CLASSES
    padding_idx: int = PAD_ID
    bidirectional: bool = False
    pooling: str = "last"
    pretrained_embeddings: str | None = None
    freeze_embeddings: bool = False
    token_dropout: float = 0.0
    class_weights: bool = False


def load_model_config(path: Path) -> ModelConfig:
    """Читает yaml. Отсутствующий файл и пустые необязательные поля не меняют LSTM по умолчанию."""
    if not path.is_file():
        return ModelConfig()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"В {path} ожидается словарь параметров")
    pretrained = payload.get("pretrained_embeddings") or None
    return ModelConfig(
        embedding_dim=int(payload.get("embedding_dim", EMBEDDING_DIM)),
        hidden_size=int(payload.get("hidden_size", HIDDEN_SIZE)),
        num_layers=int(payload.get("num_layers", NUM_LAYERS)),
        dropout=float(payload.get("dropout", DROPOUT)),
        num_classes=int(payload.get("num_classes", NUM_CLASSES)),
        padding_idx=int(payload.get("padding_idx", PAD_ID)),
        bidirectional=bool(payload.get("bidirectional", False)),
        pooling=str(payload.get("pooling", "last")),
        pretrained_embeddings=str(pretrained) if pretrained else None,
        freeze_embeddings=bool(payload.get("freeze_embeddings", False)),
        token_dropout=float(payload.get("token_dropout", 0.0)),
        class_weights=bool(payload.get("class_weights", False)),
    )
