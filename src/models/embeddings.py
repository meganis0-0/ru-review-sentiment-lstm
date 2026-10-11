"""Сборка матрицы эмбеддингов из готовых векторов слов."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn

from src.data.prepare import PAD_ID, UNK_ID

SPECIAL_TOKENS = {"<pad>", "<unk>"}


def build_embedding_matrix(
    vocab: dict[str, int],
    vectors: Mapping[str, Sequence[float]],
    dim: int,
) -> tuple[torch.Tensor, int]:
    """Кладёт известные векторы в строки словаря. Pad остаётся нулём, unk — средним."""
    if dim < 1:
        raise ValueError("Размерность эмбеддинга должна быть положительной")
    weight = torch.empty(len(vocab), dim)
    nn.init.uniform_(weight, -0.05, 0.05)
    matched: list[torch.Tensor] = []
    for token, index in vocab.items():
        if token in SPECIAL_TOKENS:
            continue
        vector = vectors.get(token)
        if vector is None:
            continue
        row = torch.as_tensor(list(vector), dtype=torch.float32)
        if tuple(row.shape) != (dim,):
            raise ValueError(
                f"Вектор «{token}» имеет размер {tuple(row.shape)}, ожидается {(dim,)}"
            )
        weight[index] = row
        matched.append(row)
    weight[PAD_ID].zero_()
    if matched and UNK_ID < weight.shape[0]:
        weight[UNK_ID] = torch.stack(matched).mean(dim=0)
    return weight, len(matched)


def vectors_from_navec(path: Path, vocab: dict[str, int]) -> dict[str, list[float]]:
    """Берёт из архива Navec только слова нашего словаря. «ё» приводится к «е»."""
    from navec import Navec

    navec = Navec.load(str(path))
    wanted = set(vocab) - SPECIAL_TOKENS
    chosen: dict[str, str] = {}
    for word in navec.vocab.words:
        key = word.lower().replace("ё", "е")
        if key not in wanted:
            continue
        if key not in chosen or word == key:
            chosen[key] = word
    return {key: navec[word].tolist() for key, word in chosen.items()}
