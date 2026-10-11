"""Батчи отзывов для LSTM: токены, длина без pad и индекс класса."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.prepare import MAX_LEN, PAD_ID, SEED
from src.data.validate import VALID_LABELS

LABEL_TO_INDEX = {label: index for index, label in enumerate(VALID_LABELS)}
DEFAULT_BATCH_SIZE = 32


class TokenIdsTransform:
    """Переводит строку token_ids в окно фиксированной длины и считает слова без pad."""

    def __init__(self, max_len: int = MAX_LEN) -> None:
        self.max_len = max_len

    def __call__(self, token_ids: str) -> tuple[list[int], int]:
        text = token_ids.strip()
        ids = [int(part) for part in text.split()][: self.max_len] if text else []
        length = sum(token_id != PAD_ID for token_id in ids)
        if len(ids) < self.max_len:
            ids.extend([PAD_ID] * (self.max_len - len(ids)))
        return ids, length


class LabelTransform:
    """Переводит имя класса в индекс: negative 0, neutral 1, positive 2."""

    def __call__(self, label: str) -> int:
        try:
            return LABEL_TO_INDEX[label]
        except KeyError as error:
            raise ValueError(f"Неизвестная метка: {label}") from error


class ReviewTransform:
    """Собирает из строки CSV поля, которые ждёт embedding."""

    def __init__(self, max_len: int = MAX_LEN) -> None:
        self.tokens = TokenIdsTransform(max_len)
        self.labels = LabelTransform()

    def __call__(self, row: dict[str, str]) -> dict[str, list[int] | int]:
        token_ids, length = self.tokens(row["token_ids"])
        return {
            "token_ids": token_ids,
            "label": self.labels(row["label"]),
            "length": length,
        }


class ReviewDataset(Dataset):
    def __init__(
        self, path: Path, transform: Callable[[dict[str, str]], dict] | None = None
    ) -> None:
        self.rows = _read_rows(path)
        self.transform = transform or ReviewTransform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        return self.transform(self.rows[index])


def drop_tokens(token_ids: list[int], rate: float, rng: random.Random) -> tuple[list[int], int]:
    """Случайно убирает слова отзыва и заново дополняет хвост pad. Хотя бы одно слово остаётся."""
    width = len(token_ids)
    content = [token for token in token_ids if token != PAD_ID]
    if rate <= 0 or len(content) <= 1:
        return list(token_ids), len(content)
    kept = [token for token in content if rng.random() >= rate]
    if not kept:
        kept = [content[0]]
    length = len(kept)
    if length < width:
        kept.extend([PAD_ID] * (width - length))
    return kept[:width], length


class TokenDropout(Dataset):
    """На каждой выдаче выкидывает часть слов, чтобы сеть не запоминала отзыв целиком."""

    def __init__(self, dataset: Dataset, rate: float, seed: int) -> None:
        self.dataset = dataset
        self.rate = rate
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        item = self.dataset[index]
        token_ids, length = drop_tokens(list(item["token_ids"]), self.rate, self.rng)
        return {"token_ids": token_ids, "label": item["label"], "length": length}


def collate_reviews(batch: list[dict[str, list[int] | int]]) -> dict[str, torch.Tensor]:
    """Складывает отзывы в тензоры: token_ids [B, T], labels [B], lengths [B]."""
    return {
        "token_ids": torch.tensor([item["token_ids"] for item in batch], dtype=torch.long),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "lengths": torch.tensor([item["length"] for item in batch], dtype=torch.long),
    }


def make_loader(
    path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    seed: int = SEED,
    max_len: int = MAX_LEN,
    num_workers: int = 0,
    token_dropout: float = 0.0,
) -> DataLoader:
    dataset: Dataset = ReviewDataset(path, transform=ReviewTransform(max_len))
    if token_dropout > 0:
        dataset = TokenDropout(dataset, token_dropout, seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_reviews,
        generator=generator,
    )


def rows_per_second(loader: DataLoader) -> float:
    started = time.perf_counter()
    seen = 0
    for batch in loader:
        seen += int(batch["token_ids"].shape[0])
    elapsed = time.perf_counter() - started
    if elapsed == 0:
        return float(seen)
    return seen / elapsed


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверить батч и скорость загрузки")
    parser.add_argument("--split", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--vocab", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = make_loader(args.split, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))
    vocab_size = _vocab_size(args.vocab)
    embedding = torch.nn.Embedding(vocab_size, 8, padding_idx=PAD_ID)
    hidden = embedding(batch["token_ids"])
    speed = rows_per_second(make_loader(args.split, batch_size=args.batch_size, shuffle=False))
    print(f"token_ids {tuple(batch['token_ids'].shape)}, labels {tuple(batch['labels'].shape)}")
    print(f"embedding {tuple(hidden.shape)}")
    print(f"{speed:.0f} строк/с")


def _vocab_size(path: Path) -> int:
    return len(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
