import csv
import time
from pathlib import Path

import torch

from src.data.dataset import (
    LabelTransform,
    ReviewDataset,
    TokenIdsTransform,
    make_loader,
)
from src.data.prepare import FIELDNAMES, MAX_LEN, PAD_ID
from src.data.validate import VALID_LABELS


def _write_reviews(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index in range(count):
            label = VALID_LABELS[index % len(VALID_LABELS)]
            token_id = index % 5 + 2
            writer.writerow(
                {
                    "review_id": str(index + 1),
                    "text": f"отзыв {index}",
                    "label": label,
                    "source": "bank",
                    "label_origin": "source",
                    "token_count": "2",
                    "token_ids": f"{token_id} {token_id} 0 0",
                }
            )


def test_transforms_pad_and_map_labels() -> None:
    token_ids, length = TokenIdsTransform(max_len=4)("5 6")

    assert token_ids == [5, 6, PAD_ID, PAD_ID]
    assert length == 2
    assert LabelTransform()("positive") == 2


def test_batch_feeds_embedding(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    _write_reviews(path, count=10)
    loader = make_loader(path, batch_size=4, shuffle=False, max_len=4)
    batch = next(iter(loader))
    hidden = torch.nn.Embedding(10, 8, padding_idx=PAD_ID)(batch["token_ids"])

    assert batch["token_ids"].shape == (4, 4)
    assert batch["token_ids"].dtype == torch.long
    assert batch["labels"].shape == (4,)
    assert batch["lengths"].tolist() == [2, 2, 2, 2]
    assert batch["labels"].tolist() == [0, 1, 2, 0]
    assert hidden.shape == (4, 4, 8)


def test_loader_keeps_file_order_and_covers_every_row(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    _write_reviews(path, count=10)
    loader = make_loader(path, batch_size=4, shuffle=False, max_len=4)
    labels = [int(label) for batch in loader for label in batch["labels"].tolist()]

    assert labels == [index % 3 for index in range(10)]
    assert len(ReviewDataset(path)) == 10


def test_loader_reads_a_thousand_rows_quickly(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    _write_reviews(path, count=1000)
    loader = make_loader(path, batch_size=64, shuffle=False, max_len=MAX_LEN)
    started = time.perf_counter()
    seen = sum(int(batch["token_ids"].shape[0]) for batch in loader)

    assert seen == 1000
    assert time.perf_counter() - started < 5
