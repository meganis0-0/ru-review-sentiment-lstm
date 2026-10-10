"""Токенизация отзывов и воспроизводимое разбиение train/val/test."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.data.validate import VALID_LABELS, load_labels, normalized_text

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_LEN = 200
MIN_FREQ = 2
MIN_WORD = 4
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

TOKEN_RE = re.compile(r"[0-9a-zа-я]+")
SPLIT_NAMES = ("train", "val", "test")
FIELDNAMES = (
    "review_id",
    "text",
    "label",
    "source",
    "label_origin",
    "token_count",
    "token_ids",
)


def tokenize(text: str) -> list[str]:
    """Режет отзыв на слова: нижний регистр, «ё» как «е», без знаков препинания."""
    normalized = normalized_text(text).lower().replace("ё", "е")
    return TOKEN_RE.findall(normalized)


def repair_text(text: str) -> str:
    """Чинит отзыв, у которого русские буквы сохранились в неверной кодировке."""
    normalized = normalized_text(text)
    if "Ð" not in normalized and "Ñ" not in normalized:
        return normalized
    try:
        return normalized.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return normalized


def clean_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Убирает мусор и оставляет один экземпляр каждого текста."""
    kept: dict[str, dict[str, str]] = {}
    dropped = {"duplicate": 0, "unusable": 0, "invalid": 0}
    for row in rows:
        if row.get("label", "").strip() not in VALID_LABELS:
            dropped["invalid"] += 1
            continue
        repaired = repair_text(row.get("text", ""))
        row = {**row, "text": repaired}
        if not _usable(tokenize(repaired)):
            dropped["unusable"] += 1
            continue
        current = kept.get(repaired)
        if current is None:
            kept[repaired] = row
            continue
        dropped["duplicate"] += 1
        if _prefer(row, current):
            kept[repaired] = row
    ordered = sorted(kept.values(), key=lambda row: int(row["review_id"]))
    return ordered, dropped


def split_rows(
    rows: list[dict[str, str]],
    seed: int = SEED,
) -> dict[str, list[dict[str, str]]]:
    """Делит отзывы 80/10/10 внутри каждой пары сфера-класс."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["label"])].append(row)

    rng = random.Random(seed)
    parts: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_NAMES}
    for key in sorted(groups):
        bucket = list(groups[key])
        rng.shuffle(bucket)
        n_test = int(len(bucket) * TEST_RATIO)
        n_val = int(len(bucket) * VAL_RATIO)
        parts["test"].extend(bucket[:n_test])
        parts["val"].extend(bucket[n_test : n_test + n_val])
        parts["train"].extend(bucket[n_test + n_val :])
    for name in SPLIT_NAMES:
        rng.shuffle(parts[name])
    return parts


def build_vocab(token_lists: list[list[str]], min_freq: int = MIN_FREQ) -> dict[str, int]:
    """Собирает словарь по обучающей части. Редкие слова становятся unk."""
    counts: Counter[str] = Counter(token for tokens in token_lists for token in tokens)
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    frequent = [
        (token, count)
        for token, count in counts.items()
        if count >= min_freq and token not in vocab
    ]
    frequent.sort(key=lambda item: (-item[1], item[0]))
    for token, _count in frequent:
        vocab[token] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
    """Оставляет начало отзыва и дополняет хвост нулями до длины окна."""
    ids = [vocab.get(token, UNK_ID) for token in tokens[:max_len]]
    ids.extend([PAD_ID] * (max_len - len(ids)))
    return ids


def prepare_dataset(
    rows: list[dict[str, str]],
    seed: int = SEED,
    max_len: int = MAX_LEN,
    min_freq: int = MIN_FREQ,
) -> dict[str, object]:
    cleaned, dropped = clean_rows(rows)
    parts = split_rows(cleaned, seed)
    vocab = build_vocab([tokenize(row["text"]) for row in parts["train"]], min_freq)
    encoded = {
        name: [_encoded_row(row, vocab, max_len) for row in parts[name]] for name in SPLIT_NAMES
    }
    return {
        "splits": encoded,
        "vocab": vocab,
        "dropped": dropped,
        "seed": seed,
        "max_len": max_len,
        "min_freq": min_freq,
    }


def write_processed(output_dir: Path, prepared: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = prepared["splits"]
    assert isinstance(splits, dict)
    for name in SPLIT_NAMES:
        _write_split(output_dir / f"{name}.csv", splits[name])
    vocab = prepared["vocab"]
    assert isinstance(vocab, dict)
    (output_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "meta.json").write_text(
        json.dumps(_meta(prepared), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _usable(tokens: list[str]) -> bool:
    return any(len(token) >= MIN_WORD and len(set(token)) > 1 for token in tokens)


def _prefer(candidate: dict[str, str], current: dict[str, str]) -> bool:
    candidate_manual = candidate.get("label_origin") == "manual"
    current_manual = current.get("label_origin") == "manual"
    if candidate_manual != current_manual:
        return candidate_manual
    return int(candidate["review_id"]) < int(current["review_id"])


def _encoded_row(row: dict[str, str], vocab: dict[str, int], max_len: int) -> dict[str, str]:
    tokens = tokenize(row["text"])
    ids = encode(tokens, vocab, max_len)
    return {
        "review_id": row["review_id"],
        "text": normalized_text(row["text"]),
        "label": row["label"],
        "source": row["source"],
        "label_origin": row.get("label_origin", "source"),
        "token_count": str(len(tokens)),
        "token_ids": " ".join(str(token_id) for token_id in ids),
    }


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _meta(prepared: dict[str, object]) -> dict[str, object]:
    splits = prepared["splits"]
    vocab = prepared["vocab"]
    assert isinstance(splits, dict)
    assert isinstance(vocab, dict)
    counts = {}
    truncated = 0
    for name in SPLIT_NAMES:
        rows = splits[name]
        assert isinstance(rows, list)
        by_label: Counter[str] = Counter(row["label"] for row in rows)
        counts[name] = {
            "rows": len(rows),
            "classes": {label: by_label[label] for label in VALID_LABELS},
        }
        truncated += sum(1 for row in rows if int(row["token_count"]) > int(prepared["max_len"]))
    return {
        "seed": prepared["seed"],
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "stratify": ["source", "label"],
        "max_len": prepared["max_len"],
        "min_freq": prepared["min_freq"],
        "min_word": MIN_WORD,
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "vocab_size": len(vocab),
        "dropped": prepared["dropped"],
        "truncated": truncated,
        "splits": counts,
        "notes": [
            "Повтор текста остаётся в одной части выборки.",
            "Строка без слова из 4 букв удалена.",
            "Текст в неверной кодировке восстановлен.",
            "Словарь собран только по train.",
            "В окне первые max_len токенов, хвост дополнен pad.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подготовить train/val/test")
    parser.add_argument("--labels", type=Path, default=Path("data/interim/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_dataset(load_labels(args.labels))
    write_processed(args.output_dir, prepared)
    meta = _meta(prepared)
    dropped = meta["dropped"]
    assert isinstance(dropped, dict)
    print("Удалено повторов: {duplicate}. Удалено мусорных строк: {unusable}.".format(**dropped))
    splits = meta["splits"]
    assert isinstance(splits, dict)
    for name in SPLIT_NAMES:
        part = splits[name]
        assert isinstance(part, dict)
        print(f"{name}: {part['rows']}")
    print(f"Словарь: {meta['vocab_size']}. Обрезано по окну: {meta['truncated']}.")


if __name__ == "__main__":
    main()
