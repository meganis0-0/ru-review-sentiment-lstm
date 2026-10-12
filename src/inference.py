"""Предсказание тональности одной из двух обученных BiLSTM."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import torch

from src.data.prepare import MAX_LEN, PAD_ID, encode, tokenize
from src.data.validate import LABEL_TITLES, VALID_LABELS
from src.models.rnn import SentimentLSTM, model_from_checkpoint

VERSIONS: dict[str, dict[str, str]] = {
    "corpus": {
        "title": "Корпус, 100 000 отзывов",
        "checkpoint": "checkpoints/best.pt",
        "fallback": "models/corpus.pt",
        "vocab": "data/processed/vocab.json",
        "limitation": (
            "Восемь сфер и исходные метки корпуса. Val F1-macro 0.722. " "Нейтральный класс слабый."
        ),
    },
    "rule": {
        "title": "Правило: банки, места, фильмы",
        "checkpoint": "checkpoints/rule/best.pt",
        "fallback": "models/rule.pt",
        "vocab": "data/rule/vocab.json",
        "limitation": (
            "Училась только на банках, местах и фильмах. "
            "Одежда, товары, продукты и аниме в обучение не входили. "
            "Val и test F1-macro 0.829."
        ),
    },
}


class Predictor:
    """Одна загруженная сеть и её словарь."""

    def __init__(self, model: SentimentLSTM, vocab: dict[str, int], version: str) -> None:
        self.model = model
        self.model.eval()
        self.vocab = vocab
        self.version = version

    def predict(self, text: str) -> dict[str, object]:
        """Класс и вероятности. Пустой текст и текст без слов — ошибка."""
        if not isinstance(text, str):
            raise TypeError("Отзыв должен быть строкой")
        tokens = tokenize(text.strip())
        if not tokens:
            raise ValueError("В отзыве нет слов")
        ids = encode(tokens, self.vocab, MAX_LEN)
        length = min(len(tokens), MAX_LEN)
        token_ids = torch.tensor([ids], dtype=torch.long)
        lengths = torch.tensor([length], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(token_ids, lengths)
        probabilities = torch.softmax(logits, dim=-1)[0]
        pairs = {label: float(probabilities[index]) for index, label in enumerate(VALID_LABELS)}
        label = max(pairs, key=pairs.get)
        return {
            "label": label,
            "title": LABEL_TITLES[label],
            "probabilities": pairs,
            "version": self.version,
            "limitation": VERSIONS[self.version]["limitation"],
        }


def predict(text: str, version: str = "corpus", root: Path | None = None) -> dict[str, object]:
    """Загружает выбранную модель и возвращает класс с вероятностями."""
    return load_predictor(version, root).predict(text)


_LOADED: dict[tuple[str, Path, Path], Predictor] = {}


def load_predictor(version: str, root: Path | None = None) -> Predictor:
    """Берёт чекпоинт из checkpoints, а если его нет — копию из models."""
    spec = _version(version)
    base = Path(__file__).resolve().parents[1] if root is None else root
    checkpoint = _existing(base, spec["checkpoint"], spec["fallback"])
    vocab_path = base / spec["vocab"]
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Нет словаря {vocab_path}")
    key = (version, checkpoint.resolve(), vocab_path.resolve())
    loaded = _LOADED.get(key)
    if loaded is None:
        loaded = Predictor(model_from_checkpoint(checkpoint), _read_vocab(vocab_path), version)
        _LOADED[key] = loaded
    return loaded


def reviews_from_file(filename: str, payload: bytes) -> list[str]:
    """Один отзыв из .txt или строки колонки text из .csv."""
    if not isinstance(filename, str) or not isinstance(payload, bytes):
        raise TypeError("Файл должен быть именем и набором байтов")
    name = filename.lower()
    try:
        raw = payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("Файл должен быть в UTF-8") from exc
    if name.endswith(".txt"):
        reviews = [raw.strip()] if raw.strip() else []
    elif name.endswith(".csv"):
        reviews = _csv_texts(raw)
    else:
        raise ValueError("Нужен файл .txt или .csv")
    if not reviews:
        raise ValueError("В отзыве нет слов")
    return reviews


def _version(version: str) -> dict[str, str]:
    if version not in VERSIONS:
        raise ValueError("Неизвестная модель: нужна corpus или rule")
    return VERSIONS[version]


def _existing(root: Path, first: str, second: str) -> Path:
    for relative in (first, second):
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError(f"Нет чекпоинта {first} и копии {second}")


def _read_vocab(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"В {path} ожидается словарь")
    vocab = {str(token): int(index) for token, index in payload.items()}
    if vocab.get("<pad>") != PAD_ID:
        raise ValueError("Словарь должен начинаться с <pad>")
    return vocab


def _csv_texts(raw: str) -> list[str]:
    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames or "text" not in reader.fieldnames:
        raise ValueError("В CSV нужна колонка text")
    return [row["text"].strip() for row in reader if row.get("text", "").strip()]
