"""Новая разметка банков, мест и фильмов по фиксированному правилу.

Негатив — в тексте больше претензии. Позитив — больше похвалы.
Нет оценки или обе стороны равны — нейтрально. Старый data/processed не меняется.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.data.prepare import prepare_dataset, tokenize, write_processed
from src.data.validate import normalized_text

GOLD_SOURCES = ("bank", "geo", "kinopoisk")
NEGATIVE_STEMS = (
    "ужас",
    "отврат",
    "недовол",
    "обман",
    "мошен",
    "разоча",
    "кошмар",
    "отстой",
    "хамств",
    "хамск",
    "груб",
    "гряз",
    "вонюч",
    "тухл",
    "просроч",
    "слома",
    "брак",
    "жалоб",
    "позор",
    "бардак",
    "мерзк",
    "противн",
    "дрянь",
    "фигня",
    "обдира",
    "навяз",
    "штраф",
    "заблок",
    "антисанитар",
    "таракан",
    "очеред",
    "игнор",
    "развод",
    "кинул",
    "глюк",
    "лагает",
    "плохо",
    "плохой",
    "плохая",
    "плохое",
    "хуже",
)
POSITIVE_STEMS = (
    "отлич",
    "супер",
    "рекоменд",
    "прекрас",
    "доволен",
    "довольна",
    "удобн",
    "вежлив",
    "приятн",
    "советую",
    "восторг",
    "идеальн",
    "комфорт",
    "профессионал",
    "внимательн",
    "аккуратн",
    "порадова",
    "благодар",
    "спасибо",
    "шикарн",
    "великолеп",
    "замечательн",
    "нрав",
    "вкусн",
    "уютн",
    "хорош",
    "лучший",
    "лучшая",
    "лучшее",
)


def rule_label(text: str) -> str:
    """Ставит класс по перевесу претензии и похвалы."""
    negative, positive = _scores(tokenize(text))
    if negative > positive:
        return "negative"
    if positive > negative:
        return "positive"
    return "neutral"


def load_gold_rows(processed_dir: Path) -> list[dict[str, str]]:
    """Тексты банков, мест и фильмов из уже собранных частей."""
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for name in ("train", "val", "test"):
        path = processed_dir / f"{name}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["source"] not in GOLD_SOURCES:
                    continue
                key = normalized_text(row["text"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "review_id": row["review_id"],
                        "text": row["text"],
                        "label": rule_label(row["text"]),
                        "source": row["source"],
                        "label_origin": "rule",
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Разметить банки, места и фильмы по правилу")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/rule"))
    args = parser.parse_args()
    rows = load_gold_rows(args.processed)
    prepared = prepare_dataset(rows)
    prepared["notes"] = [
        "Метка по правилу: перевес претензии или похвалы, иначе нейтрально.",
        "Только bank, geo и kinopoisk.",
    ]
    write_processed(args.output, prepared)
    splits = prepared["splits"]
    assert isinstance(splits, dict)
    for name in ("train", "val", "test"):
        counts = {label: 0 for label in ("negative", "neutral", "positive")}
        for row in splits[name]:
            counts[row["label"]] += 1
        print(name, len(splits[name]), counts)


def _scores(tokens: list[str]) -> tuple[int, int]:
    negative = 0
    positive = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "не" and index + 1 < len(tokens):
            probe = tokens[index + 1]
            if _matches(probe, NEGATIVE_STEMS):
                positive += 1
                index += 2
                continue
            if _matches(probe, POSITIVE_STEMS):
                negative += 1
                index += 2
                continue
        if _matches(token, NEGATIVE_STEMS):
            negative += 1
        elif _matches(token, POSITIVE_STEMS):
            positive += 1
        index += 1
    return negative, positive


def _matches(token: str, stems: tuple[str, ...]) -> bool:
    return any(token == stem or (len(stem) >= 4 and token.startswith(stem)) for stem in stems)


if __name__ == "__main__":
    main()
