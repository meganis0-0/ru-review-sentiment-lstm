from collections import Counter

from src.data.prepare import (
    MAX_LEN,
    MIN_FREQ,
    UNK_ID,
    build_vocab,
    clean_rows,
    encode,
    prepare_dataset,
    tokenize,
)
from src.data.validate import normalized_text


def _row(
    text: str,
    review_id: str,
    label: str = "neutral",
    source: str = "bank",
    label_origin: str = "source",
) -> dict[str, str]:
    return {
        "review_id": review_id,
        "text": text,
        "label": label,
        "source": source,
        "label_origin": label_origin,
    }


def _corpus() -> list[dict[str, str]]:
    rows = []
    index = 1
    for source in ("bank", "geo"):
        for label in ("negative", "neutral", "positive"):
            for copy in range(10):
                rows.append(
                    _row(
                        f"длинный отзыв номер {index} про {source} и {label}",
                        str(index),
                        label=label,
                        source=source,
                    )
                )
                index += 1
    rows.append(_row("один и тот же отзыв о юбке", "100", source="rureviews"))
    rows.append(_row("один и тот же отзыв о юбке\r\n", "101", source="ru-reviews-classification"))
    rows.append(_row("👌", "102", label="positive", source="sber"))
    rows.append(_row("Молодцы!", "103", label="positive", source="sber"))
    return rows


def test_clean_rows_repairs_broken_encoding() -> None:
    good = "Хорошая рубашка! Все подошло по размеру. Спасибо!"
    broken = good.encode("utf-8").decode("latin-1")
    cleaned, dropped = clean_rows([_row(broken, "1", label="positive", source="rureviews")])

    assert dropped["unusable"] == 0
    assert cleaned[0]["text"] == good


def test_tokenize_normalizes_yo_and_punctuation() -> None:
    assert tokenize("Ёлка, супер!") == ["елка", "супер"]


def test_clean_rows_drops_duplicate_and_keeps_real_short_review() -> None:
    cleaned, dropped = clean_rows(_corpus())
    texts = {row["text"].strip() for row in cleaned}

    assert dropped["duplicate"] == 1
    assert dropped["unusable"] == 1
    assert "Молодцы!" in texts
    assert sum("юбке" in row["text"] for row in cleaned) == 1


def test_prepare_dataset_is_stratified_and_reproducible() -> None:
    first = prepare_dataset(_corpus(), seed=42)
    second = prepare_dataset(_corpus(), seed=42)
    splits = first["splits"]

    assert [row["review_id"] for row in splits["train"]] == [
        row["review_id"] for row in second["splits"]["train"]
    ]
    for name in ("train", "val", "test"):
        for source in ("bank", "geo"):
            labels = [row["label"] for row in splits[name] if row["source"] == source]
            assert len(set(Counter(labels).values())) == 1
    keys = {
        name: {normalized_text(row["text"]) for row in splits[name]}
        for name in ("train", "val", "test")
    }
    assert keys["train"].isdisjoint(keys["val"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["val"].isdisjoint(keys["test"])


def test_vocab_and_padding_do_not_learn_from_test() -> None:
    prepared = prepare_dataset(_corpus(), seed=42, min_freq=MIN_FREQ)
    splits = prepared["splits"]
    train_counts: Counter[str] = Counter()
    for row in splits["train"]:
        train_counts.update(tokenize(row["text"]))

    for name in ("val", "test"):
        for row in splits[name]:
            ids = [int(part) for part in row["token_ids"].split()]
            assert len(ids) == MAX_LEN
            tokens = tokenize(row["text"])
            for token, token_id in zip(tokens, ids):
                if train_counts[token] < MIN_FREQ:
                    assert token_id == UNK_ID
                else:
                    assert token_id == prepared["vocab"][token]
            if len(tokens) < MAX_LEN:
                assert ids[-1] == 0


def test_frozen_val_and_test_stay_while_new_rows_join_train() -> None:
    rows = _corpus()
    first = prepare_dataset(rows, seed=42)
    held = {
        normalized_text(row["text"]): name
        for name in ("val", "test")
        for row in first["splits"][name]
    }
    extra = _row("ещё один длинный отзыв про банк и негатив", "999", "negative", "bank")
    second = prepare_dataset([*rows, extra], seed=1, frozen=held)
    again = {
        normalized_text(row["text"]): name
        for name in ("val", "test")
        for row in second["splits"][name]
    }

    assert again == held
    assert normalized_text(extra["text"]) in {
        normalized_text(row["text"]) for row in second["splits"]["train"]
    }


def test_encode_keeps_the_start_of_a_long_review() -> None:
    tokens = [f"слово{index}" for index in range(MAX_LEN + 5)]
    vocab = build_vocab([tokens], min_freq=1)
    ids = encode(tokens, vocab, MAX_LEN)

    assert len(ids) == MAX_LEN
    assert 0 not in ids
    assert ids[0] == vocab["слово0"]
