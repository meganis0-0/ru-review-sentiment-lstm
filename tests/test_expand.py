from src.data.collect import REVIEW_SOURCES
from src.data.expand import choose_additions, equal_target, fill_to_total, usable_additions
from src.data.validate import VALID_LABELS


def _row(text: str, source: str, label: str) -> dict[str, str]:
    return {"text": text, "source": source, "label": label}


def test_additions_match_the_smallest_cell() -> None:
    train_counts = {(source, label): 2 for source in REVIEW_SOURCES for label in VALID_LABELS}
    candidates = {}
    for source in REVIEW_SOURCES:
        for label in VALID_LABELS:
            extra = 1 if source == "bank" else 5
            candidates[(source, label)] = [
                _row(f"новый отзыв {source} {label} {index}", source, label)
                for index in range(extra)
            ]
    available = {key: len(rows) for key, rows in candidates.items()}
    chosen = choose_additions(candidates, train_counts, seed=42)
    counts: dict[tuple[str, str], int] = {}
    for row in chosen:
        key = (row["source"], row["label"])
        counts[key] = counts.get(key, 0) + 1

    assert equal_target(train_counts, available) == 3
    assert counts[("bank", "negative")] == 1
    assert counts[("geo", "positive")] == 1
    assert len(chosen) == 24


def test_short_cell_does_not_block_the_rest() -> None:
    train_counts = {(source, label): 2 for source in REVIEW_SOURCES for label in VALID_LABELS}
    candidates = {
        ("bank", "negative"): [_row("ещё один банковский отзыв", "bank", "negative")],
        ("geo", "negative"): [
            _row(f"новый отзыв о месте номер {index}", "geo", "negative") for index in range(20)
        ],
    }
    chosen = fill_to_total(candidates, train_counts, target_total=48 + 6, seed=42)
    counts: dict[tuple[str, str], int] = {}
    for row in chosen:
        key = (row["source"], row["label"])
        counts[key] = counts.get(key, 0) + 1

    assert counts[("bank", "negative")] == 1
    assert counts[("geo", "negative")] == 5
    assert len(chosen) == 6


def test_usable_additions_drop_held_and_short_text() -> None:
    rows = [
        _row("уже был в выборке", "bank", "negative"),
        _row("ок", "bank", "negative"),
        _row("новый длинный отзыв про банк", "bank", "negative"),
    ]
    grouped = usable_additions(rows, {"уже был в выборке"})

    assert [row["text"] for row in grouped[("bank", "negative")]] == [
        "новый длинный отзыв про банк"
    ]
