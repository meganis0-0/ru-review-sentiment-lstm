from src.data.collect import prepare_rows, sample_rows


def test_prepare_rows_keeps_review_sources_and_drops_duplicates() -> None:
    records = [
        {"text": "Вкусно", "label": "1", "src": "perekrestok"},
        {"text": "Вкусно", "label": "1", "src": "geo"},
        {"text": "  ", "label": "2", "src": "bank"},
        {"text": "Поезд задержан", "label": "2", "src": "news"},
        {"text": "Карта пришла", "label": "0", "src": "sber"},
        {"text": "Неизвестная метка", "label": "9", "src": "bank"},
    ]

    rows, skipped = prepare_rows(records)

    assert skipped == 1
    assert rows == [
        {"text": "Вкусно", "label": "positive", "source": "perekrestok"},
        {"text": "Карта пришла", "label": "neutral", "source": "sber"},
    ]


def test_sample_rows_takes_equal_cells() -> None:
    rows = []
    for source in ("bank", "geo"):
        for label in ("negative", "neutral", "positive"):
            for index in range(5):
                rows.append(
                    {
                        "text": f"{source}-{label}-{index}",
                        "label": label,
                        "source": source,
                    }
                )

    sampled = sample_rows(rows, per_cell=2, seed=42)

    assert len(sampled) == 12
    assert len({row["text"] for row in sampled}) == 12
    again = sample_rows(rows, per_cell=2, seed=42)
    assert [row["text"] for row in again] == [row["text"] for row in sampled]
