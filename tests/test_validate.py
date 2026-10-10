from src.data.validate import load_labels, validate_rows


def _row(
    text: str,
    review_id: str,
    label: str = "neutral",
    source: str = "bank",
) -> dict[str, str]:
    return {"review_id": review_id, "text": text, "label": label, "source": source}


def test_validate_rows_flags_broken_and_duplicates() -> None:
    rows = [
        _row("", "1", label="нет"),
        _row("один и тот же отзыв", "2", label="negative", source="rureviews"),
        _row("один и тот же отзыв", "2", label="negative", source="ru-reviews-classification"),
        _row("нормальный отзыв о банке", "3", label="positive"),
    ]
    report = validate_rows(rows)
    broken = report["broken"]

    assert broken["empty_text"] == 1
    assert broken["invalid_label"] == 1
    assert broken["duplicate_ids"] == 1
    assert report["duplicates"]["groups"] == 1
    assert report["duplicates"]["extra_rows"] == 1
    assert any("Битые строки" in problem for problem in report["problems"])


def test_validate_rows_treats_newline_variants_as_duplicates() -> None:
    rows = [
        _row("один\nотзыв о юбке", "1", source="rureviews"),
        _row("один\r\nотзыв о юбке", "2", source="ru-reviews-classification"),
        _row("другой отзыв", "3"),
    ]
    report = validate_rows(rows)

    assert report["duplicates"]["groups"] == 1
    assert report["duplicates"]["examples"][0]["sources"] == [
        "rureviews",
        "ru-reviews-classification",
    ]


def test_validate_rows_accepts_balanced_sample() -> None:
    rows = []
    index = 1
    for label in ("negative", "neutral", "positive"):
        for _copy in range(6):
            rows.append(_row(f"отзыв номер {index} достаточно длинный", str(index), label=label))
            index += 1
    report = validate_rows(rows)

    assert report["imbalanced"] is False
    assert report["classes"] == {"negative": 6, "neutral": 6, "positive": 6}
    assert report["length"]["high_outliers"] == 0
    assert any("Классы близки" in problem for problem in report["problems"])


def test_validate_rows_marks_class_and_length_outliers() -> None:
    rows = [_row("обычный текст отзыва", str(index), label="negative") for index in range(1, 13)]
    rows.append(_row("я" * 500, "13", label="positive"))
    report = validate_rows(rows)

    assert report["imbalanced"] is True
    assert report["classes"]["neutral"] == 0
    assert report["length"]["high_outliers"] == 1
    assert report["length"]["short"] == 0


def test_load_labels_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("text\nпривет\n", encoding="utf-8")

    try:
        load_labels(path)
    except ValueError as error:
        assert "label" in str(error)
    else:
        raise AssertionError("ожидалась ошибка из-за чужих колонок")
