import csv
from pathlib import Path

from src.data.prepare import FIELDNAMES
from src.evaluate import format_examples, format_summary, predict_file, summarize
from src.models.rnn import SentimentLSTM


def _record(
    label: str,
    predicted: str,
    source: str,
    *,
    confidence: float = 0.8,
    text: str = "нормальный отзыв",
    token_count: int = 12,
    origin: str = "source",
) -> dict[str, object]:
    return {
        "review_id": "1",
        "text": text,
        "label": label,
        "predicted": predicted,
        "source": source,
        "label_origin": origin,
        "token_count": token_count,
        "confidence": confidence,
        "correct": label == predicted,
    }


def test_summary_separates_adjacent_mistakes_from_polarity_flips() -> None:
    records = [
        _record("negative", "negative", "bank"),
        _record("neutral", "neutral", "bank"),
        _record("positive", "positive", "bank"),
        _record("negative", "positive", "rureviews", confidence=0.93, text="рубашка"),
        _record("neutral", "negative", "rureviews", confidence=0.7),
        _record("positive", "neutral", "geo"),
        _record("positive", "positive", "geo"),
    ]

    summary = summarize(records)
    text = format_summary(summary, baseline_f1=0.612, val_f1=0.644)

    assert summary["adjacent_errors"] == 2
    assert summary["polarity_errors"] == 1
    assert summary["by_source"][0]["source"] == "rureviews"
    assert "Слабее всего" in text
    assert "0.612" in text
    assert "негатив → позитив" in format_examples(summary)


def test_truncated_slice_uses_the_token_window() -> None:
    records = [
        _record("negative", "negative", "bank", token_count=250),
        _record("neutral", "positive", "bank", token_count=250),
        _record("positive", "positive", "bank", token_count=20),
    ]

    summary = summarize(records)

    assert summary["truncated"]["support"] == 2
    assert summary["truncated"]["accuracy"] == 0.5


def test_predict_file_follows_csv_order(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index, label in enumerate(("negative", "neutral", "positive", "negative")):
            writer.writerow(
                {
                    "review_id": str(index + 1),
                    "text": f"отзыв {index}",
                    "label": label,
                    "source": "bank",
                    "label_origin": "source",
                    "token_count": "2",
                    "token_ids": "2 3 0 0",
                }
            )
    model = SentimentLSTM(vocab_size=10, embedding_dim=8, hidden_size=8, dropout=0.0)
    model.eval()

    records = predict_file(model, path, batch_size=2)

    assert [row["review_id"] for row in records] == ["1", "2", "3", "4"]
    assert [row["label"] for row in records] == ["negative", "neutral", "positive", "negative"]
    assert all(row["predicted"] in {"negative", "neutral", "positive"} for row in records)
