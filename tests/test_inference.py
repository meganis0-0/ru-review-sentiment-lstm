import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.api import app
from app.main import _reviews
from src.inference import VERSIONS, Predictor, load_predictor, reviews_from_file
from src.models.rnn import SentimentLSTM


def _checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    model = SentimentLSTM(
        vocab_size=4,
        embedding_dim=8,
        hidden_size=8,
        dropout=0.0,
        bidirectional=True,
        pooling="mean",
    )
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": {
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "num_classes": 3,
                "padding_idx": 0,
                "bidirectional": True,
                "pooling": "mean",
            },
        },
        path,
    )
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
        json.dumps({"<pad>": 0, "<unk>": 1, "отличный": 2, "банк": 3}),
        encoding="utf-8",
    )
    return path, vocab


def test_predict_returns_three_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, vocab = _checkpoint(tmp_path)
    monkeypatch.setitem(VERSIONS["corpus"], "checkpoint", str(checkpoint))
    monkeypatch.setitem(VERSIONS["corpus"], "fallback", str(checkpoint))
    monkeypatch.setitem(VERSIONS["corpus"], "vocab", str(vocab))

    result = load_predictor("corpus", tmp_path).predict("отличный банк")

    assert result["version"] == "corpus"
    assert set(result["probabilities"]) == {"negative", "neutral", "positive"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
    assert result["label"] in result["probabilities"]


def test_empty_and_broken_input_fail(tmp_path: Path) -> None:
    checkpoint, vocab = _checkpoint(tmp_path)
    predictor = Predictor(_model(checkpoint), json.loads(vocab.read_text(encoding="utf-8")), "rule")

    with pytest.raises(TypeError):
        predictor.predict(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="нет слов"):
        predictor.predict("   ")
    with pytest.raises(ValueError, match="нет слов"):
        predictor.predict("...")


def test_file_reader_accepts_text_and_rejects_empty_csv() -> None:
    assert reviews_from_file("one.txt", "отличный банк".encode()) == ["отличный банк"]
    with pytest.raises(ValueError, match="text"):
        reviews_from_file("table.csv", b"label\nnegative\n")
    with pytest.raises(ValueError, match="нет слов"):
        reviews_from_file("empty.txt", b"  ")
    with pytest.raises(TypeError):
        reviews_from_file("one.txt", "не байты")  # type: ignore[arg-type]


def test_page_without_text_or_file_is_empty() -> None:
    with pytest.raises(ValueError, match="нет слов"):
        _reviews("  ", None)


def test_api_returns_json_and_rejects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.api.predict",
        lambda text, version="corpus": {
            "label": "positive",
            "title": "позитив",
            "probabilities": {"negative": 0.1, "neutral": 0.2, "positive": 0.7},
            "version": version,
            "limitation": "тест",
        },
    )
    client = TestClient(app)
    schema = client.get("/openapi.json")
    ok = client.post("/predict", json={"text": "отличный банк", "version": "rule"})
    empty = client.post("/predict", json={"text": "", "version": "corpus"})

    assert "/predict" in schema.json()["paths"]
    assert ok.status_code == 200
    assert ok.json()["version"] == "rule"
    assert empty.status_code == 422


def test_app_page_offers_both_models() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(Path(__file__).parents[1] / "app" / "main.py", default_timeout=30)
    page.run()

    assert not page.exception
    options = page.selectbox[0].options
    assert "Корпус, 100 000 отзывов" in options
    assert "Правило: банки, места, фильмы" in options


def _model(path: Path):
    from src.models.rnn import model_from_checkpoint

    return model_from_checkpoint(path)
