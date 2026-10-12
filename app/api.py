"""HTTP-предсказание для одной из двух BiLSTM."""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import predict

app = FastAPI(title="Тональность отзывов", version="0.1.0")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    version: Literal["corpus", "rule"] = "corpus"


@app.post("/predict")
def predict_review(body: PredictRequest) -> dict[str, object]:
    """Класс и вероятности выбранной модели."""
    try:
        return predict(body.text, body.version)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
