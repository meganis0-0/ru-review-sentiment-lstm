"""Страница с двумя BiLSTM: корпус и правило для банков, мест и фильмов."""

import streamlit as st

from src.inference import VERSIONS, predict, reviews_from_file

EXAMPLES = (
    "Ужасное обслуживание, очередь и хамство",
    "Отличный банк, очень рекомендую",
    "Доставили заказ вчера утром",
)


def _reviews(review: str, upload: object | None) -> list[str]:
    if upload is not None:
        data = upload.getvalue()
        name = getattr(upload, "name", "")
        payload = data if isinstance(data, bytes) else bytes(data)
        return reviews_from_file(str(name), payload)
    if not review.strip():
        raise ValueError("В отзыве нет слов")
    return [review]


def _show(result: dict[str, object]) -> None:
    st.subheader(str(result["title"]))
    probabilities = result["probabilities"]
    if not isinstance(probabilities, dict):
        return
    for label in ("negative", "neutral", "positive"):
        chance = float(probabilities[label])
        st.write(f"{label}: {chance:.0%}")
        st.progress(min(max(chance, 0.0), 1.0))


st.set_page_config(page_title="Тональность отзывов")
st.title("Тональность отзывов на русском")
st.write("Две двунаправленные LSTM. На выходе класс и вероятности.")

version = st.selectbox(
    "Модель",
    list(VERSIONS),
    format_func=lambda key: VERSIONS[key]["title"],
)
st.caption(VERSIONS[version]["limitation"])
st.warning(
    "Модель оценивает текст отзыва, а не человека. Для решения о человеке её нельзя использовать."
)

example = st.selectbox("Пример", ("", *EXAMPLES), format_func=lambda item: item or "Свой текст")
review = st.text_area("Текст отзыва", value=example)
upload = st.file_uploader("Или файл .txt / .csv с колонкой text", type=["txt", "csv"])

if st.button("Оценить"):
    try:
        reviews = _reviews(review, upload)
    except (TypeError, ValueError) as exc:
        st.error(str(exc))
    else:
        for text in reviews:
            _show(predict(text, version))
