# Тональность отзывов на русском (LSTM)

Учебный сервис, который по тексту отзыва определяет тональность: негативная, нейтральная или позитивная.

Модель помогает быстро разбирать поток отзывов: на вход подаётся текст, на выход — класс и вероятности трёх тональностей.

## Постановка

| | |
|---|---|
| Вход | Текст отзыва на русском |
| Выход | Класс `negative` / `neutral` / `positive` и вероятности |
| Сеть | LSTM |
| Метрика | F1-macro |
| Цель | F1-macro ≥ 0.80 |
| Объём данных | 1000–5000 размеченных текстов |

Если целевая метрика не будет достигнута, в отчёте фиксируются baseline, ограничения датасета и что уже пробовали.

## Стек

Python 3.11+, PyTorch, pandas, scikit-learn, Optuna, Streamlit или Gradio, Docker Compose, GitHub Actions.

## Структура

```text
app/                  интерфейс
configs/              YAML-конфиги обучения, модели и Optuna
data/                 raw → interim → processed
docker/               Dockerfile и compose
notebooks/            EDA, baseline, разбор ошибок
src/data/             датасет и предобработка
src/models/           LSTM
src/                  обучение, оценка, HPO, инференс
tests/
```

Каталог `data/` в git не хранится. Версия датасета записана в `data.dvc`, сами файлы лежат в локальном remote `storage/`. Восстановление:

```powershell
.\.venv\Scripts\dvc pull
```

Папка `storage/` в GitHub не попадает. На другой машине её нужно скопировать рядом с репозиторием либо сменить DVC remote.

Сырая выборка собирается из корпуса [Russian Sentiment Dataset](https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset). В неё входят отзывы восьми сфер: одежда, другие товары, продукты, места, фильмы, аниме, банки и банковское приложение. Новости и посты соцсетей отброшены. Метки исходного файла: `0` нейтрально, `1` позитив, `2` негатив. Из каждой пары сфера-класс берётся 200 текстов, seed 42.

```powershell
.\.venv\Scripts\python -m src.data.collect --zip $env:USERPROFILE\Downloads\russian-sentiment-dataset.zip
```

Исходные метки всех отзывов лежат в `data/interim/labels.csv`. Пилот для ручной проверки — `data/interim/pilot_tasks.json`, 50 отзывов. Инструкция и схема Label Studio: `labeling/`.

```powershell
docker compose up label-studio
.\.venv\Scripts\python -m src.data.labeling
.\.venv\Scripts\python -m src.data.labeling --export data\interim\label-studio-export.json
```

## Статус

Каркас проекта готов. Код модели, обучение, подбор гиперпараметров и интерфейс ещё не написаны — они идут отдельными задачами на канбане.

## Окружение

Зависимости ставятся только в `.venv` или в образ Docker. Системный Python для этого проекта не используется.

### venv

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\pre-commit install
.\.venv\Scripts\ruff check .
.\.venv\Scripts\black --check .
.\.venv\Scripts\isort --check-only .
.\.venv\Scripts\pytest
```

### Docker

Проверки, заготовка обучения и интерфейс:

```powershell
docker compose run --rm checks
docker compose run --rm train
docker compose up app
```

Optuna dashboard: `docker compose up optuna`, затем http://localhost:8081.
Label Studio: `docker compose up label-studio`, затем http://localhost:8080.

Интерфейс: http://localhost:8501.

GPU для обучения подключается отдельным файлом, после установки драйвера NVIDIA и поддержки GPU в Docker Desktop:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm train
```
