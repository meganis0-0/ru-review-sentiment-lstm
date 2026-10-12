# Тональность отзывов на русском (LSTM)

Учебный сервис: по тексту отзыва на русском определяет тональность. На выходе класс `negative`, `neutral` или `positive` и вероятности трёх классов.

Сеть — двунаправленная LSTM. Цель проекта — F1-macro не ниже 0.80. На исходных метках корпуса цель не взята: val F1-macro 0.722. На отдельной разметке банков, мест и фильмов та же сеть даёт val и test 0.829. Обе версии подключены к странице и к API.

Лицензия кода — MIT, copyright MEGANIS. Карточки: `MODEL_CARD.md`, `DATASET_CARD.md`.

## Постановка

| | |
|---|---|
| Вход | Текст отзыва на русском |
| Выход | Класс `negative` / `neutral` / `positive` и вероятности |
| Сеть | LSTM (двунаправленная, один слой) |
| Метрика | F1-macro |
| Цель | F1-macro ≥ 0.80 |

## Структура

```text
app/                  страница Streamlit и API
configs/              обучение, модель, Optuna
data/                 raw → interim → processed и data/rule
docker/               Dockerfile
models/               DVC-копии двух чекпоинтов
notebooks/            EDA и разбор ошибок
src/data/             сборка, разметка, словарь
src/models/           BiLSTM
src/                  обучение, Optuna, инференс
tests/
```

Каталог `data/`, чекпоинты `*.pt` и `optuna.db` в git не входят. Указатели: `data.dvc`, `models/corpus.pt.dvc`, `models/rule.pt.dvc`. Файлы лежат в локальном remote `storage/`. Папку `storage/` нужно скопировать рядом с репозиторием, затем:

```powershell
.\.venv\Scripts\dvc pull
```

## Установка

Нужен Python 3.11. Зависимости ставятся только в `.venv` или в образ Docker.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e ".[dev,app,hpo,data]"
.\.venv\Scripts\pre-commit install
.\.venv\Scripts\dvc pull
```

Архив Navec в репозиторий не входит. Его кладут по пути из `configs/model.yaml`:

`data/external/navec_hudlit_v1_12B_500K_300d_100q.tar`

Проверка кода:

```powershell
.\.venv\Scripts\ruff check .
.\.venv\Scripts\black --check .
.\.venv\Scripts\isort --check-only .
.\.venv\Scripts\pytest
```

## Данные

Сырая выборка — корпус [Russian Sentiment Dataset](https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset). В проект вошли восемь сфер: одежда, другие товары, продукты, места, фильмы, аниме, банки и банковское приложение. Новости и посты соцсетей отброшены. Метки файла: `0` нейтрально, `1` позитив, `2` негатив.

Из каждой пары сфера-класс сначала взято 200 текстов, seed 42. Train затем добран до 100 000: короткая ячейка не останавливает остальные. Val и test не менялись: по 474 отзыва. Вместе 100 948. Словарь 139 038 слов собран только по train. В окно попадают первые 200 токенов.

```powershell
.\.venv\Scripts\python -m src.data.collect --zip $env:USERPROFILE\Downloads\russian-sentiment-dataset.zip
.\.venv\Scripts\python -m src.data.prepare
.\.venv\Scripts\python -m src.data.validate
```

Вторая выборка — те же тексты банков, мест и фильмов, но метка ставится правилом: больше претензии — негатив, больше похвалы — позитив, нет оценки или ничья — нейтрально. Train 31 491, val и test по 3 932.

```powershell
.\.venv\Scripts\python -m src.data.rule_label
```

Ручная проверка 50 отзывов: `labeling/` и `notebooks/01_eda.ipynb`. Label Studio:

```powershell
docker compose up label-studio
.\.venv\Scripts\python -m src.data.labeling
```

Страница: http://localhost:8080.

## Обучение

Команда корпуса читает `configs/train.yaml` и `configs/model.yaml` и **перезаписывает** `checkpoints/best.pt`. Уже обученная модель корпуса там и лежит: эпоха 7, val F1-macro 0.722. Повторный запуск сотрёт этот файл.

```powershell
.\.venv\Scripts\python -m src.train
```

Вторая сеть пишет только в `checkpoints/rule/` и корпус не трогает. На этой выборке включены веса классов.

```powershell
.\.venv\Scripts\python -m src.train_rule
```

Готовые веса: `checkpoints/best.pt` и `checkpoints/rule/best.pt`. Если чекпоинта нет, инференс берёт копию из `models/corpus.pt` или `models/rule.pt`.

В Docker обучение — та же команда корпуса:

```powershell
docker compose run --rm train
```

GPU, если в Docker есть NVIDIA:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm train
```

## Подбор гиперпараметров

Поиск уже сделан: 20 проб, study `ru-review-sentiment-lstm`, файл `optuna.db`. Искались скорость обучения, dropout, размер скрытого состояния и размер батча. Лучшая проба — 13, val F1-macro 0.661 (скорость около 0.00252, dropout 0.172, hidden 128, batch 16). Чекпоинт корпуса этот поиск не заменял.

Новый запуск дописывает ещё 20 проб в ту же study, а не начинает с нуля:

```powershell
.\.venv\Scripts\python -m src.optimize
.\.venv\Scripts\optuna-dashboard sqlite:///optuna.db --port 8081
```

`docker compose up optuna` открывает другую, пустую study в `logs/optuna.db`. Двадцать проб смотреть нужно локальной командой выше. Страница: http://localhost:8081.

## Интерфейс

На странице две модели. `corpus` — восемь сфер и исходные метки. `rule` — банки, места и фильмы по правилу претензия/похвала. У каждой версии своя подпись с пределом.

```powershell
.\.venv\Scripts\python -m streamlit run app/main.py
.\.venv\Scripts\python -m uvicorn app.api:app --port 8000
```

Страница: http://localhost:8501. API и Swagger: http://localhost:8000/docs. Запрос `POST /predict` принимает `text` и `version` (`corpus` или `rule`). Пустой текст отвечает 422.

Можно вставить текст, выбрать пример или загрузить `.txt` и `.csv` с колонкой `text`.

## Docker

Образ ставит CPU-сборку PyTorch и зависимости `dev`, `app`, `hpo`. Каталог проекта на пути импорта задан через `PYTHONPATH=/app`.

```powershell
docker compose run --rm checks
docker compose up app
docker compose up api
```

Интерфейс: http://localhost:8501. API: http://localhost:8000/docs. В контейнер смонтированы `data`, `checkpoints` и `models`.

## Метрики

Точка сравнения до LSTM. Частотный класс всегда отвечает `positive`. Логистическая регрессия смотрит мешок слов по `token_ids` и учится только на train из 100 000. Числа ниже — test, seed 42.

| Модель | Accuracy | F1-macro |
|---|---:|---:|
| Частотный класс | 0.331 | 0.166 |
| Логистическая регрессия | 0.656 | 0.653 |

По классам регрессия на test: негатив 0.684, нейтрально 0.519, позитив 0.756. На val её F1-macro равен 0.683.

| Сеть | Где считали | F1-macro |
|---|---|---:|
| BiLSTM, исходные метки, 100 000 отзывов | val, 474 текста | 0.722 |
| BiLSTM на 3840 отзывах | test, один проход | 0.679 |
| BiLSTM, правило претензия/похвала | val и test, по 3932 | 0.829 |

Test модели на 100 000 отзывах заново не открывался. Строка 0.679 — прежний чекпоинт. 0.829 — согласие с правилом, не с исходными метками Kaggle. Подробности: `reports/baseline.json`, `reports/target_metric.md`, `reports/rule_metric.json`.

```powershell
.\.venv\Scripts\python -m src.baseline
```

## Ограничения

Модель оценивает текст отзыва, а не человека. Её нельзя использовать для кредита, найма и любого решения о человеке.

На исходных метках цель 0.80 не взята. Нейтральный класс слабый, метки корпуса шумные: в пилоте человек совпал с меткой в 31 случае из 50.

Версия `rule` не видела одежду, товары, продукты, аниме и приложение банка. Незнакомые слова сферы становятся неизвестными. Цифру 0.829 нельзя переносить на эти сферы.

Отзыв длиннее 200 токенов обрезается с начала. Новости, комментарии и посты соцсетей в обучение не входили.
