from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "README.md",
    "DATASET_CARD.md",
    "pyproject.toml",
    "configs/train.yaml",
    "configs/model.yaml",
    "configs/optuna.yaml",
    "app",
    "data.dvc",
    "docker/Dockerfile",
    "docker-compose.yml",
    "notebooks/01_eda.ipynb",
    "notebooks/03_error_analysis.ipynb",
    "src/data/dataset.py",
    "src/models/rnn.py",
    "tests",
    ".github/workflows/ci.yml",
    ".github/ISSUE_TEMPLATE/task.md",
    ".github/pull_request_template.md",
)


def test_project_layout_exists() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    assert missing == []


def test_target_metric_is_recorded() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "F1-macro" in text
    assert "0.80" in text
    assert "LSTM" in text


def test_error_analysis_notebook_reads_the_held_out_split() -> None:
    text = (ROOT / "notebooks" / "03_error_analysis.ipynb").read_text(encoding="utf-8")
    for fragment in ("test.csv", "confusion_matrix", "нейтраль", "F1-macro"):
        assert fragment in text


def test_dataset_card_covers_composition_and_license() -> None:
    text = (ROOT / "DATASET_CARD.md").read_text(encoding="utf-8")
    for fragment in ("4788", "negative", "neutral", "positive", "MIT", "Apache-2.0", "31"):
        assert fragment in text
