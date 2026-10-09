from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "README.md",
    "pyproject.toml",
    "configs/train.yaml",
    "configs/model.yaml",
    "configs/optuna.yaml",
    "app",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/external",
    "docker/Dockerfile",
    "docker-compose.yml",
    "notebooks",
    "src/data",
    "src/models",
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
