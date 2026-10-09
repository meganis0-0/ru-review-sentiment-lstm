from pathlib import Path

import pytest

from src.train import prepare_runtime


def test_prepare_runtime_creates_directories(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "train.yaml").write_text("experiment_name: test\n", encoding="utf-8")

    config_path = prepare_runtime(tmp_path)

    assert config_path == tmp_path / "configs" / "train.yaml"
    assert (tmp_path / "checkpoints").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_prepare_runtime_fails_without_config(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        prepare_runtime(tmp_path)
