from pathlib import Path

import pytest

from src.optimize import create_study, load_optuna_config, storage_url

ROOT = Path(__file__).resolve().parents[1]


def test_optuna_yaml_asks_for_a_sqlite_study() -> None:
    config = load_optuna_config(ROOT / "configs" / "optuna.yaml")

    assert config.n_trials_min <= config.n_trials <= config.n_trials_max
    assert config.n_trials == 20
    assert config.storage == "sqlite:///optuna.db"
    assert config.pruner_name == "median"
    assert config.metric == "f1_macro"
    assert config.search.hidden_sizes == (64, 128, 256)
    assert config.search.batch_sizes == (16, 32, 64)


def test_storage_url_is_absolute(tmp_path: Path) -> None:
    url = storage_url(tmp_path, "sqlite:///optuna.db")

    assert url.startswith("sqlite:///")
    assert url.endswith("/optuna.db")
    assert "sqlite:///optuna.db" != url


def test_load_optuna_config_rejects_an_unknown_pruner(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "optuna.yaml").read_text(encoding="utf-8")
    path = tmp_path / "optuna.yaml"
    path.write_text(source.replace("name: median", "name: none"), encoding="utf-8")

    with pytest.raises(ValueError, match="pruner"):
        load_optuna_config(path)


def test_study_keeps_one_trial_in_sqlite(tmp_path: Path) -> None:
    optuna = pytest.importorskip("optuna")
    config = load_optuna_config(ROOT / "configs" / "optuna.yaml")
    url = storage_url(tmp_path, "sqlite:///optuna.db")
    study = create_study(config, url)

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        return 0.5

    study.optimize(objective, n_trials=1)

    assert isinstance(study.pruner, optuna.pruners.MedianPruner)
    assert (tmp_path / "optuna.db").is_file()
    assert len(study.trials) == 1
    assert study.best_value == 0.5
