"""Подбор гиперпараметров LSTM в Optuna: пробы, pruner и SQLite."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from src.data.dataset import make_loader
from src.models.config import load_model_config
from src.models.embeddings import vectors_from_navec
from src.train import build_model, load_train_config, run_training, set_seed
from src.utils import get_logger


@dataclass(frozen=True)
class SearchSpace:
    learning_rate_low: float
    learning_rate_high: float
    learning_rate_log: bool
    dropout_low: float
    dropout_high: float
    hidden_sizes: tuple[int, ...]
    batch_sizes: tuple[int, ...]


@dataclass(frozen=True)
class OptunaConfig:
    n_trials: int
    n_trials_min: int
    n_trials_max: int
    storage: str
    study_name: str
    direction: str
    metric: str
    epochs: int
    seed: int
    pruner_name: str
    pruner_startup: int
    pruner_warmup: int
    search: SearchSpace


def load_optuna_config(path: Path) -> OptunaConfig:
    """Читает число проб, pruner и границы поиска."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"В {path} ожидается словарь параметров")
    pruner = payload.get("pruner")
    space = payload.get("search_space")
    if not isinstance(pruner, dict) or not isinstance(space, dict):
        raise ValueError(f"В {path} нет блока pruner или search_space")
    required = (
        "n_trials",
        "n_trials_min",
        "n_trials_max",
        "storage",
        "study_name",
        "direction",
        "metric",
        "epochs",
        "seed",
    )
    missing = [key for key in required if key not in payload]
    if "name" not in pruner:
        missing.append("pruner.name")
    if "n_startup_trials" not in pruner:
        missing.append("pruner.n_startup_trials")
    if "n_warmup_steps" not in pruner:
        missing.append("pruner.n_warmup_steps")
    learning_rate = space.get("learning_rate")
    dropout = space.get("dropout")
    hidden = space.get("hidden_size")
    batch = space.get("batch_size")
    if (
        not isinstance(learning_rate, dict)
        or "low" not in learning_rate
        or "high" not in learning_rate
    ):
        missing.append("search_space.learning_rate")
    if not isinstance(dropout, dict) or "low" not in dropout or "high" not in dropout:
        missing.append("search_space.dropout")
    if not isinstance(hidden, dict) or not hidden.get("choices"):
        missing.append("search_space.hidden_size")
    if not isinstance(batch, dict) or not batch.get("choices"):
        missing.append("search_space.batch_size")
    if missing:
        raise ValueError(f"В {path} не хватает полей: {', '.join(missing)}")
    config = OptunaConfig(
        n_trials=int(payload["n_trials"]),
        n_trials_min=int(payload["n_trials_min"]),
        n_trials_max=int(payload["n_trials_max"]),
        storage=str(payload["storage"]),
        study_name=str(payload["study_name"]),
        direction=str(payload["direction"]),
        metric=str(payload["metric"]),
        epochs=int(payload["epochs"]),
        seed=int(payload["seed"]),
        pruner_name=str(pruner["name"]),
        pruner_startup=int(pruner["n_startup_trials"]),
        pruner_warmup=int(pruner["n_warmup_steps"]),
        search=SearchSpace(
            learning_rate_low=float(learning_rate["low"]),
            learning_rate_high=float(learning_rate["high"]),
            learning_rate_log=bool(learning_rate.get("log", False)),
            dropout_low=float(dropout["low"]),
            dropout_high=float(dropout["high"]),
            hidden_sizes=tuple(int(item) for item in hidden["choices"]),
            batch_sizes=tuple(int(item) for item in batch["choices"]),
        ),
    )
    if not config.n_trials_min <= config.n_trials <= config.n_trials_max:
        raise ValueError(
            f"n_trials {config.n_trials} вне диапазона "
            f"{config.n_trials_min}–{config.n_trials_max}"
        )
    if config.pruner_name != "median":
        raise ValueError(f"Неизвестный pruner: {config.pruner_name}")
    if config.direction != "maximize":
        raise ValueError(f"Метрика максимизируется, в конфиге {config.direction}")
    return config


def storage_url(root: Path, storage: str) -> str:
    """Делает путь к SQLite абсолютным, чтобы база лежала в корне репозитория."""
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        raise ValueError(f"Ожидается {prefix}, получено {storage}")
    path = (root / storage.removeprefix(prefix)).resolve()
    return prefix + path.as_posix()


def create_study(config: OptunaConfig, storage: str):
    """Study на максимум F1-macro. MedianPruner обрывает пробы хуже медианы."""
    import optuna

    return optuna.create_study(
        study_name=config.study_name,
        storage=storage,
        direction=config.direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.pruner_startup,
            n_warmup_steps=config.pruner_warmup,
        ),
    )


def main() -> None:
    import optuna

    root = Path(__file__).resolve().parents[1]
    logger = get_logger("optimize")
    config = load_optuna_config(root / "configs" / "optuna.yaml")
    train_config = load_train_config(root / "configs" / "train.yaml")
    model_config = load_model_config(root / "configs" / "model.yaml")
    train_path = root / "data" / "processed" / "train.csv"
    val_path = root / "data" / "processed" / "val.csv"
    vocab_path = root / "data" / "processed" / "vocab.json"
    if not train_path.is_file() or not val_path.is_file() or not vocab_path.is_file():
        raise SystemExit("Нет data/processed. Сначала соберите выборку.")
    if not model_config.pretrained_embeddings:
        raise SystemExit("В configs/model.yaml нет предобученных эмбеддингов.")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vectors = vectors_from_navec(root / model_config.pretrained_embeddings, vocab)
    url = storage_url(root, config.storage)
    study = create_study(config, url)

    def objective(trial: optuna.Trial) -> float:
        space = config.search
        learning_rate = trial.suggest_float(
            "learning_rate",
            space.learning_rate_low,
            space.learning_rate_high,
            log=space.learning_rate_log,
        )
        dropout = trial.suggest_float("dropout", space.dropout_low, space.dropout_high)
        hidden_size = int(trial.suggest_categorical("hidden_size", list(space.hidden_sizes)))
        batch_size = int(trial.suggest_categorical("batch_size", list(space.batch_sizes)))
        tuned = replace(model_config, dropout=dropout, hidden_size=hidden_size)
        set_seed(config.seed)
        model, model_args = build_model(vocab, tuned, vectors)
        train_loader = make_loader(
            train_path,
            batch_size=batch_size,
            shuffle=True,
            seed=config.seed,
            token_dropout=tuned.token_dropout,
        )
        val_loader = make_loader(val_path, batch_size=batch_size, shuffle=False, seed=config.seed)

        def on_epoch(epoch: int, score: float) -> None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result = run_training(
                train_loader,
                val_loader,
                model,
                epochs=config.epochs,
                learning_rate=learning_rate,
                seed=config.seed,
                checkpoint_dir=folder / "checkpoints",
                log_dir=folder / "logs",
                early_stopping_patience=train_config.early_stopping_patience,
                early_stopping_min_delta=train_config.early_stopping_min_delta,
                model_args=model_args,
                on_epoch=on_epoch,
            )
        return float(result["best_f1_macro"])

    def _remember(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        logger.info("Проба %s: %s, f1-macro %s", trial.number, trial.state.name, trial.value)

    study.optimize(objective, n_trials=config.n_trials, callbacks=[_remember])
    best = study.best_trial
    logger.info("Лучшая проба %s: f1-macro %.3f, %s", best.number, best.value, best.params)
    logger.info("База проб: %s", url)


if __name__ == "__main__":
    main()
