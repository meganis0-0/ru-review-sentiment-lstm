"""Точка входа обучения.

Сам цикл обучения подключается отдельными задачами. Сейчас команда готовит
каталоги и проверяет, что конфиг на месте.
"""

from pathlib import Path

from src.utils import get_logger


def prepare_runtime(root: Path) -> Path:
    """Создаёт каталоги артефактов и проверяет конфиг обучения."""
    logger = get_logger("train")
    config_path = root / "configs" / "train.yaml"
    if not config_path.is_file():
        logger.error("Не найден конфиг %s", config_path)
        raise SystemExit(1)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logger.info("Конфиг обучения: %s", config_path)
    logger.info("Каталоги checkpoints и logs готовы. Цикл обучения ещё не подключён.")
    return config_path


def main() -> None:
    prepare_runtime(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
