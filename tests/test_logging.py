import logging

from src.utils import get_logger


def test_get_logger_writes_to_one_handler() -> None:
    logger = get_logger("ru-review-test")

    assert logger.name == "ru-review-test"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1

    same_logger = get_logger("ru-review-test")
    assert same_logger is logger
    assert len(same_logger.handlers) == 1
