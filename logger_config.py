# logger_config.py
import logging
from pathlib import Path

def setup_logger(
    name="main",
    log_file="DCLens_Gamma.log",
    level=logging.INFO,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # 防止 notebook 里重复添加

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s]: %(message)s"
    )

    # File
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(formatter)

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False  # 🔥 关键

    return logger
