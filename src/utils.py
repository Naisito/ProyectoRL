# src/utils.py
import os
import time
import random
import numpy as np
import torch
import logging
import json
from typing import Dict, Any


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dirs(root: str, run_name: str) -> dict:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(root, run_name, stamp)
    paths = {
        "base": base,
        "models": os.path.join(base, "models"),
        "logs": os.path.join(base, "logs"),
        "videos": os.path.join(base, "videos"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def configure_logging(log_dir: str, run_name: str, level: int = logging.INFO) -> None:
    """Configura logging a stdout y fichero dentro de log_dir.

    Args:
        log_dir: carpeta donde crear el archivo de log.
        run_name: prefijo del archivo de log.
        level: nivel de logging.
    """
    os.makedirs(log_dir, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logger = logging.getLogger()
    logger.setLevel(level)

    # Evitar múltiples handlers si se llama repetidamente
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(sh)

    log_file = os.path.join(log_dir, f"{run_name}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
