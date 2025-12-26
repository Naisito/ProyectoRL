# src/config.py
from dataclasses import dataclass
from typing import Any, Dict
import json

try:
    import yaml  # optional, listed in requirements
except Exception:
    yaml = None


@dataclass(frozen=True)
class TrainConfig:
    # Entorno
    env_id: str = "CarRacing-v3"
    frames_stack: int = 4
    grayscale: bool = True
    resize: int = 84
    frame_skip: int = 2
    # "domain randomization" ligero: variar semilla + no usar info externa
    seed: int = 42

    # PPO
    total_timesteps: int = 2_000_000
    learning_rate: float = 2.5e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10

    # Logging / guardado
    run_name: str = "ppo_carracing"
    log_root: str = "results"
    eval_freq: int = 50_000
    n_eval_episodes: int = 10
    save_freq: int = 100_000

    # Render
    deterministic_eval: bool = True


@dataclass(frozen=True)
class EvalConfig:
    env_id: str = "CarRacing-v3"
    frames_stack: int = 4
    grayscale: bool = True
    resize: int = 84
    frame_skip: int = 2
    seed: int = 123
    n_episodes: int = 20
    record_episodes: int = 5


def _dict_to_dataclass(datacls, data: Dict[str, Any]):
    """Construye el dataclass ignorando claves desconocidas."""
    valid_keys = {f.name for f in datacls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return datacls(**filtered)


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML no está instalado. Añade 'pyyaml' a requirements.txt")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_config(path: str | None = None, overrides: Dict[str, Any] | None = None) -> TrainConfig:
    data: Dict[str, Any] = {}
    if path:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = load_yaml(path)
    if overrides:
        data.update(overrides)
    return _dict_to_dataclass(TrainConfig, data)


def load_eval_config(path: str | None = None, overrides: Dict[str, Any] | None = None) -> EvalConfig:
    data: Dict[str, Any] = {}
    if path:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = load_yaml(path)
    if overrides:
        data.update(overrides)
    return _dict_to_dataclass(EvalConfig, data)
