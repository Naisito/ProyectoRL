# src/eval_live.py
"""Evaluación en tiempo real (sin grabación).

Este script carga un modelo PPO entrenado y ejecuta episodios renderizando el entorno
en una ventana en tiempo real. Está pensado para inspección cualitativa rápida.

Ejemplo:
    python -m src.eval_live --model results/ppo_carracing/<run>/models/best_model.zip --episodes 3

Notas:
- No guarda vídeos.
- Si quieres ver más lento/rápido, ajusta --fps.
"""

from __future__ import annotations

import argparse
import time
import logging

import numpy as np
from stable_baselines3 import PPO

from src.config import EvalConfig, load_eval_config
from src.utils import set_global_seeds, configure_logging, make_run_dirs


def _infer_resize_from_model(model: PPO) -> int | None:
    """Intenta inferir el tamaño H=W esperado por la política.

    El wrapper `PreprocessCarRacing` hace resize cuadrado. Si el modelo fue
    entrenado con (C, 96, 96) pero el eval crea (C, 84, 84), SB3 lanza un
    ValueError. Aquí lo detectamos para ajustar el eval sin tocar config.
    """

    try:
        obs_space = model.policy.observation_space
    except Exception:
        return None

    # Casos típicos:
    # - Box(low, high, shape=(C, H, W))
    # - Vec env: Box(..., shape=(n_env, C, H, W)) (menos típico en policy)
    shape = getattr(obs_space, "shape", None)
    if not shape:
        return None

    if len(shape) == 3:
        _, h, w = shape
    elif len(shape) == 4:
        _, _, h, w = shape
    else:
        return None

    if isinstance(h, int) and isinstance(w, int) and h == w:
        return int(h)
    return None


def make_live_env(cfg: EvalConfig):
    """Crea un entorno de evaluación con render humano y el mismo preprocesado."""
    # Import local para evitar dependencias circulares y mantener script ligero
    from src.envs import make_eval_env

    # Para render en vivo: render_mode="human" y sin video_dir
    env = make_eval_env(
        env_id=cfg.env_id,
        seed=cfg.seed,
        grayscale=cfg.grayscale,
        resize=cfg.resize,
        frame_skip=cfg.frame_skip,
        frames_stack=cfg.frames_stack,
        video_dir=None,
    )

    # gymnasium render_mode se configura al crear el env; make_eval_env actualmente usa
    # render_mode=None si no hay vídeo. Para forzar ventana en vivo, recreamos el env base
    # con render_mode="human" y aplicamos exactamente los mismos wrappers.
    #
    # La forma más robusta aquí es crear el env desde make_env_fn con render_mode="human"
    # y luego aplicar el mismo frame-stack de evaluación.
    from src.envs import FrameSkip, PreprocessCarRacing, _GymFrameStackToCHW
    import gymnasium as gym

    base = gym.make(cfg.env_id, render_mode="human")
    base.reset(seed=cfg.seed)
    base = FrameSkip(base, skip=cfg.frame_skip)
    base = PreprocessCarRacing(base, grayscale=cfg.grayscale, resize=cfg.resize)

    frame_stack_cls = getattr(gym.wrappers, "FrameStackObservation", None) or getattr(gym.wrappers, "FrameStack", None)
    if frame_stack_cls is None:
        raise RuntimeError("No FrameStack wrapper found in gym.wrappers")

    try:
        base = frame_stack_cls(base, num_stack=cfg.frames_stack)
    except TypeError:
        try:
            base = frame_stack_cls(base, cfg.frames_stack)
        except Exception:
            base = frame_stack_cls(base, n_stack=cfg.frames_stack)

    base = _GymFrameStackToCHW(base, channels_first=True)

    # Cerramos el env que se creó arriba (sin render) y devolvemos el nuevo
    try:
        env.close()
    except Exception:
        pass

    return base


def evaluate_live(model_path: str, cfg: EvalConfig, fps: float, deterministic: bool):
    set_global_seeds(cfg.seed)

    # Logging en results/eval_live/<timestamp>/logs (sin vídeos)
    paths = make_run_dirs("results", "eval_live")
    configure_logging(paths["logs"], "eval_live")

    logging.info("Evaluación en vivo (sin grabación)")
    logging.info("Modelo: %s", model_path)

    model = PPO.load(model_path)

    # Ajuste automático de resize si el modelo espera otra resolución.
    inferred_resize = _infer_resize_from_model(model)
    if inferred_resize is not None and inferred_resize != cfg.resize:
        logging.info(
            "Ajustando resize de eval de %s a %s para coincidir con el modelo",
            cfg.resize,
            inferred_resize,
        )
        cfg = EvalConfig(**{**cfg.__dict__, "resize": int(inferred_resize)})

    env = make_live_env(cfg)

    rewards: list[float] = []
    lengths: list[int] = []

    dt = 0.0 if fps <= 0 else 1.0 / fps

    for ep in range(cfg.n_episodes):
        obs, info = env.reset(seed=cfg.seed + ep)
        terminated = truncated = False
        ep_r = 0.0
        steps = 0

        # Render primer frame
        try:
            env.render()
        except Exception:
            pass

        while not (terminated or truncated):
            t0 = time.perf_counter()

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)

            ep_r += float(r)
            steps += 1

            # Render y control de FPS
            try:
                env.render()
            except Exception:
                pass

            if dt > 0:
                elapsed = time.perf_counter() - t0
                sleep_for = dt - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        rewards.append(ep_r)
        lengths.append(steps)

        print(f"Episodio {ep+1}/{cfg.n_episodes} | reward={ep_r:.1f} | len={steps}")
        logging.info("Episodio %s/%s | reward=%.3f | len=%s", ep + 1, cfg.n_episodes, ep_r, steps)

    env.close()

    print("\n===== Evaluación en vivo (resumen) =====")
    print(f"Modelo: {model_path}")
    print(f"Episodios: {cfg.n_episodes}")
    print(f"Reward media: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Longitud media: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluación en tiempo real (render) de un modelo PPO")
    parser.add_argument("--model", type=str, required=True, help="Ruta a best_model.zip o final_model.zip")
    parser.add_argument("--config", type=str, default=None, help="Ruta a YAML/JSON con parámetros de evaluación")
    parser.add_argument("--episodes", type=int, default=None, help="Sobrescribir número de episodios")
    parser.add_argument("--seed", type=int, default=None, help="Sobrescribir seed de evaluación")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS objetivo (0 para sin límite)")
    parser.add_argument("--stochastic", action="store_true", help="Usar política estocástica (no determinista)")
    args = parser.parse_args()

    cfg = load_eval_config(args.config) if args.config else EvalConfig()
    if args.episodes is not None:
        cfg = EvalConfig(**{**cfg.__dict__, "n_episodes": int(args.episodes)})
    if args.seed is not None:
        cfg = EvalConfig(**{**cfg.__dict__, "seed": int(args.seed)})

    evaluate_live(args.model, cfg, fps=args.fps, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()
