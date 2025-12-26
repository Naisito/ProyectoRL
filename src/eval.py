# src/eval.py
import argparse
import numpy as np
import logging

from stable_baselines3 import PPO

from src.config import load_eval_config, EvalConfig
from src.utils import set_global_seeds, make_run_dirs, configure_logging, write_json
from src.envs import make_eval_env


def evaluate(model_path: str, cfg: EvalConfig):
    set_global_seeds(cfg.seed)

    # Guardamos vídeos en una carpeta nueva (independiente del entrenamiento)
    paths = make_run_dirs("results", "eval")
    video_dir = paths["videos"]

    configure_logging(paths["logs"], "eval")
    logging.info(f"Evaluando modelo: {model_path}")

    env = make_eval_env(
        env_id=cfg.env_id,
        seed=cfg.seed,
        grayscale=cfg.grayscale,
        resize=cfg.resize,
        frame_skip=cfg.frame_skip,
        frames_stack=cfg.frames_stack,
        video_dir=video_dir,
        record_trigger_episodes=cfg.record_episodes,
    )

    model = PPO.load(model_path)

    rewards = []
    lengths = []

    for ep in range(cfg.n_episodes):
        obs, info = env.reset(seed=cfg.seed + ep)
        done = False
        ep_r = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r += float(r)
            steps += 1

        rewards.append(ep_r)
        lengths.append(steps)

    env.close()

    print("\n===== Evaluación CarRacing =====")
    print(f"Modelo: {model_path}")
    print(f"Episodios: {cfg.n_episodes}")
    print(f"Reward media: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Longitud media: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"🎥 Vídeos: {video_dir}")
    logging.info(f"Vídeos guardados en: {video_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ruta a best_model.zip o final_model.zip")
    parser.add_argument("--config", type=str, default=None, help="Ruta a YAML/JSON con parámetros de evaluación")
    args = parser.parse_args()

    cfg = load_eval_config(args.config) if args.config else EvalConfig()
    # Guardar config de eval para trazabilidad
    try:
        write_json("results/eval_config.json", cfg.__dict__)
    except Exception:
        logging.exception("No se pudo guardar config de evaluación")

    evaluate(args.model, cfg)


if __name__ == "__main__":
    main()
