# src/train.py
import os
import argparse
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import NatureCNN

from src.config import load_train_config, TrainConfig
from src.utils import (
    set_global_seeds,
    make_run_dirs,
    configure_logging,
    write_json,
    get_device,
)
from src.envs import make_vec_env
from src.callbacks import build_callbacks


def main():
    parser = argparse.ArgumentParser(description="Entrena PPO en CarRacing con config centralizada")
    parser.add_argument("--config", type=str, default=None, help="Ruta a YAML/JSON con hiperparámetros")
    parser.add_argument("--n-envs", type=int, default=None, help="Número de entornos vectorizados (sobrescribe config)")
    parser.add_argument("--device", type=str, default=None, help="Forzar dispositivo (cuda/cpu)")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Sobrescribe total_timesteps del config")
    parser.add_argument("--visual", action="store_true", help="Muestra render y frames preprocesados en tiempo real (n_envs forzado a 1)")
    parser.add_argument("--render-freq", type=int, default=1, help="Frecuencia (en pasos) para actualizar la ventana visual (reduce overhead)")
    parser.add_argument("--visual-only-preproc", action="store_true", help="Mostrar solo la imagen preprocesada (no el render raw), reduce overhead")
    parser.add_argument("--visual-scale", type=int, default=4, help="Escala de visualización para la imagen preprocesada (p.ej. 4 -> 336x336)")
    parser.add_argument("--num-threads", type=int, default=None, help="Número de hilos OpenMP/torch a usar (p.ej. para BLAS); por defecto usa cpu_count())")
    args = parser.parse_args()

    overrides = {}
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps

    cfg = load_train_config(args.config, overrides=overrides)

    # Seeds y device
    set_global_seeds(cfg.seed)
    device = args.device if args.device else get_device()
    # Controlar número de hilos para operaciones BLAS/torch (reduce sobreuso de CPU)
    import torch as _torch
    if args.num_threads is not None:
        _torch.set_num_threads(max(1, int(args.num_threads)))
    else:
        # limitar a la cantidad de CPUs disponibles por defecto
        try:
            import os as _os
            ncpu = _os.cpu_count() or 1
            _torch.set_num_threads(ncpu)
        except Exception:
            pass

    paths = make_run_dirs(cfg.log_root, cfg.run_name)
    print(f"📁 Run dir: {paths['base']}")

    # Logging
    configure_logging(paths["logs"], cfg.run_name)
    logging.info("Inicio del run")
    logging.info(f"Device: {device}")

    # Guardar configuración para reproducibilidad
    try:
        write_json(os.path.join(paths["base"], "config.json"), cfg.__dict__)
    except Exception:
        logging.exception("No se pudo guardar config.json")

    n_envs = getattr(cfg, "n_envs", 4)
    if args.n_envs is not None:
        n_envs = args.n_envs
    if args.visual:
        n_envs = 1

    # Entrenamiento: ajusta n_envs según hardware
    train_env = make_vec_env(
        env_id=cfg.env_id,
        seed=cfg.seed,
        n_envs=n_envs,
        grayscale=cfg.grayscale,
        resize=cfg.resize,
        frame_skip=cfg.frame_skip,
        frames_stack=cfg.frames_stack,
        reward_clip=False,
        render_mode="rgb_array" if args.visual else None,
    )

    eval_env = make_vec_env(
        env_id=cfg.env_id,
        seed=cfg.seed + 10_000,
        n_envs=1,
        grayscale=cfg.grayscale,
        resize=cfg.resize,
        frame_skip=cfg.frame_skip,
        frames_stack=cfg.frames_stack,
        reward_clip=False,
        render_mode="rgb_array" if args.visual else None,
    )

    callbacks = build_callbacks(
        eval_env=eval_env,
        model_dir=paths["models"],
        log_dir=paths["logs"],
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        save_freq=cfg.save_freq,
        visual={
            "render_freq": max(1, int(args.render_freq)),
            "grayscale": cfg.grayscale,
            "resize": cfg.resize,
            "show_raw": not args.visual_only_preproc,
            "proc_scale": max(1, int(args.visual_scale)),
        } if args.visual else None,
    )

    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # Si las observaciones ya están normalizadas (float32 en [0,1]) y en canales-first,
    # puede ser necesario desactivar la normalización automática de la policy.
    # SB3 espera imágenes en un formato concreto; si ves errores sobre NatureCNN, prueba
    # a cambiar normalize_images a False (se pasa en policy_kwargs).
    policy_kwargs["normalize_images"] = False

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        verbose=1,
        tensorboard_log=paths["logs"],
        policy_kwargs=policy_kwargs,
        seed=cfg.seed,
        device=device,
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        tb_log_name=cfg.run_name,
        progress_bar=True,
    )

    final_path = os.path.join(paths["models"], "final_model.zip")
    model.save(final_path)
    logging.info(f"Modelo final guardado en: {final_path}")
    logging.info(f"TensorBoard: tensorboard --logdir {cfg.log_root}")


if __name__ == "__main__":
    main()
