# src/train.py
import os
import argparse
import logging
import json

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import NatureCNN

from src.config import load_train_config
from src.utils import (
    set_global_seeds,
    make_run_dirs,
    configure_logging,
    write_json,
    get_device,
)
from src.envs_CL import make_vec_env
from src.callbacks import build_callbacks


def main():
    parser = argparse.ArgumentParser(description="Entrena PPO en CarRacing con curriculum learning")
    parser.add_argument("--config", type=str, default=None, help="Ruta a YAML/JSON con hiperparámetros")
    parser.add_argument("--n-envs", type=int, default=None, help="Número de entornos vectorizados")
    parser.add_argument("--device", type=str, default=None, help="Forzar dispositivo (cuda/cpu)")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Sobrescribe total_timesteps")
    parser.add_argument("--visual", action="store_true", help="Modo visual (n_envs=1)")
    parser.add_argument("--render-freq", type=int, default=1)
    parser.add_argument("--visual-only-preproc", action="store_true")
    parser.add_argument("--visual-scale", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument(
        "--curriculum-seeds",
        type=str,
        default="carracing_curriculum_seeds.json",
        help="JSON con seeds del curriculum (easy/medium/hard)",
    )
    args = parser.parse_args()

    overrides = {}
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps

    cfg = load_train_config(args.config, overrides=overrides)

    # ------------------------------------------------------------------
    # Seeds y device
    # ------------------------------------------------------------------
    set_global_seeds(cfg.seed)
    device = args.device if args.device else get_device()

    import torch as _torch
    if args.num_threads is not None:
        _torch.set_num_threads(max(1, int(args.num_threads)))
    else:
        try:
            _torch.set_num_threads(os.cpu_count() or 1)
        except Exception:
            pass

    paths = make_run_dirs(cfg.log_root, cfg.run_name)
    print(f"📁 Run dir: {paths['base']}")

    configure_logging(paths["logs"], cfg.run_name)
    logging.info("Inicio del entrenamiento")
    logging.info(f"Device: {device}")

    try:
        write_json(os.path.join(paths["base"], "config.json"), cfg.__dict__)
    except Exception:
        logging.exception("No se pudo guardar config.json")

    # ------------------------------------------------------------------
    # Número de entornos
    # ------------------------------------------------------------------
    n_envs = getattr(cfg, "n_envs", 4)
    if args.n_envs is not None:
        n_envs = args.n_envs
    if args.visual:
        n_envs = 1

    # ------------------------------------------------------------------
    # 🔥 Cargar curriculum learning seeds
    # ------------------------------------------------------------------
    if not os.path.exists(args.curriculum_seeds):
        curriculum_seeds = generate_default_curriculum_seeds(args.curriculum_seeds)
    else:
        with open(args.curriculum_seeds, "r") as f:
            curriculum_seeds = json.load(f)


    curriculum = [
        ("easy",   curriculum_seeds["easy"],   int(cfg.total_timesteps * 0.25)),
        ("medium", curriculum_seeds["medium"], int(cfg.total_timesteps * 0.35)),
        ("hard",   curriculum_seeds["hard"],   cfg.total_timesteps),
    ]

    # ------------------------------------------------------------------
    # Entorno de evaluación (SIEMPRE circuito completo)
    # ------------------------------------------------------------------
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
        curriculum_seeds=curriculum_seeds["hard"],
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

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=512),
        normalize_images=False,
    )

    model = None
    trained_steps = 0

    # ------------------------------------------------------------------
    # 🔥 Curriculum training loop
    # ------------------------------------------------------------------
    for phase_name, phase_seeds, phase_steps in curriculum:
        logging.info(f"🚦 Curriculum phase: {phase_name}")

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
            curriculum_seeds=phase_seeds,
        )

        if model is None:
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
        else:
            model.set_env(train_env)

        model.learn(
            total_timesteps=phase_steps - trained_steps,
            callback=callbacks,
            tb_log_name=f"{cfg.run_name}_{phase_name}",
            reset_num_timesteps=False,
            progress_bar=True,
        )

        trained_steps = phase_steps

    # ------------------------------------------------------------------
    # Guardar modelo final
    # ------------------------------------------------------------------
    final_path = os.path.join(paths["models"], "final_model.zip")
    model.save(final_path)
    logging.info(f"Modelo final guardado en: {final_path}")
    logging.info(f"TensorBoard: tensorboard --logdir {cfg.log_root}")

def generate_default_curriculum_seeds(path: str):
    """
    Fallback simple si no existe el fichero de curriculum.
    Usa rangos de seeds razonables.
    """
    logging.warning("⚠️ No se encontró el fichero de curriculum seeds.")
    logging.warning("➡️ Generando curriculum por defecto.")

    curriculum = {
        "easy":   list(range(0, 50)),
        "medium": list(range(200, 300)),
        "hard":   list(range(600, 750)),
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        import json
        json.dump(curriculum, f, indent=2)

    logging.info(f"✅ Curriculum seeds generado en: {path}")
    return curriculum

if __name__ == "__main__":
    main()
