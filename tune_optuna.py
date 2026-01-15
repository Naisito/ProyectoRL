import os
import sys
import json
import random
import optuna
import numpy as np

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
import plotly.io as pio

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.callbacks import BaseCallback

from src.config import load_train_config
from src.envs import make_vec_env
from src.utils import set_global_seeds, get_device


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_batch_size(trial, n_steps: int, n_envs: int) -> int:
    """En PPO conviene batch_size divisor de (n_steps*n_envs)."""
    rollout = n_steps * n_envs
    candidates = [64, 128, 256, 512, 1024]
    valid = [b for b in candidates if b <= rollout and rollout % b == 0]
    if not valid:
        # fallback: mayor posible <= rollout
        valid = [max([b for b in candidates if b <= rollout] or [64])]
    return trial.suggest_categorical("batch_size", valid)


class OptunaEvalPruneCallback(BaseCallback):
    """Evalúa cada eval_freq pasos y permite pruning."""
    def __init__(self, trial, eval_env, eval_freq: int, n_eval_episodes: int = 5, deterministic: bool = True):
        super().__init__()
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
            )
            self.trial.report(float(mean_reward), step=self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True


def build_objective(cfg, n_steps_per_trial: int, n_envs: int):
    """
    Devuelve una función objective(trial) cerrada sobre cfg, etc.
    """
    def objective(trial: optuna.Trial) -> float:
        # --- espacio de búsqueda (puedes ajustarlo) ---
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 1e-5, 5e-2, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        n_epochs = trial.suggest_int("n_epochs", 3, 15)

        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        batch_size = _pick_batch_size(trial, n_steps=n_steps, n_envs=n_envs)

        # seeds reproducibles
        set_global_seeds(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        device = get_device()

        # --- envs (usando tu make_vec_env) ---
        train_env = make_vec_env(
            env_id=cfg.env_id,
            seed=cfg.seed,
            n_envs=n_envs,
            grayscale=cfg.grayscale,
            resize=cfg.resize,
            frame_skip=cfg.frame_skip,
            frames_stack=cfg.frames_stack,
            reward_clip=False,
            render_mode=None,
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
            render_mode=None,
        )

        policy_kwargs = dict(
            features_extractor_class=NatureCNN,
            features_extractor_kwargs=dict(features_dim=512),
            normalize_images=False,  # como en tu train.py
        )

        try:
            model = PPO(
                policy="CnnPolicy",
                env=train_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                verbose=0,
                policy_kwargs=policy_kwargs,
                seed=cfg.seed,
                device=device,
            )

            # pruning “en vivo”
            eval_freq = max(10_000, n_steps_per_trial // 5)
            cb = OptunaEvalPruneCallback(
                trial=trial,
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=max(3, cfg.n_eval_episodes // 2),
            )

            model.learn(total_timesteps=n_steps_per_trial, callback=cb, progress_bar=True)

            # score final
            mean_reward, _ = evaluate_policy(
                model, eval_env, n_eval_episodes=cfg.n_eval_episodes, deterministic=True
            )
            return float(mean_reward)

        finally:
            try:
                train_env.close()
            except Exception:
                pass
            try:
                eval_env.close()
            except Exception:
                pass

    return objective


if __name__ == "__main__":
    """
    Uso:
      python tune_optuna.py configs/entrenamiento1_unai.yaml
    (y opcionalmente puedes sobreescribir con variables de entorno abajo si quieres)
    """
    if len(sys.argv) < 2:
        print("Uso: python tune_optuna.py <config.yaml|json>")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = load_train_config(config_path)

    # --- parámetros de Optuna (puedes moverlos a JSON si quieres, pero así ya funciona) ---
    # Si quieres sobreescribir rápido:
    n_trials = int(os.environ.get("N_TRIALS", "30"))
    n_steps_per_trial = int(os.environ.get("TIMESTEPS_PER_TRIAL", "200000"))
    n_envs = int(os.environ.get("N_ENVS", "4"))
    seed = int(os.environ.get("SEED", str(cfg.seed)))

    # DB / study
    _ensure_dir("optuna")
    storage_file = os.environ.get("OPTUNA_STORAGE", "sqlite:///optuna/optuna_all.db")
    study_name = os.environ.get("OPTUNA_STUDY", "PPO_CarRacing")

    # output
    output_dir = os.environ.get("OPTUNA_OUT", "optuna/ppo_carracing")
    _ensure_dir(output_dir)
    plots_dir = os.path.join(output_dir, "plots")
    _ensure_dir(plots_dir)

    # sampler/pruner
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage_file,
        load_if_exists=True,
    )

    print(f"Buscando mejores hiperparámetros para PPO ({n_trials} trials)...")
    objective = build_objective(cfg, n_steps_per_trial=n_steps_per_trial, n_envs=n_envs)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    # guardar best params
    best_params = json.dumps(study.best_trial.params, indent=4)
    with open(os.path.join(output_dir, "best_trial.json"), "w", encoding="utf-8") as f:
        f.write(best_params)

    print("Mejores hiperparámetros encontrados:")
    print(best_params)

    # plots HTML (como en clase)
    print("Generando gráficos de Optuna...")
    fig = plot_optimization_history(study)
    pio.write_html(fig, file=os.path.join(plots_dir, "optimization_history.html"), auto_open=False)

    fig = plot_parallel_coordinate(study)
    pio.write_html(fig, file=os.path.join(plots_dir, "parallel_coordinate.html"), auto_open=False)

    fig = plot_slice(study)
    pio.write_html(fig, file=os.path.join(plots_dir, "slice.html"), auto_open=False)

    fig = plot_param_importances(study)
    pio.write_html(fig, file=os.path.join(plots_dir, "param_importances.html"), auto_open=False)

    print(f"Gráficos guardados en: {plots_dir}")
    print(f"DB Optuna: {storage_file}")
