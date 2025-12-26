# src/callbacks.py
from typing import Optional

import cv2
import numpy as np

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class VisualizeCallback(BaseCallback):
    """Muestra el render del primer entorno y la versión preprocesada en ventanas OpenCV.

    Nota: para que esto funcione, crea el env con `render_mode='rgb_array'` y usa `n_envs=1`.
    """
    def __init__(self, render_freq: int = 1, grayscale: bool = True, resize: int = 84, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = max(1, int(render_freq))
        self.grayscale = grayscale
        self.resize = resize
        self._win_raw = "env_raw"
        self._win_proc = "env_preprocessed"

    def _on_step(self) -> bool:
        # Solo cada render_freq pasos
        if self.n_calls % self.render_freq != 0:
            return True

        try:
            env = self.training_env
            # DummyVecEnv tiene .envs con la lista de entornos
            first = None
            if hasattr(env, "envs"):
                first = env.envs[0]
            elif hasattr(env, "venv") and hasattr(env.venv, "envs"):
                first = env.venv.envs[0]

            if first is None:
                return True

            # Pedimos un frame RGB via render()
            try:
                frame = first.render()
            except Exception:
                # Algunas versiones requieren mode argument
                try:
                    frame = first.render(mode="rgb_array")
                except Exception:
                    return True

            if frame is None:
                return True

            # Mostrar frame raw
            rgb = np.asarray(frame)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(self._win_raw, bgr)

            # Crear versión preprocesada similar a PreprocessCarRacing
            proc = rgb
            if self.grayscale:
                proc = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
                proc = cv2.resize(proc, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
                proc_show = (proc * 255).astype(np.uint8) if proc.dtype == np.float32 else proc
            else:
                proc = cv2.resize(proc, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
                proc_show = cv2.cvtColor(proc, cv2.COLOR_RGB2BGR)

            # Si es float en [0,1], convertimos a uint8
            if proc_show.dtype == np.float32 or proc_show.dtype == np.float64:
                proc_show = (np.clip(proc_show, 0.0, 1.0) * 255).astype(np.uint8)

            cv2.imshow(self._win_proc, proc_show)
            cv2.waitKey(1)
        except Exception:
            # No fallar el entrenamiento por la visualización
            pass

        return True


def build_callbacks(
    eval_env: VecEnv,
    model_dir: str,
    log_dir: str,
    eval_freq: int,
    n_eval_episodes: int,
    save_freq: int,
    visual: Optional[dict] = None,
):
    checkpoint = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix="ppo_carracing_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    evaluator = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    callbacks = [checkpoint, evaluator]
    if visual:
        vc = VisualizeCallback(
            render_freq=visual.get("render_freq", 1),
            grayscale=visual.get("grayscale", True),
            resize=visual.get("resize", 84),
        )
        callbacks.append(vc)

    return callbacks
