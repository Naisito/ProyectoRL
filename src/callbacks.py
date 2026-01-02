# src/callbacks.py
from typing import Optional

import cv2
import numpy as np
import threading
import queue
import time

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class VisualizeCallback(BaseCallback):
    """Muestra el render del primer entorno y la versión preprocesada en ventanas OpenCV.

    Para reducir el impacto en el bucle de entrenamiento, la visualización se realiza en un hilo
    separado. El hilo consume imágenes de una cola pequeña y descarta frames si la cola está llena.

    Nota: para que esto funcione, crea el env con `render_mode='rgb_array'` y usa `n_envs=1`.
    """
    def __init__(
        self,
        render_freq: int = 1,
        grayscale: bool = True,
        resize: int = 84,
        show_raw: bool = True,
        proc_scale: int = 4,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.render_freq = max(1, int(render_freq))
        self.grayscale = grayscale
        self.resize = resize
        self.show_raw = bool(show_raw)
        self.proc_scale = max(1, int(proc_scale))

        self._win_raw = "env_raw"
        self._win_proc = "env_preprocessed"

        # Cola pequeña, si está llena descartamos frames viejos (no bloqueante)
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _start_thread(self):
        if self._thread is not None and self._thread.is_alive():
            return

        def worker(q: "queue.Queue[tuple]", stop_event: threading.Event):
            while not stop_event.is_set():
                try:
                    rgb, proc_show = q.get(timeout=0.5)
                except Exception:
                    continue

                try:
                    if self.show_raw and rgb is not None:
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        cv2.imshow(self._win_raw, bgr)

                    if proc_show is not None:
                        # Escalamos para visualización sin afectar lo que percibe el modelo
                        if self.proc_scale != 1:
                            h, w = proc_show.shape[:2]
                            proc_disp = cv2.resize(proc_show, (w * self.proc_scale, h * self.proc_scale), interpolation=cv2.INTER_NEAREST)
                        else:
                            proc_disp = proc_show
                        cv2.imshow(self._win_proc, proc_disp)

                    # waitKey en hilo para que no bloquee el hilo principal
                    cv2.waitKey(1)
                except Exception:
                    # Evitar que la hebra muera por un error de render
                    pass

        self._stop_event.clear()
        self._thread = threading.Thread(target=worker, args=(self._q, self._stop_event), daemon=True)
        self._thread.start()

    def _stop_thread(self):
        try:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
        except Exception:
            pass

    def _on_training_start(self) -> None:
        # Lanzar hilo de visualización
        self._start_thread()

    def _on_training_end(self) -> None:
        # Detener hilo y cerrar ventanas
        self._stop_thread()
        try:
            cv2.destroyWindow(self._win_proc)
            if self.show_raw:
                cv2.destroyWindow(self._win_raw)
        except Exception:
            pass

    def _on_step(self) -> bool:
        # Solo cada render_freq pasos
        if self.n_calls % self.render_freq != 0:
            return True

        try:
            env = self.training_env
            # Obtener primer env
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
                try:
                    frame = first.render(mode="rgb_array")
                except Exception:
                    frame = None

            if frame is None:
                return True

            rgb = np.asarray(frame)

            # Crear versión preprocesada (sin modificar la observación real)
            proc = rgb
            if self.grayscale:
                proc = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
                proc = cv2.resize(proc, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
                proc_show = (proc * 255).astype(np.uint8) if proc.dtype == np.float32 else proc
            else:
                proc = cv2.resize(proc, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
                proc_show = cv2.cvtColor(proc, cv2.COLOR_RGB2BGR)

            if proc_show.dtype == np.float32 or proc_show.dtype == np.float64:
                proc_show = (np.clip(proc_show, 0.0, 1.0) * 255).astype(np.uint8)

            # Push non-blocking: si la cola está llena descartamos el frame
            try:
                self._q.put_nowait((rgb if self.show_raw else None, proc_show))
            except queue.Full:
                # descartamos si la cola está llena
                pass
        except Exception:
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
