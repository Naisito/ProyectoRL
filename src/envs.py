# src/envs.py
from __future__ import annotations

from typing import Callable, Optional
import gymnasium as gym
import numpy as np
import cv2

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


class FrameSkip(gym.Wrapper):
    """Repite la misma acción N pasos y acumula recompensa."""
    def __init__(self, env: gym.Env, skip: int = 2):
        super().__init__(env)
        assert skip >= 1
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        obs = None

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class PreprocessCarRacing(gym.ObservationWrapper):
    """
    Convierte obs (H,W,3) uint8 -> (C,H,W) float32 en [0,1],
    opcionalmente a gris y resize a 84x84.
    """
    def __init__(self, env: gym.Env, grayscale: bool = True, resize: int = 84):
        super().__init__(env)
        self.grayscale = grayscale
        self.resize = resize

        c = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(c, resize, resize),
            dtype=np.float32
        )

    def observation(self, obs):
        # obs: (H,W,3) uint8
        img = obs

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # (H,W)
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)  # (1,H,W)
        else:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # (3,H,W)

        return img


class RewardClip(gym.RewardWrapper):
    """Clipping suave para estabilizar (opcional, útil en algunos runs)."""
    def __init__(self, env: gym.Env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def reward(self, reward):
        return float(np.clip(reward, self.low, self.high))


def make_env_fn(
    env_id: str,
    seed: int,
    grayscale: bool = True,
    resize: int = 84,
    frame_skip: int = 2,
    reward_clip: bool = False,
    render_mode: Optional[str] = None,
) -> Callable[[], gym.Env]:
    """
    Devuelve un factory para VecEnv.
    """
    def _make() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed)

        env = FrameSkip(env, skip=frame_skip)
        env = PreprocessCarRacing(env, grayscale=grayscale, resize=resize)
        if reward_clip:
            env = RewardClip(env, low=-1.0, high=1.0)

        env = Monitor(env)
        return env

    return _make


def make_vec_env(
    env_id: str,
    seed: int,
    n_envs: int = 1,
    grayscale: bool = True,
    resize: int = 84,
    frame_skip: int = 2,
    frames_stack: int = 4,
    reward_clip: bool = False,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """
    Crea VecEnv + FrameStack para PPO.
    """
    env_fns = [make_env_fn(
        env_id=env_id,
        seed=seed + i,
        grayscale=grayscale,
        resize=resize,
        frame_skip=frame_skip,
        reward_clip=reward_clip,
        render_mode=render_mode,
    ) for i in range(n_envs)]

    # Use subprocess vectorized env when n_envs > 1 to parallelize env.step calls
    if n_envs > 1:
        vec = SubprocVecEnv(env_fns)
    else:
        vec = DummyVecEnv(env_fns)
    vec = VecFrameStack(vec, n_stack=frames_stack, channels_order="first")
    return vec


def make_eval_env(
    env_id: str,
    seed: int,
    grayscale: bool = True,
    resize: int = 84,
    frame_skip: int = 2,
    frames_stack: int = 4,
    video_dir: Optional[str] = None,
    record_trigger_episodes: int = 5,
) -> gym.Env:
    """
    Env de evaluación opcionalmente con vídeo.
    OJO: RecordVideo va fuera del VecEnv; aquí devolvemos env normal para eval manual.
    """
    render_mode = "rgb_array" if video_dir else None
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)

    env = FrameSkip(env, skip=frame_skip)

    if video_dir:
        env = gym.wrappers.RecordVideo(
            env,
            video_dir=video_dir,
            episode_trigger=lambda ep: ep < record_trigger_episodes,
            name_prefix="carracing"
        )

    env = PreprocessCarRacing(env, grayscale=grayscale, resize=resize)
    env = gym.wrappers.FrameStack(env, num_stack=frames_stack)  # (stack, C,H,W) lógico

    # Convertimos el FrameStack de gym a array (C*stack, H, W) para el modelo SB3:
    env = _GymFrameStackToCHW(env, channels_first=True)

    env = Monitor(env)
    return env


class _GymFrameStackToCHW(gym.ObservationWrapper):
    """
    gym FrameStack devuelve LazyFrames apiladas; aquí lo convertimos a np.array.
    Y lo aplanamos en canales: (stack, C, H, W) -> (stack*C, H, W)
    """
    def __init__(self, env: gym.Env, channels_first: bool = True):
        super().__init__(env)
        self.channels_first = channels_first

        obs_space = env.observation_space
        # Esperamos (stack, C, H, W)
        assert len(obs_space.shape) == 4, f"Obs shape inesperada: {obs_space.shape}"
        stack, c, h, w = obs_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(stack * c, h, w),
            dtype=np.float32
        )

    def observation(self, obs):
        arr = np.array(obs, dtype=np.float32)  # (stack, C, H, W)
        arr = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3]))
        return arr
