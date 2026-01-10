# src/envs.py
from __future__ import annotations

from typing import Callable, Optional
import gymnasium as gym
import numpy as np
import cv2

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


class FrameSkip(gym.Wrapper):
    """Repeats the same action N steps and accumulates reward."""
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
    Converts obs (H,W,3) uint8 -> (C,H,W) float32 in [0,1],
    optionally grayscale and resize to 84x84.
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
        img = obs

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
        else:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))

        return img


# -------------------------------------------------------------------
# 🔥 NEW: Reward shaping wrapper
# -------------------------------------------------------------------
class CarRacingRewardShaping(gym.Wrapper):
    """
    Reward shaping to reduce spinning and encourage smooth driving.
    Action = [steer, gas, brake]
    """
    def __init__(
        self,
        env: gym.Env,
        steer_penalty: float = 0.05,
        steer_gas_penalty: float = 0.10,
        brake_reward: float = 0.05,
    ):
        super().__init__(env)
        self.steer_penalty = steer_penalty
        self.steer_gas_penalty = steer_gas_penalty
        self.brake_reward = brake_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        steer = float(action[0])
        gas   = float(action[1])
        brake = float(action[2])

        # Penalize large steering (smoothness)
        reward -= self.steer_penalty * abs(steer)

        # Strong penalty for steering hard while accelerating (main anti-spin term)
        reward -= self.steer_gas_penalty * abs(steer) * gas

        # Small reward for braking (helps in tight curves)
        reward += self.brake_reward * brake

        return obs, reward, terminated, truncated, info


class RewardClip(gym.RewardWrapper):
    """Soft clipping for stability (optional)."""
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

    def _make() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed)

        env = FrameSkip(env, skip=frame_skip)

        # 🔥 reward shaping BEFORE preprocessing & monitor
        env = CarRacingRewardShaping(env)

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

    env_fns = [make_env_fn(
        env_id=env_id,
        seed=seed + i,
        grayscale=grayscale,
        resize=resize,
        frame_skip=frame_skip,
        reward_clip=reward_clip,
        render_mode=render_mode,
    ) for i in range(n_envs)]

    vec = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    vec = VecFrameStack(vec, n_stack=frames_stack, channels_order="first")
    return vec
