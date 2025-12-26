"""Utility to inspect the observation space and a sample observation for the training env.

Run with:
    python -m src.debug_obs

Or with custom config:
    python -m src.debug_obs --config configs/ppo_carracing.yaml
"""
import argparse
import numpy as np
from src.config import load_train_config, TrainConfig
from src.envs import make_vec_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_train_config(args.config) if args.config else TrainConfig()
    env = make_vec_env(
        env_id=cfg.env_id,
        seed=cfg.seed,
        n_envs=args.n_envs,
        grayscale=cfg.grayscale,
        resize=cfg.resize,
        frame_skip=cfg.frame_skip,
        frames_stack=cfg.frames_stack,
        reward_clip=False,
    )

    print("Observation space:", env.observation_space)
    obs = env.reset()
    # If DummyVecEnv, env.reset() returns array shaped (n_envs, ...)
    print("Type obs:", type(obs))
    try:
        print("Obs shape:", np.array(obs).shape)
    except Exception as e:
        print("Could not convert obs to array:", e)


if __name__ == '__main__':
    main()
