"""
Trains a DQN agent on FlappyBird-v0.

Each run is tied to a --seed so the team can relaunch the same training
multiple times and compare curves (see the assignment's "relance plusieurs
fois" requirement). Evaluation happens periodically during training and is
logged to logs/eval_seed_<seed>/evaluations.npz for plotting, and the best
checkpoint seen during training is saved separately from the final model.

Usage:
    python training/train_dqn.py --seed 0 --timesteps 200000
"""
import argparse

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


def make_env():
    return Monitor(gym.make("FlappyBird-v0", use_lidar=False))


def train(seed: int, timesteps: int, eval_freq: int, eval_episodes: int):
    env = make_env()
    eval_env = make_env()

    ckpt_dir = f"models/checkpoints/seed_{seed}"
    eval_log_dir = f"logs/eval_seed_{seed}"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=ckpt_dir,
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )

    model = DQN(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(f"{ckpt_dir}/final_model")

    env.close()
    eval_env.close()
    print(f"\nDone. Best model + eval log under {ckpt_dir} / {eval_log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    train(args.seed, args.timesteps, args.eval_freq, args.eval_episodes)
