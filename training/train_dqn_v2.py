"""
DQN V2 for Flappy Bird.

Changes vs V1:
- Reward shaping focused more strongly on passing pipes
- Larger neural network
- Larger replay buffer
- Longer initial exploration
- Longer exploration schedule
- Slower target-network updates

Usage:
    python training/train_dqn_v2.py --seed 5 --timesteps 2000000
"""

import argparse

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


class FlappyRewardWrapper(gym.Wrapper):
    """
    Reward shaping:

    +0.01 for every frame survived
    +5.0 for passing a pipe
    -1.0 when dying

    The real Flappy score remains available in info["score"].
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_score = info.get("score", 0)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        current_score = info.get("score", self.previous_score)

        # Tiny reward for staying alive
        reward = 0.01

        # Strong reward for actually passing pipes
        pipes_passed = current_score - self.previous_score

        if pipes_passed > 0:
            reward += 5.0 * pipes_passed

        # Death penalty
        if terminated:
            reward -= 1.0

        self.previous_score = current_score

        return obs, reward, terminated, truncated, info


def make_env():
    env = gym.make(
        "FlappyBird-v0",
        use_lidar=False,
    )

    env = FlappyRewardWrapper(env)
    env = Monitor(env)

    return env


def train(seed: int, timesteps: int, eval_freq: int, eval_episodes: int):
    env = make_env()
    eval_env = make_env()

    ckpt_dir = f"models/checkpoints/v2_seed_{seed}"
    eval_log_dir = f"logs/eval_v2_seed_{seed}"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=ckpt_dir,
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1,
    )

    model = DQN(
        "MlpPolicy",
        env,

        seed=seed,
        verbose=1,

        # Learning
        learning_rate=1e-4,
        gamma=0.99,

        # Replay memory
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=128,

        # Training frequency
        train_freq=4,
        gradient_steps=1,

        # Target network
        target_update_interval=5_000,

        # Exploration
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,

        # Bigger network
        policy_kwargs=dict(
            net_arch=[256, 256]
        ),

        device="auto",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
    )

    model.save(f"{ckpt_dir}/final_model")

    env.close()
    eval_env.close()

    print()
    print("Training finished.")
    print(f"Best model: {ckpt_dir}/best_model.zip")
    print(f"Final model: {ckpt_dir}/final_model.zip")
    print(f"Evaluation logs: {eval_log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=2_000_000)

    # Don't evaluate every 5k anymore.
    # Longer interval + more episodes = more reliable evaluation.
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)

    args = parser.parse_args()

    train(
        args.seed,
        args.timesteps,
        args.eval_freq,
        args.eval_episodes,
    )