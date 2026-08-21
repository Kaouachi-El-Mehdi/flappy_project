import argparse
import statistics

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
from stable_baselines3 import DQN

from train_dqn_v2 import FlappyRewardWrapper


def make_env(render=False):
    env = gym.make(
        "FlappyBird-v0",
        render_mode="human" if render else None,
        use_lidar=False,
    )
    env = FlappyRewardWrapper(env)
    return env


def evaluate(model_path: str, episodes: int, render: bool):
    env = make_env(render)
    model = DQN.load(model_path)

    scores = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        episode_score = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            episode_score = info.get("score", episode_score)

        scores.append(episode_score)
        print(f"episode {ep + 1}/{episodes}: score={episode_score}")

    env.close()

    print("\n--- DQN V2 evaluation ---")
    print(f"model: {model_path}")
    print(f"episodes: {episodes}")
    print(f"mean score: {statistics.mean(scores):.2f}")
    print(f"max score:  {max(scores)}")
    print(f"min score:  {min(scores)}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints/v2_seed_5/best_model.zip",
    )

    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render)