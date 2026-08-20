"""
Baseline agent: picks a uniformly random action every step.
This is the reference score the trained agent must beat.

Usage:
    python agents/random_agent.py --episodes 30
    python agents/random_agent.py --episodes 3 --render   # watch it play
"""
import argparse
import statistics

import flappy_bird_gymnasium  # noqa: F401  (registers the FlappyBird-v0 env)
import gymnasium as gym


def run(episodes: int, render: bool):
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
    scores = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        episode_score = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_score = info.get("score", episode_score)

        scores.append(episode_score)
        print(f"episode {ep + 1}/{episodes}: score={episode_score}")

    env.close()

    print("\n--- Random agent baseline ---")
    print(f"episodes: {episodes}")
    print(f"mean score: {statistics.mean(scores):.2f}")
    print(f"max score:  {max(scores)}")
    print(f"min score:  {min(scores)}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    run(args.episodes, args.render)
