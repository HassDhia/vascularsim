"""Evaluation script for trained VascularNav agents.

Usage:
    python -m vascularsim.training.evaluate --model-path checkpoints/ppo_vascularnav.zip
"""

from __future__ import annotations

import argparse
from typing import Any

import gymnasium
import networkx as nx
import numpy as np

from vascularsim.envs.wrappers import FlatNavObservation

# Ensure env registration
import vascularsim.envs  # noqa: F401


def evaluate_agent(
    model_path: str,
    env: gymnasium.Env,
    n_episodes: int = 100,
) -> dict[str, Any]:
    """Evaluate a trained SB3 model on the given environment.

    Args:
        model_path: Path to saved SB3 model (.zip).
        env: Gymnasium environment (should be wrapped with FlatNavObservation).
        n_episodes: Number of evaluation episodes.

    Returns:
        Dict with success_rate, mean_episode_length, mean_reward, path_length_ratio.
    """
    from stable_baselines3 import PPO

    model = PPO.load(model_path)

    successes = 0
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    path_ratios: list[float] = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        # Try to compute shortest path for ratio
        shortest_path_len: int | None = None
        try:
            inner_env = env.unwrapped
            undirected = inner_env._undirected  # noqa: SLF001
            start_node = info.get("start_node")
            target_node = info.get("target_node")
            if start_node and target_node:
                shortest_path_len = nx.shortest_path_length(undirected, start_node, target_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound, AttributeError):
            shortest_path_len = None

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)

        if terminated:
            successes += 1
            if shortest_path_len is not None and shortest_path_len > 0:
                path_ratios.append(steps / shortest_path_len)

    return {
        "success_rate": successes / max(n_episodes, 1),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(episode_rewards)),
        "path_length_ratio": float(np.mean(path_ratios)) if path_ratios else float("nan"),
        "n_episodes": n_episodes,
    }


def evaluate_random(
    env: gymnasium.Env,
    n_episodes: int = 100,
) -> dict[str, Any]:
    """Evaluate a random-action baseline on the given environment.

    Args:
        env: Gymnasium environment (should be wrapped with FlatNavObservation).
        n_episodes: Number of evaluation episodes.

    Returns:
        Dict with success_rate, mean_episode_length, mean_reward, path_length_ratio.
    """
    successes = 0
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    path_ratios: list[float] = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        shortest_path_len: int | None = None
        try:
            inner_env = env.unwrapped
            undirected = inner_env._undirected  # noqa: SLF001
            start_node = info.get("start_node")
            target_node = info.get("target_node")
            if start_node and target_node:
                shortest_path_len = nx.shortest_path_length(undirected, start_node, target_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound, AttributeError):
            shortest_path_len = None

        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)

        if terminated:
            successes += 1
            if shortest_path_len is not None and shortest_path_len > 0:
                path_ratios.append(steps / shortest_path_len)

    return {
        "success_rate": successes / max(n_episodes, 1),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(episode_rewards)),
        "path_length_ratio": float(np.mean(path_ratios)) if path_ratios else float("nan"),
        "n_episodes": n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VascularNav agent")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved SB3 model (.zip)")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--env-id", type=str, default="VascularNav-v0", help="Gymnasium environment ID")
    args = parser.parse_args()

    env = gymnasium.make(args.env_id)
    env = FlatNavObservation(env)
    env.reset(seed=args.seed)

    print(f"Evaluating trained agent: {args.model_path}")
    ppo_metrics = evaluate_agent(args.model_path, env, args.n_episodes)

    print(f"\nEvaluating random baseline ({args.n_episodes} episodes)...")
    random_metrics = evaluate_random(env, args.n_episodes)

    # Print comparison table
    print("\n" + "=" * 55)
    print(f"{'Metric':<25} {'PPO':>12} {'Random':>12}")
    print("-" * 55)
    for key in ["success_rate", "mean_episode_length", "mean_reward", "path_length_ratio"]:
        ppo_val = ppo_metrics[key]
        rand_val = random_metrics[key]
        if key == "success_rate":
            print(f"{key:<25} {ppo_val:>11.1%} {rand_val:>11.1%}")
        else:
            print(f"{key:<25} {ppo_val:>12.2f} {rand_val:>12.2f}")
    print("=" * 55)

    env.close()


if __name__ == "__main__":
    main()
