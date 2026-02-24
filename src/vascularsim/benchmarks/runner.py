"""Benchmark runner for evaluating navigation agents across difficulty tiers.

Runs an agent through multiple episodes on each benchmark tier and collects
per-tier success rate, mean steps, mean reward, and timing metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vascularsim.benchmarks.environments import TIER_NAMES, make_benchmark_graph
from vascularsim.envs.vascular_nav import VascularNavEnv
from vascularsim.envs.wrappers import FlatNavObservation


def run_benchmark(
    agent_fn: Callable[[np.ndarray], int],
    tiers: tuple[int, ...] = (1, 2, 3, 4, 5),
    n_episodes: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Run benchmark episodes across specified tiers.

    Args:
        agent_fn: Callable that takes a flat observation (np.ndarray) and
            returns an integer action.
        tiers: Which difficulty tiers to evaluate.
        n_episodes: Number of episodes per tier.
        seed: Random seed for graph generation and env resets.

    Returns:
        Results dictionary with per-tier metrics.
    """
    agent_name = getattr(agent_fn, "__name__", "unknown")

    tier_results: dict[str, dict[str, Any]] = {}

    for tier in tiers:
        graph = make_benchmark_graph(tier, seed=seed)
        base_env = VascularNavEnv(graph=graph, max_steps=500)
        env = FlatNavObservation(base_env)

        successes: list[bool] = []
        episode_lengths: list[int] = []
        total_rewards: list[float] = []
        times_per_step: list[float] = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            t_start = time.monotonic()

            while not done:
                action = agent_fn(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated

            elapsed = time.monotonic() - t_start
            successes.append(terminated)
            episode_lengths.append(ep_steps)
            total_rewards.append(ep_reward)
            times_per_step.append(elapsed / max(ep_steps, 1))

        tier_results[str(tier)] = {
            "name": TIER_NAMES[tier],
            "success_rate": float(np.mean(successes)),
            "mean_steps": float(np.mean(episode_lengths)),
            "mean_reward": float(np.mean(total_rewards)),
            "mean_time_per_step": float(np.mean(times_per_step)),
        }

    return {
        "agent_name": agent_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_episodes": n_episodes,
        "tiers": tier_results,
    }


def save_results(results: dict[str, Any], output_path: str) -> None:
    """Save benchmark results as pretty-printed JSON.

    Args:
        results: Results dictionary from run_benchmark.
        output_path: File path to write.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")


def _print_results_table(results: dict[str, Any]) -> None:
    """Print a formatted table of benchmark results to stdout."""
    print(f"\nBenchmark Results: {results['agent_name']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Seed: {results['seed']}  |  Episodes per tier: {results['n_episodes']}")
    print("-" * 78)
    print(f"{'Tier':<6} {'Name':<14} {'Success%':>10} {'Mean Steps':>12} "
          f"{'Mean Reward':>12} {'ms/step':>10}")
    print("-" * 78)

    for tier_key in sorted(results["tiers"], key=int):
        t = results["tiers"][tier_key]
        print(f"{tier_key:<6} {t['name']:<14} {t['success_rate']*100:>9.1f}% "
              f"{t['mean_steps']:>12.1f} {t['mean_reward']:>12.2f} "
              f"{t['mean_time_per_step']*1000:>9.3f}")

    print("-" * 78)


def main() -> None:
    """CLI entry point for running benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run VascularSim navigation benchmarks")
    parser.add_argument("--agent", choices=["random", "ppo"], default="random",
                        help="Agent type to benchmark (default: random)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model (required for ppo)")
    parser.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Difficulty tiers to run (default: 1 2 3 4 5)")
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Episodes per tier (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON path (default: benchmark_results.json)")
    args = parser.parse_args()

    if args.agent == "random":
        # For the random agent we need the env's action space, so we create
        # a closure that captures a per-tier RNG.  Since run_benchmark creates
        # the env internally, we use a simple random-action approach.
        rng = np.random.default_rng(args.seed)

        def random_agent(obs: np.ndarray) -> int:
            # Action space size varies by tier, but we pick a small random int.
            # Invalid actions map to "stay", so this is safe.
            return int(rng.integers(0, 10))

        random_agent.__name__ = "random"
        agent_fn = random_agent

    elif args.agent == "ppo":
        if args.model_path is None:
            print("ERROR: --model-path required for ppo agent", file=sys.stderr)
            sys.exit(1)
        try:
            from stable_baselines3 import PPO
            model = PPO.load(args.model_path)

            def ppo_agent(obs: np.ndarray) -> int:
                action, _ = model.predict(obs, deterministic=True)
                return int(action)

            ppo_agent.__name__ = "ppo"
            agent_fn = ppo_agent
        except ImportError:
            print("ERROR: stable_baselines3 required for ppo agent", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"ERROR: Unknown agent type: {args.agent}", file=sys.stderr)
        sys.exit(1)

    results = run_benchmark(
        agent_fn=agent_fn,
        tiers=tuple(args.tiers),
        n_episodes=args.n_episodes,
        seed=args.seed,
    )

    _print_results_table(results)
    save_results(results, args.output)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
