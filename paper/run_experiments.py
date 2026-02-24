"""Run all experiments needed for the VascularSim paper.

Outputs JSON files consumed by the LaTeX document:
  - benchmark_all_tiers.json   (random agent, 5 tiers, 50 eps each)
  - shortest_path_baseline.json (oracle, 5 tiers, 50 eps each)
  - ppo_vs_baselines.json      (PPO vs random vs oracle on default env)
  - training_curve.json        (PPO reward over timesteps)
"""

import json
import os
import sys
import time

import gymnasium
import networkx as nx
import numpy as np

# Ensure vascularsim envs are registered
import vascularsim.envs  # noqa: F401
from vascularsim.benchmarks.environments import TIER_NAMES, make_benchmark_graph
from vascularsim.envs.vascular_nav import VascularNavEnv
from vascularsim.envs.wrappers import FlatNavObservation

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Shortest-path oracle agent ──────────────────────────────────

def make_oracle_agent(env):
    """Create an agent that always follows the shortest path."""
    inner = env.unwrapped
    undirected = inner._undirected
    node_list = inner._node_list

    def oracle(obs):
        agent_node = inner._agent_node
        target_node = inner._target_node
        try:
            path = nx.shortest_path(undirected, agent_node, target_node)
        except nx.NetworkXNoPath:
            return inner._max_neighbors  # stay
        if len(path) < 2:
            return inner._max_neighbors  # already at target
        next_node = path[1]
        neighbours = sorted(undirected.neighbors(agent_node))
        if next_node in neighbours:
            return neighbours.index(next_node)
        return inner._max_neighbors  # fallback: stay

    return oracle


# ── Evaluation helper ───────────────────────────────────────────

def evaluate_agent_fn(env, agent_fn, n_episodes=50, seed=42):
    """Evaluate an agent function on an environment."""
    successes = []
    ep_lengths = []
    ep_rewards = []
    path_ratios = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        # Compute shortest path for ratio
        sp_len = None
        try:
            inner = env.unwrapped
            sp_len = nx.shortest_path_length(
                inner._undirected, inner._agent_node, inner._target_node
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound, AttributeError):
            sp_len = None

        while not done:
            action = agent_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        successes.append(terminated)
        ep_lengths.append(steps)
        ep_rewards.append(total_reward)
        if terminated and sp_len and sp_len > 0:
            path_ratios.append(steps / sp_len)

    return {
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(ep_lengths)),
        "mean_reward": float(np.mean(ep_rewards)),
        "path_ratio": float(np.mean(path_ratios)) if path_ratios else float("nan"),
        "n_episodes": n_episodes,
    }


# ── Experiment 1: Per-tier benchmark (random + oracle) ──────────

def run_tier_benchmarks():
    """Run random and oracle agents across all 5 tiers."""
    print("\n=== Experiment 1: Per-tier benchmarks ===")
    results = {}

    for tier in range(1, 6):
        print(f"\n  Tier {tier} ({TIER_NAMES[tier]})...")
        graph = make_benchmark_graph(tier, seed=42)
        base_env = VascularNavEnv(graph=graph, max_steps=500)
        env = FlatNavObservation(base_env)

        rng = np.random.default_rng(42)

        def random_agent(obs):
            return int(rng.integers(0, 10))

        oracle_fn = make_oracle_agent(env)

        print(f"    Random agent (50 episodes)...")
        random_res = evaluate_agent_fn(env, random_agent, n_episodes=50, seed=42)

        print(f"    Oracle agent (50 episodes)...")
        oracle_res = evaluate_agent_fn(env, oracle_fn, n_episodes=50, seed=42)

        results[str(tier)] = {
            "name": TIER_NAMES[tier],
            "nodes": graph.num_nodes,
            "random": random_res,
            "oracle": oracle_res,
        }

        print(f"    Random: {random_res['success_rate']*100:.0f}% success, "
              f"{random_res['mean_steps']:.1f} steps")
        print(f"    Oracle: {oracle_res['success_rate']*100:.0f}% success, "
              f"{oracle_res['mean_steps']:.1f} steps, "
              f"ratio {oracle_res['path_ratio']:.2f}")

        env.close()

    path = os.path.join(OUTPUT_DIR, "benchmark_all_tiers.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {path}")
    return results


# ── Experiment 2: PPO training with curve logging ───────────────

def run_ppo_training():
    """Train PPO and log training curve data."""
    print("\n=== Experiment 2: PPO training with curve ===")
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    class CurveCallback(BaseCallback):
        """Log episode rewards during training."""
        def __init__(self):
            super().__init__()
            self.curve_data = []
            self._ep_rewards = []

        def _on_step(self):
            # Check for episode completion
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.curve_data.append({
                        "timestep": self.num_timesteps,
                        "reward": float(info["episode"]["r"]),
                        "length": int(info["episode"]["l"]),
                    })
            return True

    env = gymnasium.make("VascularNav-v0")
    env = FlatNavObservation(env)

    # Wrap with Monitor for episode stats
    from stable_baselines3.common.monitor import Monitor
    import tempfile
    log_dir = tempfile.mkdtemp()
    env = Monitor(env, log_dir)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.01,
        seed=42, verbose=0,
    )

    callback = CurveCallback()
    print("  Training PPO for 200k steps...")
    t0 = time.time()
    model.learn(total_timesteps=200_000, callback=callback)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({200_000/elapsed:.0f} steps/s)")

    # Save training curve
    curve_path = os.path.join(OUTPUT_DIR, "training_curve.json")
    with open(curve_path, "w") as f:
        json.dump(callback.curve_data, f)
    print(f"  Saved curve ({len(callback.curve_data)} episodes): {curve_path}")

    # Save model for evaluation
    model_path = os.path.join(OUTPUT_DIR, "ppo_vascularnav")
    model.save(model_path)
    env.close()

    return model_path, elapsed


# ── Experiment 3: PPO vs baselines on default env ───────────────

def run_comparison(model_path):
    """Compare PPO, random, and oracle on the default env."""
    print("\n=== Experiment 3: PPO vs baselines ===")
    from stable_baselines3 import PPO

    env = gymnasium.make("VascularNav-v0")
    env = FlatNavObservation(env)

    # PPO agent
    model = PPO.load(model_path)

    def ppo_agent(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    # Random agent
    rng = np.random.default_rng(42)
    def random_agent(obs):
        return int(rng.integers(0, env.action_space.n))

    # Oracle agent
    oracle_fn = make_oracle_agent(env)

    print("  PPO (100 episodes)...")
    ppo_res = evaluate_agent_fn(env, ppo_agent, n_episodes=100, seed=42)
    print(f"    Success: {ppo_res['success_rate']*100:.0f}%, "
          f"Steps: {ppo_res['mean_steps']:.1f}, "
          f"Ratio: {ppo_res['path_ratio']:.2f}")

    rng = np.random.default_rng(42)  # reset
    print("  Random (100 episodes)...")
    random_res = evaluate_agent_fn(env, random_agent, n_episodes=100, seed=42)
    print(f"    Success: {random_res['success_rate']*100:.0f}%, "
          f"Steps: {random_res['mean_steps']:.1f}, "
          f"Ratio: {random_res['path_ratio']:.2f}")

    print("  Oracle (100 episodes)...")
    oracle_res = evaluate_agent_fn(env, oracle_fn, n_episodes=100, seed=42)
    print(f"    Success: {oracle_res['success_rate']*100:.0f}%, "
          f"Steps: {oracle_res['mean_steps']:.1f}, "
          f"Ratio: {oracle_res['path_ratio']:.2f}")

    comparison = {
        "ppo": ppo_res,
        "random": random_res,
        "oracle": oracle_res,
    }

    path = os.path.join(OUTPUT_DIR, "ppo_vs_baselines.json")
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {path}")
    env.close()
    return comparison


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("VascularSim Paper Experiments")
    print("=" * 50)

    # Run all experiments
    tier_results = run_tier_benchmarks()
    model_path, train_time = run_ppo_training()
    comparison = run_comparison(model_path)

    print("\n" + "=" * 50)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Training time: {train_time:.1f}s")
    print("Output files:")
    for f in ["benchmark_all_tiers.json", "training_curve.json", "ppo_vs_baselines.json"]:
        print(f"  {os.path.join(OUTPUT_DIR, f)}")
