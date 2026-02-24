"""PPO training script for VascularNav-v0.

Usage:
    python -m vascularsim.training.train --timesteps 50000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import gymnasium

from vascularsim.envs.wrappers import FlatNavObservation

# Ensure env registration
import vascularsim.envs  # noqa: F401


def make_env(env_id: str, seed: int) -> gymnasium.Env:
    """Create and wrap a VascularNav environment for SB3."""
    env = gymnasium.make(env_id)
    env = FlatNavObservation(env)
    env.reset(seed=seed)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on VascularNav")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model")
    parser.add_argument("--env-id", type=str, default="VascularNav-v0", help="Gymnasium environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Lazy import so the module can be imported without SB3 installed
    from stable_baselines3 import PPO

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Creating environment: {args.env_id}")
    env = make_env(args.env_id, args.seed)

    hyperparams = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "env_id": args.env_id,
    }

    print("Initializing PPO with MlpPolicy")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        n_epochs=hyperparams["n_epochs"],
        gamma=hyperparams["gamma"],
        ent_coef=hyperparams["ent_coef"],
        seed=args.seed,
        verbose=1,
    )

    print(f"Training for {args.timesteps} timesteps...")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - t0

    model_path = os.path.join(args.output_dir, "ppo_vascularnav")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Save training config
    config_path = os.path.join(args.output_dir, "training_config.json")
    hyperparams["training_time_seconds"] = round(elapsed, 2)
    with open(config_path, "w") as f:
        json.dump(hyperparams, f, indent=2)
    print(f"Config saved to {config_path}")

    # Final stats
    print("\n=== Training Complete ===")
    print(f"  Timesteps:      {args.timesteps}")
    print(f"  Wall time:      {elapsed:.1f}s")
    print(f"  Steps/sec:      {args.timesteps / elapsed:.0f}")
    print(f"  Model:          {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
