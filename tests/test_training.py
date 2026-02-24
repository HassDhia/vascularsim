"""Tests for training infrastructure (wrapper + evaluation).

These tests are fast -- no actual PPO training is performed.
"""

from __future__ import annotations

import gymnasium
import numpy as np

import vascularsim.envs  # noqa: F401  -- register envs
from vascularsim.envs.wrappers import FlatNavObservation
from vascularsim.training.evaluate import evaluate_random


def _make_wrapped_env(seed: int = 0) -> gymnasium.Env:
    """Create a FlatNavObservation-wrapped VascularNav-v0."""
    env = gymnasium.make("VascularNav-v0")
    env = FlatNavObservation(env)
    env.reset(seed=seed)
    return env


class TestFlatWrapper:
    def test_flat_wrapper_shape(self) -> None:
        env = _make_wrapped_env()
        obs, _ = env.reset()
        assert obs.shape == (8,), f"Expected shape (8,), got {obs.shape}"
        env.close()

    def test_flat_wrapper_in_observation_space(self) -> None:
        env = _make_wrapped_env()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            f"Observation {obs} not in observation_space {env.observation_space}"
        )
        env.close()

    def test_flat_wrapper_step(self) -> None:
        env = _make_wrapped_env()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (8,), f"Step obs shape: expected (8,), got {obs.shape}"
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_flat_wrapper_dtype(self) -> None:
        env = _make_wrapped_env()
        obs, _ = env.reset()
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        env.close()


class TestEvaluation:
    def test_evaluate_random(self) -> None:
        env = _make_wrapped_env(seed=123)
        metrics = evaluate_random(env, n_episodes=10)
        assert "success_rate" in metrics
        assert "mean_episode_length" in metrics
        assert "mean_reward" in metrics
        assert "path_length_ratio" in metrics
        assert 0.0 <= metrics["success_rate"] <= 1.0
        assert metrics["mean_episode_length"] > 0
        assert metrics["n_episodes"] == 10
        env.close()


class TestImports:
    def test_training_imports(self) -> None:
        """All training modules import without error (no SB3 required at import time)."""
        import vascularsim.training  # noqa: F401
        from vascularsim.envs.wrappers import FlatNavObservation  # noqa: F811
        from vascularsim.training.evaluate import evaluate_random  # noqa: F811

        assert callable(FlatNavObservation)
        assert callable(evaluate_random)
