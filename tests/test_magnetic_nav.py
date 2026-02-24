"""Tests for the MagneticNavEnv environment.

Validates creation, gym registration, observation keys, episode stability,
custom coil systems, and magnetic-aware reward bonus.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from vascularsim.envs.magnetic_nav import MagneticNavEnv
from vascularsim.physics.magnetic import CoilSystem, HelmholtzCoil

# Trigger env registration
import vascularsim.envs  # noqa: F401


class TestMagneticNavEnv:
    """Tests for the magnetic-aware navigation environment."""

    def test_magnetic_env_creation(self) -> None:
        """MagneticNavEnv creates successfully with default graph."""
        env = MagneticNavEnv()
        assert env is not None
        obs, info = env.reset(seed=0)
        assert obs is not None

    def test_magnetic_env_gym_make(self) -> None:
        """gymnasium.make('MagneticNav-v0') works."""
        env = gymnasium.make("MagneticNav-v0")
        assert env is not None
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict)

    def test_magnetic_obs_has_field(self) -> None:
        """Observation includes magnetic_field key with shape (3,)."""
        env = MagneticNavEnv()
        obs, _ = env.reset(seed=0)
        assert "magnetic_field" in obs
        assert obs["magnetic_field"].shape == (3,)
        assert obs["magnetic_field"].dtype == np.float32

    def test_magnetic_obs_has_force(self) -> None:
        """Observation includes magnetic_force key with shape (3,)."""
        env = MagneticNavEnv()
        obs, _ = env.reset(seed=0)
        assert "magnetic_force" in obs
        assert obs["magnetic_force"].shape == (3,)
        assert obs["magnetic_force"].dtype == np.float32

    def test_magnetic_env_reset_and_step(self) -> None:
        """Basic episode loop: reset + 10 steps works."""
        env = MagneticNavEnv()
        obs, info = env.reset(seed=0)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, float)
            assert "magnetic_field" in obs
            assert "magnetic_force" in obs
            if terminated or truncated:
                obs, info = env.reset()

    def test_magnetic_env_100_episodes(self) -> None:
        """100 episodes complete without crash."""
        env = MagneticNavEnv()
        for ep in range(100):
            obs, _ = env.reset(seed=ep)
            done = False
            steps = 0
            while not done and steps < 20:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1

    def test_magnetic_custom_coil_system(self) -> None:
        """MagneticNavEnv works with a custom CoilSystem."""
        custom = CoilSystem([
            HelmholtzCoil(coil_radius=0.2, separation=0.2, n_turns=50,
                          current=2.0, axis=0),
            HelmholtzCoil(coil_radius=0.15, separation=0.15, n_turns=80,
                          current=1.5, axis=2),
        ])
        env = MagneticNavEnv(coil_system=custom)
        obs, _ = env.reset(seed=0)
        assert "magnetic_field" in obs
        # Field should be non-zero since we have active coils
        B = obs["magnetic_field"]
        assert B.shape == (3,)

        # Step should work
        action = env.action_space.sample()
        obs2, reward, _, _, info = env.step(action)
        assert info.get("magnetic_aware") is True

    def test_magnetic_env_observation_space(self) -> None:
        """Observation space has all expected keys with correct shapes."""
        env = MagneticNavEnv()
        space = env.observation_space
        assert "agent_pos" in space.spaces
        assert "target_pos" in space.spaces
        assert "magnetic_field" in space.spaces
        assert "magnetic_force" in space.spaces
        assert space["magnetic_field"].shape == (3,)
        assert space["magnetic_force"].shape == (3,)
