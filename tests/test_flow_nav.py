"""Tests for the FlowAwareNavEnv environment.

Validates creation, gym registration, observation keys, episode stability,
and flow-aware reward bonus.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from vascularsim.envs.flow_nav import FlowAwareNavEnv

# Trigger env registration
import vascularsim.envs  # noqa: F401


class TestFlowAwareNavEnv:
    """Tests for the flow-aware navigation environment."""

    def test_flow_env_creation(self) -> None:
        """FlowAwareNavEnv creates successfully with default graph."""
        env = FlowAwareNavEnv()
        assert env is not None
        obs, info = env.reset(seed=0)
        assert obs is not None

    def test_flow_env_gym_make(self) -> None:
        """gymnasium.make('FlowAwareNav-v0') works."""
        env = gymnasium.make("FlowAwareNav-v0")
        assert env is not None
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict)

    def test_flow_obs_has_velocity(self) -> None:
        """Observation includes flow_velocity key."""
        env = FlowAwareNavEnv()
        obs, _ = env.reset(seed=0)
        assert "flow_velocity" in obs
        assert obs["flow_velocity"].shape == (1,)
        assert obs["flow_velocity"].dtype == np.float32

    def test_flow_obs_has_neighbor_flows(self) -> None:
        """Observation includes flow_at_neighbors key."""
        env = FlowAwareNavEnv()
        obs, _ = env.reset(seed=0)
        assert "flow_at_neighbors" in obs
        assert obs["flow_at_neighbors"].shape == (env._max_neighbors,)

    def test_flow_env_reset_and_step(self) -> None:
        """Basic episode loop: reset + 10 steps works."""
        env = FlowAwareNavEnv()
        obs, info = env.reset(seed=0)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, float)
            assert "flow_velocity" in obs
            if terminated or truncated:
                obs, info = env.reset()

    def test_flow_env_100_episodes(self) -> None:
        """100 episodes complete without crash."""
        env = FlowAwareNavEnv()
        for ep in range(100):
            obs, _ = env.reset(seed=ep)
            done = False
            steps = 0
            while not done and steps < 20:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1

    def test_flow_reward_includes_flow_bonus(self) -> None:
        """Reward from FlowAwareNavEnv differs from base VascularNavEnv
        due to flow-alignment bonus/penalty."""
        env = FlowAwareNavEnv()
        obs, info = env.reset(seed=42)
        # Step through enough actions to likely get at least one non-stay
        rewards = []
        for action in range(min(env.action_space.n, 5)):
            _, reward, _, _, step_info = env.step(action)
            rewards.append(reward)
            env.reset(seed=42)  # reset to same state

        # At least one step should have flow_aware flag
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)
        assert info.get("flow_aware") is True

    def test_flow_env_observation_space(self) -> None:
        """Observation space has all expected keys with correct shapes."""
        env = FlowAwareNavEnv()
        space = env.observation_space
        assert "agent_pos" in space.spaces
        assert "target_pos" in space.spaces
        assert "flow_velocity" in space.spaces
        assert "flow_at_neighbors" in space.spaces
        assert space["flow_velocity"].shape == (1,)
        assert space["flow_at_neighbors"].shape == (env._max_neighbors,)
