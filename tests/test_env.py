"""Tests for the VascularNavEnv Gymnasium environment."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest

# Trigger env registration on import
import vascularsim.envs  # noqa: F401
from vascularsim.envs.vascular_nav import VascularNavEnv


class TestEnvCreation:
    """VascularNavEnv instantiation."""

    def test_env_creation(self) -> None:
        env = VascularNavEnv()
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_gym_make(self) -> None:
        env = gymnasium.make("VascularNav-v0")
        assert env is not None


class TestEnvReset:
    """VascularNavEnv.reset() behaviour."""

    def test_env_reset(self) -> None:
        env = VascularNavEnv()
        obs, info = env.reset(seed=42)

        assert isinstance(obs, dict)
        assert "agent_pos" in obs
        assert "target_pos" in obs
        assert "agent_node" in obs
        assert "distance_to_target" in obs
        assert isinstance(info, dict)

    def test_reset_deterministic_with_seed(self) -> None:
        env = VascularNavEnv()
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1["agent_pos"], obs2["agent_pos"])
        np.testing.assert_array_equal(obs1["target_pos"], obs2["target_pos"])


class TestEnvStep:
    """VascularNavEnv.step() behaviour."""

    def test_env_step(self) -> None:
        env = VascularNavEnv()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_reward_is_finite(self) -> None:
        env = VascularNavEnv()
        env.reset(seed=42)
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            assert np.isfinite(reward)
            if terminated or truncated:
                break


class TestObservationSpace:
    """Observation matches declared observation_space."""

    def test_observation_space(self) -> None:
        env = VascularNavEnv()
        obs, _ = env.reset(seed=42)

        assert env.observation_space.contains(obs), (
            f"Observation {obs} is outside the declared space"
        )

    def test_observation_space_after_step(self) -> None:
        env = VascularNavEnv()
        env.reset(seed=42)
        obs, *_ = env.step(0)
        assert env.observation_space.contains(obs)


class TestActionMasking:
    """Invalid actions result in stay + penalty."""

    def test_invalid_action_stay_penalty(self) -> None:
        env = VascularNavEnv()
        env.reset(seed=42)

        # Use an action index that is definitely out of range
        # (the max action = stay action)
        stay_action = env.action_space.n - 1
        obs_before = env._get_obs()
        obs_after, reward, _, _, info = env.step(stay_action)

        # Agent should not have moved
        np.testing.assert_array_equal(
            obs_before["agent_pos"], obs_after["agent_pos"]
        )
        # Reward includes stay penalty (-0.1) plus time penalty (-0.01)
        assert reward < -0.05, f"Expected negative reward for staying, got {reward}"
        assert info.get("stayed") is True

    def test_very_large_action_is_stay(self) -> None:
        env = VascularNavEnv()
        env.reset(seed=42)
        pos_before = env._get_obs()["agent_pos"].copy()
        # Action well beyond neighbor count but within action space
        action = env.action_space.n - 1
        obs, reward, _, _, info = env.step(action)
        np.testing.assert_array_equal(pos_before, obs["agent_pos"])


class TestRandomAgent:
    """Run 100 episodes with random actions, verify stability."""

    def test_random_agent_100_episodes(self) -> None:
        env = VascularNavEnv()
        terminated_count = 0

        for ep in range(100):
            obs, _ = env.reset(seed=ep)
            assert env.observation_space.contains(obs)

            for step_i in range(env.max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                assert env.observation_space.contains(obs), (
                    f"Episode {ep}, step {step_i}: obs outside space"
                )
                assert np.isfinite(reward)

                if terminated:
                    terminated_count += 1
                    break
                if truncated:
                    break

        # At least some episodes should reach the target with random actions
        # on a small 30-node graph. Even random walks should occasionally
        # stumble onto the target.
        assert terminated_count > 0, (
            f"No episodes terminated (reached target) out of 100 random runs"
        )


class TestDefaultGraph:
    """Verify the default graph structure."""

    def test_default_graph_node_count(self) -> None:
        vg = VascularNavEnv._make_default_graph()
        # 20 trunk + 10 branch = 30 nodes
        assert vg.num_nodes == 30

    def test_default_graph_edge_count(self) -> None:
        vg = VascularNavEnv._make_default_graph()
        # 19 trunk edges + 1 trunk-to-branch + 9 branch edges = 29
        assert vg.num_edges == 29

    def test_default_graph_positions(self) -> None:
        vg = VascularNavEnv._make_default_graph()
        # First trunk node at origin
        pos0 = vg.get_node_pos("0_0")
        np.testing.assert_array_almost_equal(pos0, [0.0, 0.0, 0.0])

        # Last trunk node at x=19
        pos19 = vg.get_node_pos("0_19")
        np.testing.assert_array_almost_equal(pos19, [19.0, 0.0, 0.0])
