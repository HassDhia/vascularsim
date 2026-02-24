"""Tests for VascularSim benchmark environments and runner."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from vascularsim.benchmarks.environments import TIER_NAMES, make_benchmark_graph
from vascularsim.benchmarks.runner import run_benchmark, save_results
from vascularsim.envs.vascular_nav import VascularNavEnv
from vascularsim.envs.wrappers import FlatNavObservation


class TestTier1Straight:
    """Tier 1: straight vessel has 20 nodes and 19 edges."""

    def test_tier_1_straight(self) -> None:
        vg = make_benchmark_graph(1)
        assert vg.num_nodes == 20
        assert vg.num_edges == 19

    def test_tier_1_node_ids(self) -> None:
        vg = make_benchmark_graph(1)
        for i in range(20):
            assert f"0_{i}" in vg.nodes

    def test_tier_1_positions_linear(self) -> None:
        vg = make_benchmark_graph(1)
        for i in range(20):
            pos = vg.get_node_pos(f"0_{i}")
            np.testing.assert_allclose(pos, [float(i), 0.0, 0.0])


class TestTier2Bifurcation:
    """Tier 2: single bifurcation has ~35 nodes."""

    def test_tier_2_bifurcation(self) -> None:
        vg = make_benchmark_graph(2)
        # 15 trunk + 10 branch A + 10 branch B = 35
        assert vg.num_nodes == 35

    def test_tier_2_has_branches(self) -> None:
        vg = make_benchmark_graph(2)
        # Node 0_7 should have outgoing edges to both branches
        neighbors = vg.get_neighbors("0_7")
        # Should connect to 0_8 (trunk), 1_0 (branch A), 2_0 (branch B)
        assert len(neighbors) == 3


class TestTier3Tree:
    """Tier 3: vascular tree has ~50 nodes."""

    def test_tier_3_tree(self) -> None:
        vg = make_benchmark_graph(3)
        # 10 root + 2*8 level1 + 4*6 level2 = 10 + 16 + 24 = 50
        assert vg.num_nodes == 50

    def test_tier_3_is_tree_structure(self) -> None:
        vg = make_benchmark_graph(3)
        # A tree with N nodes has N-1 edges (+ inter-segment connections)
        # Root: 9 edges, L1: 2*(7+1) = 16, L2: 4*(5+1) = 24 => total = 9 + 16 + 24 = 49
        assert vg.num_edges == 49


class TestTier4Ring:
    """Tier 4: ring network has loops."""

    def test_tier_4_ring(self) -> None:
        vg = make_benchmark_graph(4)
        # 15 trunk + 10 upper + 10 lower + 8 dead end = 43
        assert vg.num_nodes == 43

    def test_tier_4_has_cycles(self) -> None:
        vg = make_benchmark_graph(4)
        # The undirected version should contain at least one cycle
        undirected = vg._graph.to_undirected()
        cycles = list(nx.cycle_basis(undirected))
        assert len(cycles) > 0, "Tier 4 ring network should contain cycles"


class TestTier5Dense:
    """Tier 5: dense mesh has 100+ nodes."""

    def test_tier_5_dense(self) -> None:
        vg = make_benchmark_graph(5)
        assert vg.num_nodes >= 100, (
            f"Tier 5 should have 100+ nodes, got {vg.num_nodes}"
        )

    def test_tier_5_has_variety(self) -> None:
        vg = make_benchmark_graph(5)
        # Should have multiple edge types / radii
        radii = set()
        for node_id in vg.nodes:
            radii.add(round(vg.get_node_radius(node_id), 2))
        assert len(radii) > 3, "Tier 5 should have varied radii"


class TestAllTiersCreateValidEnvs:
    """Each tier's graph works with VascularNavEnv."""

    @pytest.mark.parametrize("tier", [1, 2, 3, 4, 5])
    def test_all_tiers_create_valid_envs(self, tier: int) -> None:
        graph = make_benchmark_graph(tier)
        env = VascularNavEnv(graph=graph, max_steps=500)
        obs, info = env.reset(seed=42)

        assert env.observation_space.contains(obs)
        assert "start_node" in info
        assert "target_node" in info

        # Take a step to verify env works
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert env.observation_space.contains(obs2)
        assert np.isfinite(reward)

    @pytest.mark.parametrize("tier", [1, 2, 3, 4, 5])
    def test_all_tiers_work_with_wrapper(self, tier: int) -> None:
        graph = make_benchmark_graph(tier)
        base_env = VascularNavEnv(graph=graph, max_steps=500)
        env = FlatNavObservation(base_env)
        obs, _ = env.reset(seed=42)

        assert obs.shape == (8,)
        assert obs.dtype == np.float32


class TestBenchmarkRunnerRandom:
    """Benchmark runner works with a random agent on tier 1."""

    def test_benchmark_runner_random(self) -> None:
        rng = np.random.default_rng(123)

        def random_agent(obs: np.ndarray) -> int:
            return int(rng.integers(0, 5))

        random_agent.__name__ = "test_random"

        results = run_benchmark(
            agent_fn=random_agent,
            tiers=(1,),
            n_episodes=5,
            seed=42,
        )

        assert results["agent_name"] == "test_random"
        assert results["n_episodes"] == 5
        assert "1" in results["tiers"]

        tier1 = results["tiers"]["1"]
        assert 0.0 <= tier1["success_rate"] <= 1.0
        assert tier1["mean_steps"] > 0
        assert np.isfinite(tier1["mean_reward"])
        assert tier1["mean_time_per_step"] > 0


class TestResultsStructure:
    """Results dict has expected keys."""

    def test_results_structure(self) -> None:
        rng = np.random.default_rng(0)

        def dummy(obs: np.ndarray) -> int:
            return int(rng.integers(0, 3))

        dummy.__name__ = "dummy"

        results = run_benchmark(agent_fn=dummy, tiers=(1,), n_episodes=3, seed=0)

        # Top-level keys
        assert "agent_name" in results
        assert "timestamp" in results
        assert "seed" in results
        assert "n_episodes" in results
        assert "tiers" in results

        # Tier-level keys
        tier = results["tiers"]["1"]
        assert "name" in tier
        assert "success_rate" in tier
        assert "mean_steps" in tier
        assert "mean_reward" in tier
        assert "mean_time_per_step" in tier

    def test_save_results(self) -> None:
        results = {
            "agent_name": "test",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "seed": 42,
            "n_episodes": 1,
            "tiers": {"1": {"name": "straight", "success_rate": 0.5,
                            "mean_steps": 10, "mean_reward": 1.0,
                            "mean_time_per_step": 0.001}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "subdir" / "results.json")
            save_results(results, path)
            loaded = json.loads(Path(path).read_text())
            assert loaded["agent_name"] == "test"
            assert loaded["tiers"]["1"]["success_rate"] == 0.5


class TestReproducibleSeeds:
    """Same seed produces identical graphs."""

    @pytest.mark.parametrize("tier", [1, 2, 3, 4, 5])
    def test_reproducible_seeds(self, tier: int) -> None:
        g1 = make_benchmark_graph(tier, seed=999)
        g2 = make_benchmark_graph(tier, seed=999)

        assert g1.num_nodes == g2.num_nodes
        assert g1.num_edges == g2.num_edges

        # Verify positions are identical
        for node_id in g1.nodes:
            np.testing.assert_array_equal(
                g1.get_node_pos(node_id),
                g2.get_node_pos(node_id),
            )

    def test_different_seeds_differ(self) -> None:
        # Tiers 3+ use randomness, so different seeds should produce different graphs
        g1 = make_benchmark_graph(3, seed=1)
        g2 = make_benchmark_graph(3, seed=2)

        # Node positions should differ (different random angles)
        positions_differ = False
        for node_id in g1.nodes:
            if node_id in g2.nodes:
                p1 = g1.get_node_pos(node_id)
                p2 = g2.get_node_pos(node_id)
                if not np.allclose(p1, p2):
                    positions_differ = True
                    break
        assert positions_differ, "Different seeds should produce different graphs"
