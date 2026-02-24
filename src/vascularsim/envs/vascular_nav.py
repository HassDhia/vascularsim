"""Gymnasium environment for microbot navigation through vascular networks.

The agent moves node-to-node along a VascularGraph, rewarded for reaching
a randomly chosen target node within a step budget.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import networkx as nx
import numpy as np
from gymnasium import spaces

from vascularsim.graph import VascularGraph


class VascularNavEnv(gymnasium.Env):
    """RL environment for navigating a vascular graph.

    Observation space (Dict):
        agent_pos: Box(3,)           — 3-D position of the agent
        target_pos: Box(3,)          — 3-D position of the target
        agent_node: Discrete(N)      — current node index
        distance_to_target: Box(1,)  — Euclidean distance to target

    Action space: Discrete(max_degree + 1)
        Actions 0..max_degree-1 select a neighbour from the sorted
        neighbour list.  The last action index means "stay in place".
        Invalid actions (index >= len(neighbours)) also result in
        staying with a small penalty.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        graph: VascularGraph | None = None,
        max_steps: int = 500,
    ) -> None:
        super().__init__()

        if graph is None:
            graph = self._make_default_graph()

        self.graph = graph
        self.max_steps = max_steps

        # Ordered list of node ids for index <-> id mapping
        self._node_list: list[str] = sorted(self.graph.nodes)
        self._node_to_idx: dict[str, int] = {
            nid: i for i, nid in enumerate(self._node_list)
        }
        n_nodes = len(self._node_list)

        # Build an undirected view so the agent can traverse edges
        # in both directions (vessels are physically bidirectional).
        self._undirected: nx.Graph = self.graph._graph.to_undirected()  # noqa: SLF001

        # Max degree across all nodes (undirected)
        max_degree = max(dict(self._undirected.degree()).values()) if n_nodes > 0 else 1
        # +1 for the explicit "stay" action
        self._max_neighbors = max_degree
        n_actions = max_degree + 1

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "target_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "agent_node": spaces.Discrete(n_nodes),
                "distance_to_target": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

        # Episode state (set properly in reset)
        self._agent_node: str = self._node_list[0]
        self._target_node: str = self._node_list[-1]
        self._steps: int = 0
        self._initial_distance: float = 1.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)

        rng = self.np_random
        n = len(self._node_list)

        # Try to pick two nodes at least 5 hops apart
        start_idx: int | None = None
        target_idx: int | None = None
        MIN_HOPS = 5

        for _ in range(50):  # 50 random attempts
            s = int(rng.integers(0, n))
            t = int(rng.integers(0, n))
            if s == t:
                continue
            try:
                sp_len = nx.shortest_path_length(
                    self._undirected,
                    self._node_list[s],
                    self._node_list[t],
                )
                if sp_len >= MIN_HOPS:
                    start_idx, target_idx = s, t
                    break
            except nx.NetworkXNoPath:
                continue

        # Fallback: any two distinct nodes
        if start_idx is None or target_idx is None:
            start_idx = int(rng.integers(0, n))
            target_idx = (start_idx + n // 2) % n  # spread apart
            if target_idx == start_idx:
                target_idx = (start_idx + 1) % n

        self._agent_node = self._node_list[start_idx]
        self._target_node = self._node_list[target_idx]
        self._steps = 0
        self._initial_distance = float(
            np.linalg.norm(
                self.graph.get_node_pos(self._agent_node)
                - self.graph.get_node_pos(self._target_node)
            )
        )
        if self._initial_distance < 1e-9:
            self._initial_distance = 1.0

        obs = self._get_obs()
        info: dict[str, Any] = {"start_node": self._agent_node, "target_node": self._target_node}
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._steps += 1

        prev_pos = self.graph.get_node_pos(self._agent_node).astype(np.float64)
        prev_distance = float(
            np.linalg.norm(
                prev_pos - self.graph.get_node_pos(self._target_node).astype(np.float64)
            )
        )

        # Resolve action to neighbour
        neighbours = sorted(self._undirected.neighbors(self._agent_node))
        stay_action = self._max_neighbors  # last valid action index

        stayed = False
        if action == stay_action or action >= len(neighbours):
            # Invalid or explicit stay
            stayed = True
        else:
            self._agent_node = neighbours[action]

        # Compute reward
        new_pos = self.graph.get_node_pos(self._agent_node).astype(np.float64)
        target_pos = self.graph.get_node_pos(self._target_node).astype(np.float64)
        new_distance = float(np.linalg.norm(new_pos - target_pos))

        reward = -0.01  # time penalty

        if stayed:
            reward += -0.1  # stay penalty
        else:
            # Normalised progress reward
            progress = (prev_distance - new_distance) / self._initial_distance
            reward += progress * 1.0

        terminated = False
        if self._agent_node == self._target_node:
            reward += 10.0
            terminated = True

        truncated = self._steps >= self.max_steps

        obs = self._get_obs()
        info: dict[str, Any] = {
            "steps": self._steps,
            "distance_to_target": new_distance,
            "stayed": stayed,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict[str, Any]:
        agent_pos = self.graph.get_node_pos(self._agent_node).astype(np.float32)
        target_pos = self.graph.get_node_pos(self._target_node).astype(np.float32)
        distance = float(np.linalg.norm(agent_pos - target_pos))
        return {
            "agent_pos": agent_pos,
            "target_pos": target_pos,
            "agent_node": self._node_to_idx[self._agent_node],
            "distance_to_target": np.array([distance], dtype=np.float32),
        }

    @staticmethod
    def _make_default_graph() -> VascularGraph:
        """Create a simple test graph: 20-node trunk + 10-node branch.

        Main trunk runs along the x-axis with radius 0.5.
        Branch diverges from node 10 at 45 degrees with radius 0.3.
        """
        vg = VascularGraph()
        g = vg._graph  # noqa: SLF001

        # Main trunk: 20 nodes along x-axis
        for i in range(20):
            node_id = f"0_{i}"
            g.add_node(node_id, pos=np.array([float(i), 0.0, 0.0]), radius=0.5)
            if i > 0:
                prev_id = f"0_{i - 1}"
                g.add_edge(prev_id, node_id, length=1.0, mean_radius=0.5)

        # Branch: 10 nodes diverging from node 10 at 45 degrees
        branch_start_pos = np.array([10.0, 0.0, 0.0])
        direction = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
        for i in range(10):
            node_id = f"1_{i}"
            pos = branch_start_pos + direction * (i + 1)
            g.add_node(node_id, pos=pos, radius=0.3)
            if i == 0:
                # Connect branch start to main trunk node 10
                g.add_edge("0_10", node_id, length=1.0, mean_radius=0.4)
            else:
                prev_id = f"1_{i - 1}"
                g.add_edge(prev_id, node_id, length=1.0, mean_radius=0.3)

        return vg
