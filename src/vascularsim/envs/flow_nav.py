"""Flow-aware navigation environment.

Extends VascularNavEnv with hemodynamic observations (flow velocity at
the current edge and at neighbouring edges) and a flow-alignment reward
bonus: moving WITH the flow earns a bonus, moving AGAINST incurs a penalty.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from vascularsim.envs.vascular_nav import VascularNavEnv
from vascularsim.graph import VascularGraph
from vascularsim.physics.hemodynamics import compute_hemodynamics


class FlowAwareNavEnv(VascularNavEnv):
    """Navigation environment augmented with hemodynamic observations.

    Additional observation keys:
        flow_velocity: Box(1,)  -- flow velocity at current edge
        flow_at_neighbors: Box(max_neighbors,) -- flow at neighbour edges

    Reward modification:
        Moving WITH flow direction earns +0.05 bonus.
        Moving AGAINST flow direction incurs -0.05 penalty.
    """

    def __init__(
        self,
        graph: VascularGraph | None = None,
        max_steps: int = 500,
    ) -> None:
        super().__init__(graph=graph, max_steps=max_steps)

        # Run hemodynamics to populate edges with flow data
        self._hemo_result = compute_hemodynamics(self.graph)

        # Extend observation space
        self.observation_space.spaces["flow_velocity"] = spaces.Box(
            low=0.0, high=np.inf, shape=(1,), dtype=np.float32,
        )
        self.observation_space.spaces["flow_at_neighbors"] = spaces.Box(
            low=0.0, high=np.inf, shape=(self._max_neighbors,), dtype=np.float32,
        )

    def _get_flow_at_node(self, node: str) -> float:
        """Get representative flow velocity at a node.

        Takes the maximum flow velocity of all edges adjacent to the node
        (both incoming and outgoing in the directed graph).
        """
        g = self.graph._graph  # noqa: SLF001
        velocities = []
        for pred in g.predecessors(node):
            v = self._hemo_result.edge_velocities.get((pred, node), 0.0)
            velocities.append(v)
        for succ in g.successors(node):
            v = self._hemo_result.edge_velocities.get((node, succ), 0.0)
            velocities.append(v)
        return max(velocities) if velocities else 0.0

    def _get_edge_flow(self, u: str, v: str) -> float:
        """Get flow velocity for the edge between u and v.

        Checks both directions since the agent traverses an undirected view.
        """
        fwd = self._hemo_result.edge_velocities.get((u, v), 0.0)
        bwd = self._hemo_result.edge_velocities.get((v, u), 0.0)
        return max(fwd, bwd)

    def _get_obs(self) -> dict[str, Any]:
        obs = super()._get_obs()

        # Flow velocity at current node
        flow_vel = self._get_flow_at_node(self._agent_node)
        obs["flow_velocity"] = np.array([flow_vel], dtype=np.float32)

        # Flow velocities at neighbour edges
        neighbours = sorted(self._undirected.neighbors(self._agent_node))
        neighbor_flows = np.zeros(self._max_neighbors, dtype=np.float32)
        for i, nbr in enumerate(neighbours):
            if i >= self._max_neighbors:
                break
            neighbor_flows[i] = self._get_edge_flow(self._agent_node, nbr)
        obs["flow_at_neighbors"] = neighbor_flows

        return obs

    def step(
        self, action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        prev_node = self._agent_node

        obs, reward, terminated, truncated, info = super().step(action)

        # Flow-alignment bonus
        if self._agent_node != prev_node:
            # Agent moved: check if movement aligned with flow
            fwd_flow = self._hemo_result.edge_velocities.get(
                (prev_node, self._agent_node), 0.0
            )
            bwd_flow = self._hemo_result.edge_velocities.get(
                (self._agent_node, prev_node), 0.0
            )
            if fwd_flow > bwd_flow:
                # Moving WITH flow
                reward += 0.05
            elif bwd_flow > fwd_flow:
                # Moving AGAINST flow
                reward -= 0.05

        info["flow_aware"] = True
        return obs, reward, terminated, truncated, info
