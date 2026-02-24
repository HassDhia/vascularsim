"""Magnetic-aware navigation environment.

Extends VascularNavEnv with magnetic field and force observations from
a CoilSystem.  The agent receives a reward bonus when its movement aligns
with the magnetic gradient force direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from vascularsim.envs.vascular_nav import VascularNavEnv
from vascularsim.graph import VascularGraph
from vascularsim.physics.magnetic import CoilSystem, magnetic_force


class MagneticNavEnv(VascularNavEnv):
    """Navigation environment augmented with magnetic field observations.

    Additional observation keys:
        magnetic_field: Box(3,)  -- B-field vector at current position
        magnetic_force: Box(3,)  -- gradient force on microbot

    Reward modification:
        Movement aligned with the magnetic force direction earns a bonus.
    """

    def __init__(
        self,
        graph: VascularGraph | None = None,
        max_steps: int = 500,
        coil_system: CoilSystem | None = None,
    ) -> None:
        super().__init__(graph=graph, max_steps=max_steps)

        self.coil_system = coil_system or CoilSystem.three_axis()

        # Extend observation space
        self.observation_space.spaces["magnetic_field"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32,
        )
        self.observation_space.spaces["magnetic_force"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32,
        )

    def _compute_magnetic_obs(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute B-field and force at a 3-D position."""
        pos_f64 = np.asarray(pos, dtype=np.float64)
        B = self.coil_system.field_at(pos_f64)
        grad = self.coil_system.gradient_at(pos_f64)
        F = magnetic_force(grad)
        return B.astype(np.float32), F.astype(np.float32)

    def _get_obs(self) -> dict[str, Any]:
        obs = super()._get_obs()
        agent_pos = self.graph.get_node_pos(self._agent_node)
        B, F = self._compute_magnetic_obs(agent_pos)
        obs["magnetic_field"] = B
        obs["magnetic_force"] = F
        return obs

    def step(
        self, action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        prev_node = self._agent_node
        prev_pos = self.graph.get_node_pos(prev_node).astype(np.float64)

        obs, reward, terminated, truncated, info = super().step(action)

        # Magnetic alignment bonus
        if self._agent_node != prev_node:
            new_pos = self.graph.get_node_pos(self._agent_node).astype(np.float64)
            movement = new_pos - prev_pos
            move_norm = np.linalg.norm(movement)

            if move_norm > 1e-12:
                _, F = self._compute_magnetic_obs(prev_pos)
                force_norm = np.linalg.norm(F)
                if force_norm > 1e-15:
                    # Cosine similarity between movement and force
                    alignment = float(
                        np.dot(movement, F.astype(np.float64))
                        / (move_norm * force_norm)
                    )
                    # Scale reward by alignment: +0.03 fully aligned, -0.03 opposed
                    reward += 0.03 * alignment

        info["magnetic_aware"] = True
        return obs, reward, terminated, truncated, info
