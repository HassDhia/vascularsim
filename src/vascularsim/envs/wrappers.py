"""Observation wrappers for SB3 compatibility.

SB3's MlpPolicy requires a flat Box observation space, but VascularNavEnv
uses a Dict.  FlatNavObservation flattens the dict into a single Box(8,).
"""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces


class FlatNavObservation(gymnasium.ObservationWrapper):
    """Flatten VascularNavEnv's Dict observation into Box(8,).

    Layout: [agent_pos(3), target_pos(3), agent_node_normalized(1), distance_to_target(1)]

    agent_node is normalized to [0, 1] by dividing by the number of nodes.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)

        # Determine number of nodes from the wrapped env's Discrete space
        agent_node_space = env.observation_space["agent_node"]
        self._num_nodes = agent_node_space.n

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

    def observation(self, observation: dict) -> np.ndarray:
        """Flatten dict observation into a single float32 array."""
        agent_pos = np.asarray(observation["agent_pos"], dtype=np.float32)
        target_pos = np.asarray(observation["target_pos"], dtype=np.float32)
        agent_node_norm = np.array(
            [observation["agent_node"] / max(self._num_nodes, 1)],
            dtype=np.float32,
        )
        distance = np.asarray(observation["distance_to_target"], dtype=np.float32)

        return np.concatenate([agent_pos, target_pos, agent_node_norm, distance])
