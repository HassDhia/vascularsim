"""Vascular graph construction from parsed tube data.

Converts a list of Tube objects (centerline + radius) into a
NetworkX DiGraph where nodes carry spatial position and vessel
radius, and edges carry length and mean radius.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from vascularsim.data.tubetk import Tube


class VascularGraph:
    """Directed graph representation of a vascular network.

    Nodes represent sampled centerline points with 3-D position and
    vessel radius.  Edges connect sequential points within a tube and
    link parent tubes to child tubes.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_tubes(cls, tubes: list[Tube]) -> "VascularGraph":
        """Build a VascularGraph from parsed Tube objects.

        Node IDs use the format ``"{tube_id}_{point_idx}"``.

        For parent-child connectivity: if a tube's parent_id >= 0 and
        corresponds to another tube in the list, the parent's last node
        is connected to the child's first node.

        Args:
            tubes: List of Tube dataclass instances.

        Returns:
            Populated VascularGraph.
        """
        vg = cls()
        g = vg._graph

        # Map tube.id -> tube for parent lookups
        tube_map: dict[int, Tube] = {t.id: t for t in tubes}

        for tube in tubes:
            n_pts = tube.points.shape[0]
            for idx in range(n_pts):
                node_id = f"{tube.id}_{idx}"
                pos = tube.points[idx, :3]
                radius = float(tube.points[idx, 3])
                g.add_node(node_id, pos=pos, radius=radius)

                # Edge to previous point in same tube
                if idx > 0:
                    prev_id = f"{tube.id}_{idx - 1}"
                    prev_pos = tube.points[idx - 1, :3]
                    length = float(np.linalg.norm(pos - prev_pos))
                    mean_r = (radius + float(tube.points[idx - 1, 3])) / 2.0
                    g.add_edge(prev_id, node_id, length=length, mean_radius=mean_r)

            # Connect parent -> child
            if tube.parent_id >= 0 and tube.parent_id in tube_map:
                parent_tube = tube_map[tube.parent_id]
                parent_last_idx = parent_tube.points.shape[0] - 1
                parent_node = f"{parent_tube.id}_{parent_last_idx}"
                child_node = f"{tube.id}_0"

                if g.has_node(parent_node) and g.has_node(child_node):
                    p_pos = parent_tube.points[parent_last_idx, :3]
                    c_pos = tube.points[0, :3]
                    length = float(np.linalg.norm(c_pos - p_pos))
                    mean_r = (
                        float(parent_tube.points[parent_last_idx, 3])
                        + float(tube.points[0, 3])
                    ) / 2.0
                    g.add_edge(parent_node, child_node, length=length, mean_radius=mean_r)

        return vg

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def nodes(self) -> list[str]:
        return list(self._graph.nodes)

    def get_node_pos(self, node_id: str) -> np.ndarray:
        """Return the 3-D position of *node_id*."""
        return self._graph.nodes[node_id]["pos"]

    def get_node_radius(self, node_id: str) -> float:
        """Return the vessel radius at *node_id*."""
        return self._graph.nodes[node_id]["radius"]

    def get_neighbors(self, node_id: str) -> list[str]:
        """Return successor node IDs of *node_id*."""
        return list(self._graph.successors(node_id))

    def get_edge_length(self, u: str, v: str) -> float:
        """Return the Euclidean length of edge (u, v)."""
        return self._graph.edges[u, v]["length"]
