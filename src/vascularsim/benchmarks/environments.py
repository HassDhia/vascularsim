"""Benchmark environment factory with 5 difficulty tiers.

Each tier produces a deterministic VascularGraph of increasing complexity,
from a trivial straight vessel to a dense mesh with loops and dead ends.
"""

from __future__ import annotations

import numpy as np

from vascularsim.graph import VascularGraph

TIER_NAMES: dict[int, str] = {
    1: "straight",
    2: "bifurcation",
    3: "tree",
    4: "ring",
    5: "dense_mesh",
}


def _add_segment(
    vg: VascularGraph,
    tube_id: int,
    start_pos: np.ndarray,
    direction: np.ndarray,
    n_points: int,
    spacing: float,
    radius: float,
    connect_from: str | None = None,
) -> list[str]:
    """Add a linear segment of nodes to the graph.

    Returns the list of node IDs created.
    """
    g = vg._graph  # noqa: SLF001
    node_ids: list[str] = []

    for i in range(n_points):
        node_id = f"{tube_id}_{i}"
        pos = start_pos + direction * spacing * i
        g.add_node(node_id, pos=pos.copy(), radius=radius)
        node_ids.append(node_id)

        if i > 0:
            prev_id = f"{tube_id}_{i - 1}"
            prev_pos = g.nodes[prev_id]["pos"]
            length = float(np.linalg.norm(pos - prev_pos))
            mean_r = (radius + g.nodes[prev_id]["radius"]) / 2.0
            g.add_edge(prev_id, node_id, length=length, mean_radius=mean_r)

    # Connect from a parent node to the first node of this segment
    if connect_from is not None and len(node_ids) > 0:
        parent_pos = g.nodes[connect_from]["pos"]
        child_pos = g.nodes[node_ids[0]]["pos"]
        length = float(np.linalg.norm(child_pos - parent_pos))
        parent_r = g.nodes[connect_from]["radius"]
        child_r = g.nodes[node_ids[0]]["radius"]
        g.add_edge(connect_from, node_ids[0], length=length, mean_radius=(parent_r + child_r) / 2.0)

    return node_ids


def _direction_from_angle(angle_deg: float) -> np.ndarray:
    """Return a 3D unit direction vector from an angle in degrees (XY plane)."""
    rad = np.radians(angle_deg)
    return np.array([np.cos(rad), np.sin(rad), 0.0])


def _make_tier1(seed: int) -> VascularGraph:
    """Tier 1: Straight vessel -- 20 nodes in a line along x-axis."""
    vg = VascularGraph()
    _add_segment(vg, tube_id=0, start_pos=np.array([0.0, 0.0, 0.0]),
                 direction=np.array([1.0, 0.0, 0.0]), n_points=20,
                 spacing=1.0, radius=0.5)
    return vg


def _make_tier2(seed: int) -> VascularGraph:
    """Tier 2: Single bifurcation -- main trunk + two branches from node 7."""
    vg = VascularGraph()
    g = vg._graph  # noqa: SLF001

    # Main trunk: 15 nodes along x-axis
    _add_segment(vg, tube_id=0, start_pos=np.array([0.0, 0.0, 0.0]),
                 direction=np.array([1.0, 0.0, 0.0]), n_points=15,
                 spacing=1.0, radius=0.5)

    # Branch A: 10 nodes from node 0_7, going up-right at 30 degrees
    branch_a_start = g.nodes["0_7"]["pos"].copy()
    _add_segment(vg, tube_id=1,
                 start_pos=branch_a_start + _direction_from_angle(30.0) * 1.0,
                 direction=_direction_from_angle(30.0), n_points=10,
                 spacing=1.0, radius=0.4, connect_from="0_7")

    # Branch B: 10 nodes from node 0_7, going down-right at -30 degrees
    branch_b_start = g.nodes["0_7"]["pos"].copy()
    _add_segment(vg, tube_id=2,
                 start_pos=branch_b_start + _direction_from_angle(-30.0) * 1.0,
                 direction=_direction_from_angle(-30.0), n_points=10,
                 spacing=1.0, radius=0.4, connect_from="0_7")

    return vg


def _make_tier3(seed: int) -> VascularGraph:
    """Tier 3: Vascular tree -- 3-level binary tree with random angles."""
    rng = np.random.default_rng(seed)
    vg = VascularGraph()
    g = vg._graph  # noqa: SLF001

    tube_counter = 0

    # Root trunk: 10 nodes along x-axis
    _add_segment(vg, tube_id=tube_counter, start_pos=np.array([0.0, 0.0, 0.0]),
                 direction=np.array([1.0, 0.0, 0.0]), n_points=10,
                 spacing=1.0, radius=0.5)
    root_end = f"{tube_counter}_9"
    tube_counter += 1

    # Level 1: two branches from root end
    level1_ends: list[str] = []
    for i in range(2):
        angle = rng.uniform(20, 50) * (1 if i == 0 else -1)
        direction = _direction_from_angle(angle)
        parent_pos = g.nodes[root_end]["pos"].copy()
        _add_segment(vg, tube_id=tube_counter,
                     start_pos=parent_pos + direction * 1.0,
                     direction=direction, n_points=8,
                     spacing=1.0, radius=0.4, connect_from=root_end)
        level1_ends.append(f"{tube_counter}_7")
        tube_counter += 1

    # Level 2: two sub-branches from each level-1 end
    for parent_end in level1_ends:
        parent_pos = g.nodes[parent_end]["pos"].copy()
        for i in range(2):
            angle = rng.uniform(15, 60) * (1 if i == 0 else -1)
            direction = _direction_from_angle(angle)
            _add_segment(vg, tube_id=tube_counter,
                         start_pos=parent_pos + direction * 1.0,
                         direction=direction, n_points=6,
                         spacing=1.0, radius=0.3, connect_from=parent_end)
            tube_counter += 1

    return vg


def _make_tier4(seed: int) -> VascularGraph:
    """Tier 4: Ring network -- trunk with a loop plus dead-end branch."""
    vg = VascularGraph()
    g = vg._graph  # noqa: SLF001

    # Main trunk: 15 nodes along x-axis
    _add_segment(vg, tube_id=0, start_pos=np.array([0.0, 0.0, 0.0]),
                 direction=np.array([1.0, 0.0, 0.0]), n_points=15,
                 spacing=1.0, radius=0.5)

    # Upper branch from node 0_5 to a point above, then curving back to 0_12
    upper_start = g.nodes["0_5"]["pos"].copy()
    upper_end_target = g.nodes["0_12"]["pos"].copy()

    # Create upper arc: 10 nodes
    tube_id = 1
    for i in range(10):
        node_id = f"{tube_id}_{i}"
        t = (i + 1) / 11.0  # interpolation parameter
        # Arc above the trunk
        x = upper_start[0] + (upper_end_target[0] - upper_start[0]) * t
        y = 3.0 * np.sin(np.pi * t)  # arc height
        pos = np.array([x, y, 0.0])
        g.add_node(node_id, pos=pos, radius=0.4)

        if i > 0:
            prev_id = f"{tube_id}_{i - 1}"
            prev_pos = g.nodes[prev_id]["pos"]
            length = float(np.linalg.norm(pos - prev_pos))
            g.add_edge(prev_id, node_id, length=length, mean_radius=0.4)

    # Connect 0_5 -> 1_0
    child_pos = g.nodes["1_0"]["pos"]
    length = float(np.linalg.norm(child_pos - upper_start))
    g.add_edge("0_5", "1_0", length=length, mean_radius=0.45)

    # Connect 1_9 -> 0_12 (closes the loop)
    last_upper = g.nodes["1_9"]["pos"]
    length = float(np.linalg.norm(upper_end_target - last_upper))
    g.add_edge("1_9", "0_12", length=length, mean_radius=0.45)

    # Lower branch from node 0_5 (parallel path, also reconnects to 0_12)
    tube_id = 2
    for i in range(10):
        node_id = f"{tube_id}_{i}"
        t = (i + 1) / 11.0
        x = upper_start[0] + (upper_end_target[0] - upper_start[0]) * t
        y = -3.0 * np.sin(np.pi * t)  # arc below
        pos = np.array([x, y, 0.0])
        g.add_node(node_id, pos=pos, radius=0.4)

        if i > 0:
            prev_id = f"{tube_id}_{i - 1}"
            prev_pos = g.nodes[prev_id]["pos"]
            length = float(np.linalg.norm(pos - prev_pos))
            g.add_edge(prev_id, node_id, length=length, mean_radius=0.4)

    child_pos = g.nodes["2_0"]["pos"]
    length = float(np.linalg.norm(child_pos - upper_start))
    g.add_edge("0_5", "2_0", length=length, mean_radius=0.45)

    last_lower = g.nodes["2_9"]["pos"]
    length = float(np.linalg.norm(upper_end_target - last_lower))
    g.add_edge("2_9", "0_12", length=length, mean_radius=0.45)

    # Dead-end branch off the upper ring at node 1_5
    ring_branch_start = g.nodes["1_5"]["pos"].copy()
    _add_segment(vg, tube_id=3,
                 start_pos=ring_branch_start + np.array([0.0, 1.5, 0.0]),
                 direction=np.array([0.3, 1.0, 0.0]) / np.linalg.norm([0.3, 1.0, 0.0]),
                 n_points=8, spacing=1.0, radius=0.3, connect_from="1_5")

    return vg


def _make_tier5(seed: int) -> VascularGraph:
    """Tier 5: Dense mesh -- 100+ nodes with branches, loops, and dead ends."""
    rng = np.random.default_rng(seed)
    vg = VascularGraph()
    g = vg._graph  # noqa: SLF001

    tube_counter = 0

    # Main trunk: 20 nodes
    trunk_nodes = _add_segment(
        vg, tube_id=tube_counter, start_pos=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]), n_points=20,
        spacing=1.0, radius=0.6)
    tube_counter += 1

    # Track all branch endpoints for potential loop connections
    branch_ends: list[str] = []
    all_segments: list[list[str]] = [trunk_nodes]

    # Spawn 8 primary branches at random points along the trunk
    branch_points = sorted(rng.choice(range(3, 18), size=8, replace=False))
    for bp in branch_points:
        parent_node = f"0_{bp}"
        parent_pos = g.nodes[parent_node]["pos"].copy()
        n_points = int(rng.integers(8, 16))
        angle = float(rng.uniform(-80, 80))
        direction = _direction_from_angle(angle)
        radius = float(rng.uniform(0.2, 0.5))

        nodes = _add_segment(
            vg, tube_id=tube_counter,
            start_pos=parent_pos + direction * 1.0,
            direction=direction, n_points=n_points,
            spacing=float(rng.uniform(0.8, 1.2)), radius=radius,
            connect_from=parent_node)
        all_segments.append(nodes)
        branch_ends.append(nodes[-1])
        tube_counter += 1

        # Sub-branches (50% chance for each primary branch)
        if rng.random() < 0.5 and len(nodes) > 4:
            sub_parent_idx = int(rng.integers(2, len(nodes) - 1))
            sub_parent = nodes[sub_parent_idx]
            sub_parent_pos = g.nodes[sub_parent]["pos"].copy()
            sub_angle = float(rng.uniform(-60, 60))
            sub_dir = _direction_from_angle(sub_angle)
            sub_n = int(rng.integers(5, 10))
            sub_radius = float(rng.uniform(0.2, 0.4))

            sub_nodes = _add_segment(
                vg, tube_id=tube_counter,
                start_pos=sub_parent_pos + sub_dir * 0.8,
                direction=sub_dir, n_points=sub_n,
                spacing=float(rng.uniform(0.7, 1.0)), radius=sub_radius,
                connect_from=sub_parent)
            all_segments.append(sub_nodes)
            branch_ends.append(sub_nodes[-1])
            tube_counter += 1

    # Create loops: connect some branch endpoints to nodes on other segments
    # Pick 3-4 loop connections
    n_loops = int(rng.integers(3, 5))
    if len(branch_ends) >= 2:
        loop_sources = list(rng.choice(branch_ends, size=min(n_loops, len(branch_ends)), replace=False))
        for src in loop_sources:
            src_pos = g.nodes[src]["pos"]
            # Find a node on a different segment that is reasonably close
            best_target = None
            best_dist = float("inf")
            for seg in all_segments:
                for candidate in seg:
                    if candidate == src:
                        continue
                    # Don't connect to nodes on the same tube
                    src_tube = src.split("_")[0]
                    cand_tube = candidate.split("_")[0]
                    if src_tube == cand_tube:
                        continue
                    d = float(np.linalg.norm(g.nodes[candidate]["pos"] - src_pos))
                    if 2.0 < d < 8.0 and d < best_dist:
                        best_dist = d
                        best_target = candidate

            if best_target is not None:
                src_r = g.nodes[src]["radius"]
                tgt_r = g.nodes[best_target]["radius"]
                g.add_edge(src, best_target, length=best_dist,
                           mean_radius=(src_r + tgt_r) / 2.0)

    # Add a few dead-end stubs for extra complexity
    for _ in range(3):
        # Pick a random existing node
        existing_nodes = list(g.nodes)
        parent = existing_nodes[int(rng.integers(0, len(existing_nodes)))]
        parent_pos = g.nodes[parent]["pos"].copy()
        angle = float(rng.uniform(0, 360))
        direction = _direction_from_angle(angle)
        stub_n = int(rng.integers(3, 6))
        stub_radius = float(rng.uniform(0.2, 0.35))

        _add_segment(
            vg, tube_id=tube_counter,
            start_pos=parent_pos + direction * 0.5,
            direction=direction, n_points=stub_n,
            spacing=float(rng.uniform(0.5, 0.8)), radius=stub_radius,
            connect_from=parent)
        tube_counter += 1

    return vg


_TIER_FACTORIES = {
    1: _make_tier1,
    2: _make_tier2,
    3: _make_tier3,
    4: _make_tier4,
    5: _make_tier5,
}


def make_benchmark_graph(tier: int, seed: int = 42) -> VascularGraph:
    """Create a benchmark vascular graph for the given difficulty tier.

    Args:
        tier: Difficulty tier (1-5).
        seed: Random seed for reproducibility.

    Returns:
        A VascularGraph of the specified complexity.

    Raises:
        ValueError: If tier is not in 1-5.
    """
    if tier not in _TIER_FACTORIES:
        raise ValueError(f"Unknown tier {tier}. Valid tiers: {sorted(_TIER_FACTORIES)}")
    return _TIER_FACTORIES[tier](seed)
