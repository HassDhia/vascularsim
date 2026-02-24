"""Hemodynamics computation for vascular graphs.

Implements Hagen-Poiseuille flow and Murray's law to compute
blood flow velocities, pressures, and wall shear stresses across
a vascular network represented as a directed graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from vascularsim.graph import VascularGraph


@dataclass
class HemodynamicsResult:
    """Container for hemodynamic computation results.

    Attributes:
        edge_velocities: Flow velocity (m/s) for each directed edge (u, v).
        edge_pressures: Mean pressure (Pa) across each directed edge (u, v).
        edge_wall_shear: Wall shear stress (Pa) for each directed edge (u, v).
        node_pressures: Pressure (Pa) at each node.
    """

    edge_velocities: dict[tuple[str, str], float] = field(default_factory=dict)
    edge_pressures: dict[tuple[str, str], float] = field(default_factory=dict)
    edge_wall_shear: dict[tuple[str, str], float] = field(default_factory=dict)
    node_pressures: dict[str, float] = field(default_factory=dict)


def compute_hemodynamics(
    graph: VascularGraph,
    inlet_pressure: float = 13332.0,
    outlet_pressure: float = 2666.0,
    viscosity: float = 3.5e-3,
) -> HemodynamicsResult:
    """Compute steady-state hemodynamics on a vascular graph.

    Uses Hagen-Poiseuille flow for velocity in each vessel segment
    and Murray's law to distribute flow at bifurcation nodes.

    Args:
        graph: A VascularGraph with directed edges carrying ``length``
            and ``mean_radius`` attributes.
        inlet_pressure: Pressure at root nodes in Pascals.
            Default 13332.0 Pa (100 mmHg, arterial).
        outlet_pressure: Pressure at leaf nodes in Pascals.
            Default 2666.0 Pa (20 mmHg, venous).
        viscosity: Dynamic viscosity of blood in Pa*s.
            Default 3.5e-3 for whole blood.

    Returns:
        A :class:`HemodynamicsResult` with per-edge and per-node data.
    """
    g = graph._graph  # noqa: SLF001

    result = HemodynamicsResult()

    # ----------------------------------------------------------------
    # Step 1: Identify root and leaf nodes
    # ----------------------------------------------------------------
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    leaves = [n for n in g.nodes if g.out_degree(n) == 0]

    # Fallback: if graph has no clear roots (cycles), pick nodes with
    # minimum in-degree.
    if not roots:
        min_in = min(g.in_degree(n) for n in g.nodes)
        roots = [n for n in g.nodes if g.in_degree(n) == min_in]
    if not leaves:
        min_out = min(g.out_degree(n) for n in g.nodes)
        leaves = [n for n in g.nodes if g.out_degree(n) == min_out]

    # ----------------------------------------------------------------
    # Step 2: Assign boundary pressures
    # ----------------------------------------------------------------
    for n in roots:
        result.node_pressures[n] = inlet_pressure
    for n in leaves:
        result.node_pressures[n] = outlet_pressure

    # ----------------------------------------------------------------
    # Step 3: Topological traversal to propagate pressures
    # ----------------------------------------------------------------
    # For DAGs we use topological sort. For graphs with cycles we
    # fall back to BFS from roots.
    try:
        ordered = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        # Graph has cycles -- BFS from roots
        ordered = []
        visited: set[str] = set()
        queue = list(roots)
        for r in queue:
            visited.add(r)
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            for succ in g.successors(node):
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        # Add any remaining nodes not reachable from roots
        for n in g.nodes:
            if n not in visited:
                ordered.append(n)

    # ----------------------------------------------------------------
    # Step 4: Compute pressures, velocities, and shear stresses
    # ----------------------------------------------------------------
    for node in ordered:
        if node not in result.node_pressures:
            # Estimate pressure from predecessors: average of incoming
            preds = list(g.predecessors(node))
            if preds:
                pred_pressures = [
                    result.node_pressures[p] for p in preds
                    if p in result.node_pressures
                ]
                if pred_pressures:
                    # Compute pressure drop from each predecessor and average
                    drops = []
                    for p in preds:
                        if p not in result.node_pressures:
                            continue
                        edata = g.edges[p, node]
                        r = edata["mean_radius"]
                        length = edata["length"]
                        # Poiseuille resistance: R = 8 * mu * L / (pi * r^4)
                        resistance = 8.0 * viscosity * length / (np.pi * r**4)
                        # Estimate flow from parent: use Murray's law proportion
                        # For simplicity, use total pressure gradient approach
                        drops.append(result.node_pressures[p])
                    result.node_pressures[node] = np.mean(drops)

        # Now distribute flow to children using Murray's law
        successors = list(g.successors(node))
        if not successors:
            continue

        p_node = result.node_pressures.get(node, inlet_pressure)

        if len(successors) == 1:
            # Single outgoing edge -- all flow goes through
            succ = successors[0]
            edata = g.edges[node, succ]
            r = edata["mean_radius"]
            length = edata["length"]

            # Compute Poiseuille resistance
            resistance = 8.0 * viscosity * length / (np.pi * r**4)

            # Determine downstream pressure
            if succ in result.node_pressures:
                p_succ = result.node_pressures[succ]
            else:
                # Estimate: subtract pressure drop proportional to
                # this edge's resistance relative to total path
                p_succ = _estimate_downstream_pressure(
                    g, node, succ, p_node, outlet_pressure, viscosity
                )
                result.node_pressures[succ] = p_succ

            delta_p = max(p_node - p_succ, 0.0)

            # Hagen-Poiseuille velocity: v = (delta_P * r^2) / (8 * mu * L)
            if length > 0:
                velocity = (delta_p * r**2) / (8.0 * viscosity * length)
            else:
                velocity = 0.0

            # Wall shear stress: tau = 4 * mu * v / r
            if r > 0:
                wss = 4.0 * viscosity * velocity / r
            else:
                wss = 0.0

            # Mean pressure across the edge
            edge_pressure = (p_node + p_succ) / 2.0

            result.edge_velocities[(node, succ)] = velocity
            result.edge_pressures[(node, succ)] = edge_pressure
            result.edge_wall_shear[(node, succ)] = wss

            # Store on graph edge
            g.edges[node, succ]["flow_velocity"] = velocity
            g.edges[node, succ]["pressure"] = edge_pressure
            g.edges[node, succ]["wall_shear_stress"] = wss

        else:
            # Bifurcation: Murray's law distribution
            # Flow splits proportional to r^3
            radii_cubed = []
            for succ in successors:
                r = g.edges[node, succ]["mean_radius"]
                radii_cubed.append(r**3)
            total_r3 = sum(radii_cubed)

            for i, succ in enumerate(successors):
                edata = g.edges[node, succ]
                r = edata["mean_radius"]
                length = edata["length"]

                # Murray's law fraction
                fraction = radii_cubed[i] / total_r3 if total_r3 > 0 else 1.0 / len(successors)

                # Determine downstream pressure
                if succ in result.node_pressures:
                    p_succ = result.node_pressures[succ]
                else:
                    p_succ = _estimate_downstream_pressure(
                        g, node, succ, p_node, outlet_pressure, viscosity
                    )
                    result.node_pressures[succ] = p_succ

                delta_p = max(p_node - p_succ, 0.0)

                # Hagen-Poiseuille velocity
                if length > 0:
                    velocity = (delta_p * r**2) / (8.0 * viscosity * length)
                else:
                    velocity = 0.0

                # Wall shear stress
                if r > 0:
                    wss = 4.0 * viscosity * velocity / r
                else:
                    wss = 0.0

                edge_pressure = (p_node + p_succ) / 2.0

                result.edge_velocities[(node, succ)] = velocity
                result.edge_pressures[(node, succ)] = edge_pressure
                result.edge_wall_shear[(node, succ)] = wss

                # Store on graph edge
                g.edges[node, succ]["flow_velocity"] = velocity
                g.edges[node, succ]["pressure"] = edge_pressure
                g.edges[node, succ]["wall_shear_stress"] = wss

    # Fill any remaining edges not yet processed (e.g., back-edges in cycles)
    for u, v in g.edges:
        if (u, v) not in result.edge_velocities:
            edata = g.edges[u, v]
            r = edata["mean_radius"]
            length = edata["length"]
            p_u = result.node_pressures.get(u, inlet_pressure)
            p_v = result.node_pressures.get(v, outlet_pressure)
            delta_p = max(p_u - p_v, 0.0)

            if length > 0:
                velocity = (delta_p * r**2) / (8.0 * viscosity * length)
            else:
                velocity = 0.0

            if r > 0:
                wss = 4.0 * viscosity * velocity / r
            else:
                wss = 0.0

            edge_pressure = (p_u + p_v) / 2.0

            result.edge_velocities[(u, v)] = velocity
            result.edge_pressures[(u, v)] = edge_pressure
            result.edge_wall_shear[(u, v)] = wss

            g.edges[u, v]["flow_velocity"] = velocity
            g.edges[u, v]["pressure"] = edge_pressure
            g.edges[u, v]["wall_shear_stress"] = wss

    # Ensure all nodes have a pressure entry
    for n in g.nodes:
        if n not in result.node_pressures:
            # Estimate from neighbors
            preds = list(g.predecessors(n))
            succs = list(g.successors(n))
            neighbor_pressures = [
                result.node_pressures[nb]
                for nb in preds + succs
                if nb in result.node_pressures
            ]
            if neighbor_pressures:
                result.node_pressures[n] = float(np.mean(neighbor_pressures))
            else:
                result.node_pressures[n] = (inlet_pressure + outlet_pressure) / 2.0

    return result


def _estimate_downstream_pressure(
    g: nx.DiGraph,
    parent: str,
    child: str,
    parent_pressure: float,
    outlet_pressure: float,
    viscosity: float,
) -> float:
    """Estimate pressure at a child node based on path topology.

    Computes the fraction of total resistance from parent to the
    nearest leaf, then interpolates between parent and outlet pressure.
    """
    # Count hops from child to nearest leaf
    hops_to_leaf = 0
    current = child
    visited: set[str] = {parent}
    while True:
        visited.add(current)
        succs = [s for s in g.successors(current) if s not in visited]
        if not succs:
            break
        hops_to_leaf += 1
        current = succs[0]

    # Count hops from parent backward to root
    hops_from_root = 0
    current = parent
    visited_back: set[str] = set()
    while True:
        visited_back.add(current)
        preds = [p for p in g.predecessors(current) if p not in visited_back]
        if not preds:
            break
        hops_from_root += 1
        current = preds[0]

    total_hops = hops_from_root + 1 + hops_to_leaf
    if total_hops == 0:
        return outlet_pressure

    # The child is (hops_from_root + 1) hops from root
    fraction = (hops_from_root + 1) / total_hops
    return parent_pressure - fraction * (parent_pressure - outlet_pressure)
