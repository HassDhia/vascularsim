"""Tests for TubeTK parsing and VascularGraph construction.

Uses a synthetic .tre file so tests run offline with no network dependency.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from vascularsim.data.tubetk import Tube, parse_tre
from vascularsim.graph import VascularGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_TRE = textwrap.dedent("""\
    ObjectType = Scene
    NDims = 3
    ID = 0
    Name = Scene
    Comment = ""
    NObjects = 2
    ObjectType = Tube
    NDims = 3
    BinaryData = False
    TransformMatrix = 1 0 0 0 1 0 0 0 1
    Offset = 0 0 0
    CenterOfRotation = 0 0 0
    ElementSpacing = 1 1 1
    ObjectColor = 1 0 0 1
    ID = 1
    ParentID = -1
    Name = Tube1
    NPoints = 4
    Points
    0.0 0.0 0.0 1.0 0 0 0 0
    1.0 0.0 0.0 1.0 0 0 0 0
    2.0 0.0 0.0 1.0 0 0 0 0
    3.0 0.0 0.0 1.0 0 0 0 0
    ObjectType = Tube
    NDims = 3
    BinaryData = False
    TransformMatrix = 1 0 0 0 1 0 0 0 1
    Offset = 0 0 0
    CenterOfRotation = 0 0 0
    ElementSpacing = 1 1 1
    ObjectColor = 0 1 0 1
    ID = 2
    ParentID = 1
    Name = Tube2
    NPoints = 3
    Points
    4.0 1.0 0.0 0.5 0 0 0 0
    5.0 2.0 0.0 0.5 0 0 0 0
    6.0 3.0 0.0 0.5 0 0 0 0
""")


@pytest.fixture
def tre_path(tmp_path: Path) -> Path:
    """Write the synthetic .tre to a temp file and return its path."""
    p = tmp_path / "synthetic.tre"
    p.write_text(SYNTHETIC_TRE)
    return p


@pytest.fixture
def tubes(tre_path: Path) -> list[Tube]:
    return parse_tre(tre_path)


@pytest.fixture
def graph(tubes: list[Tube]) -> VascularGraph:
    return VascularGraph.from_tubes(tubes)


# ---------------------------------------------------------------------------
# parse_tre tests
# ---------------------------------------------------------------------------

class TestParseTre:
    def test_returns_correct_tube_count(self, tubes: list[Tube]) -> None:
        assert len(tubes) == 2

    def test_tube_ids(self, tubes: list[Tube]) -> None:
        assert tubes[0].id == 1
        assert tubes[1].id == 2

    def test_parent_ids(self, tubes: list[Tube]) -> None:
        assert tubes[0].parent_id == -1
        assert tubes[1].parent_id == 1

    def test_point_shapes(self, tubes: list[Tube]) -> None:
        assert tubes[0].points.shape == (4, 4)
        assert tubes[1].points.shape == (3, 4)

    def test_point_values(self, tubes: list[Tube]) -> None:
        # First tube, first point: x=0, y=0, z=0, r=1.0
        np.testing.assert_allclose(tubes[0].points[0], [0.0, 0.0, 0.0, 1.0])
        # Second tube, last point: x=6, y=3, z=0, r=0.5
        np.testing.assert_allclose(tubes[1].points[-1], [6.0, 3.0, 0.0, 0.5])


# ---------------------------------------------------------------------------
# VascularGraph tests
# ---------------------------------------------------------------------------

class TestVascularGraph:
    def test_node_count(self, graph: VascularGraph) -> None:
        # 4 points + 3 points = 7 nodes
        assert graph.num_nodes == 7

    def test_edge_count(self, graph: VascularGraph) -> None:
        # 3 intra-tube1 + 2 intra-tube2 + 1 parent-child = 6
        assert graph.num_edges == 6

    def test_node_list(self, graph: VascularGraph) -> None:
        expected = {"1_0", "1_1", "1_2", "1_3", "2_0", "2_1", "2_2"}
        assert set(graph.nodes) == expected

    def test_node_position(self, graph: VascularGraph) -> None:
        pos = graph.get_node_pos("1_0")
        np.testing.assert_allclose(pos, [0.0, 0.0, 0.0])

    def test_node_radius(self, graph: VascularGraph) -> None:
        assert graph.get_node_radius("1_0") == pytest.approx(1.0)
        assert graph.get_node_radius("2_0") == pytest.approx(0.5)

    def test_intra_tube_edge_length(self, graph: VascularGraph) -> None:
        # Tube 1: consecutive x-coords differ by 1.0
        length = graph.get_edge_length("1_0", "1_1")
        assert length == pytest.approx(1.0)

    def test_inter_tube_edge_length(self, graph: VascularGraph) -> None:
        # Parent last node 1_3 at (3,0,0) -> child first node 2_0 at (4,1,0)
        length = graph.get_edge_length("1_3", "2_0")
        expected = np.sqrt(1.0**2 + 1.0**2)  # sqrt(2)
        assert length == pytest.approx(expected)

    def test_parent_child_connectivity(self, graph: VascularGraph) -> None:
        neighbors = graph.get_neighbors("1_3")
        assert "2_0" in neighbors

    def test_neighbors_within_tube(self, graph: VascularGraph) -> None:
        neighbors = graph.get_neighbors("1_1")
        assert "1_2" in neighbors

    def test_get_neighbors_leaf_node(self, graph: VascularGraph) -> None:
        # Last node of child tube has no successors
        neighbors = graph.get_neighbors("2_2")
        assert neighbors == []


# ---------------------------------------------------------------------------
# Stats output (runs as test so we see it in pytest -v output)
# ---------------------------------------------------------------------------

class TestStats:
    def test_print_stats(self, graph: VascularGraph, capsys: pytest.CaptureFixture[str]) -> None:
        print(f"\n--- VascularGraph Stats ---")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.num_edges}")
        print(f"  Node IDs: {sorted(graph.nodes)}")
        assert graph.num_nodes > 0
