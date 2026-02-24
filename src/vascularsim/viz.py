"""3-D visualization of vascular graphs using PyVista.

Renders VascularGraph edges as tube meshes coloured by vessel radius.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from vascularsim.graph import VascularGraph


def render_graph(
    graph: "VascularGraph",
    output_path: str | Path | None = None,
    show: bool = False,
) -> pv.Plotter:
    """Render a VascularGraph as PyVista tube meshes.

    Each edge is drawn as a tube whose start/end radii come from the
    endpoint node radii. Tubes are coloured by average radius using the
    ``coolwarm`` colourmap.

    Args:
        graph: A populated VascularGraph instance.
        output_path: If provided, save a PNG screenshot to this path.
        show: If ``True``, open an interactive 3-D window.

    Returns:
        The PyVista :class:`pyvista.Plotter` used for rendering.
    """
    off_screen = not show
    plotter = pv.Plotter(off_screen=off_screen)

    # Collect per-edge mean radii for colourmap normalisation
    edge_meshes: list[pv.PolyData] = []
    edge_radii: list[float] = []

    internal_graph = graph._graph  # noqa: SLF001 — access internal nx graph
    for u, v in internal_graph.edges():
        pos_u = graph.get_node_pos(u).astype(np.float64)
        pos_v = graph.get_node_pos(v).astype(np.float64)
        r_u = graph.get_node_radius(u)
        r_v = graph.get_node_radius(v)

        # Guard against degenerate zero-length edges
        seg_len = np.linalg.norm(pos_v - pos_u)
        if seg_len < 1e-12:
            continue

        # Create a line between the two points, then tube it
        line = pv.Line(pos_u, pos_v, resolution=1)

        # Use average radius for constant-radius tube (PyVista tube()
        # doesn't natively support linearly-varying radius on a single
        # segment, so average is the practical choice).
        avg_r = (r_u + r_v) / 2.0
        if avg_r < 1e-6:
            avg_r = 0.01  # minimum visible radius

        tube = line.tube(radius=avg_r, n_sides=12)
        edge_meshes.append(tube)
        edge_radii.append(avg_r)

    if not edge_meshes:
        plotter.add_text("Empty graph — no edges to render", font_size=14)
    else:
        # Combine all tubes into a single mesh for efficient rendering
        # Assign scalars for colourmap
        r_min = min(edge_radii)
        r_max = max(edge_radii) if max(edge_radii) > r_min else r_min + 1e-6

        for mesh, avg_r in zip(edge_meshes, edge_radii):
            # Assign uniform scalar to every cell in this tube
            mesh["radius"] = np.full(mesh.n_cells, avg_r)
            plotter.add_mesh(
                mesh,
                scalars="radius",
                cmap="coolwarm",
                clim=[r_min, r_max],
                show_scalar_bar=False,
            )

        # Add a single scalar bar
        plotter.add_scalar_bar(
            title="Vessel Radius",
            n_labels=5,
            fmt="%.3f",
        )

    plotter.add_axes()
    plotter.camera.zoom(1.2)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(output_path), auto_close=not show)
    elif show:
        plotter.show()

    return plotter
