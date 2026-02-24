"""TubeTK dataset loader and .tre format parser.

TubeTK provides segmented vascular tube data from medical imaging.
The .tre (MetaIO tube) format stores centerline + radius data for
each vessel segment as a "tube group" inside a scene.

Reference:
    https://data.kitware.com/#collection/591086ee8d777f16d01e0724
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

KITWARE_API = "https://data.kitware.com/api/v1"
SAMPLE_ITEM_ID = "58a371ca8d777f0721a64dc8"


@dataclass
class Tube:
    """A single vascular tube with centerline points and radii.

    Attributes:
        id: Unique identifier for this tube within the scene.
        parent_id: ID of the parent tube (-1 if root).
        points: (N, 4) array of [x, y, z, radius] per centerline point.
    """

    id: int
    parent_id: int
    points: np.ndarray  # shape (N, 4) — x, y, z, r


def download_sample(dest_dir: str | Path) -> Path:
    """Download a sample TubeTK .tre file for testing.

    Uses the Kitware Girder API to fetch a Normal-subject Bullitt
    dataset file.  Skips the download if the file already exists.

    Args:
        dest_dir: Directory to save the file into.

    Returns:
        Path to the downloaded .tre file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "sample.tre"

    if dest_path.exists():
        return dest_path

    url = f"{KITWARE_API}/item/{SAMPLE_ITEM_ID}/download"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    dest_path.write_bytes(resp.content)
    return dest_path


def parse_tre(filepath: str | Path) -> list[Tube]:
    """Parse a TubeTK MetaIO .tre file into a list of Tube objects.

    Strategy (robust to minor format variations):
        1. Read all lines, find NObjects for expected group count.
        2. Scan for each "NPoints = N" line.
        3. Look backwards from NPoints to extract ID and ParentID.
        4. Read the next N numeric rows as point data (x, y, z, r ...).

    Args:
        filepath: Path to a .tre file.

    Returns:
        List of Tube objects parsed from the file.
    """
    with open(filepath) as f:
        lines = f.read().strip().split("\n")

    # --- Determine expected group count (informational) ---
    n_objects: int | None = None
    for line in lines:
        if "NObjects" in line and "=" in line:
            n_objects = int(line.split("=")[1].strip())
            break

    tubes: list[Tube] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect a point-count declaration
        if "NPoints" in line and "=" in line:
            try:
                n_points = int(line.split("=")[1].strip())
            except ValueError:
                i += 1
                continue

            # Scan backwards for ID and ParentID in this group header
            tube_id = 0
            parent_id = -1
            for j in range(max(0, i - 25), i):
                hdr = lines[j].strip()
                # Match "ID = <int>" but not "ParentID"
                if hdr.startswith("ID") and "=" in hdr and "Parent" not in hdr:
                    try:
                        tube_id = int(hdr.split("=")[1].strip())
                    except ValueError:
                        pass
                if "ParentID" in hdr and "=" in hdr:
                    try:
                        parent_id = int(hdr.split("=")[1].strip())
                    except ValueError:
                        pass

            # Collect numeric point rows
            points: list[list[float]] = []
            k = i + 1
            while k < len(lines) and len(points) < n_points:
                row = lines[k].strip()
                if row and row[0] in "0123456789.-+":
                    vals = row.split()
                    if len(vals) >= 4:
                        points.append(
                            [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]
                        )
                    elif len(vals) >= 3:
                        # Fallback: some files have no radius column
                        points.append(
                            [float(vals[0]), float(vals[1]), float(vals[2]), 0.0]
                        )
                k += 1

            if points:
                tubes.append(
                    Tube(id=tube_id, parent_id=parent_id, points=np.array(points))
                )
            i = k
        else:
            i += 1

    return tubes
