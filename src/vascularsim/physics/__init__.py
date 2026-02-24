"""Physics modules for vascular microbot simulation.

Exports:
    compute_hemodynamics: Compute flow velocities, pressures, and
        wall shear stresses on a VascularGraph.
    HemodynamicsResult: Dataclass holding per-edge and per-node results.
    HelmholtzCoil: Single Helmholtz coil pair model.
    CoilSystem: Multi-coil superposition system.
    magnetic_force: Gradient force on a magnetic dipole.
    magnetic_torque: Torque on a magnetic dipole in a field.
    FlowSurrogate: Neural MLP surrogate for fast flow prediction.
"""

from vascularsim.physics.hemodynamics import (
    HemodynamicsResult,
    compute_hemodynamics,
)
from vascularsim.physics.magnetic import (
    HelmholtzCoil,
    CoilSystem,
    magnetic_force,
    magnetic_torque,
)
from vascularsim.physics.surrogate import FlowSurrogate

__all__ = [
    "compute_hemodynamics",
    "HemodynamicsResult",
    "HelmholtzCoil",
    "CoilSystem",
    "magnetic_force",
    "magnetic_torque",
    "FlowSurrogate",
]
