"""Magnetic field models for vascular microbot actuation.

Implements Helmholtz coil pairs, multi-coil systems, and force/torque
calculations on magnetic microbots.  Pure NumPy -- no external solvers.

Physics reference:
    - Helmholtz coil on-axis field: Biot-Savart law for circular loops
    - Off-axis: first-order gradient expansion from Maxwell's equations
    - Microbot force: F_i = sum_j(m_j * dB_j / dx_i)
    - Microbot torque: tau = m x B
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Permeability of free space (T*m/A)
MU_0: float = 4.0 * np.pi * 1e-7

# Default finite-difference step for gradient computation (metres)
_FD_STEP: float = 1e-6


class HelmholtzCoil:
    """A pair of co-axial circular coils in the Helmholtz configuration.

    The two coils are centred at +/- separation/2 along the chosen axis,
    each with the given radius, number of turns, and current.  When
    separation == radius the pair satisfies the Helmholtz condition for
    maximal field uniformity at the midpoint.

    Args:
        coil_radius: Radius of each coil in metres.
        separation: Distance between the two coils in metres.
        n_turns: Number of wire turns per coil.
        current: Current through the coils in amperes.
        axis: Alignment axis (0=x, 1=y, 2=z).
    """

    def __init__(
        self,
        coil_radius: float = 0.1,
        separation: float = 0.1,
        n_turns: int = 100,
        current: float = 1.0,
        axis: int = 2,
    ) -> None:
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
        self.coil_radius = coil_radius
        self.separation = separation
        self.n_turns = n_turns
        self.current = current
        self.axis = axis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_axis_field_magnitude(self, z: float) -> float:
        """Compute on-axis B magnitude from a single loop at origin.

        Uses Biot-Savart for a circular loop:
            B = mu_0 * n * I * R^2 / (2 * (R^2 + z^2)^(3/2))

        Args:
            z: Signed distance along the coil axis from the loop centre.

        Returns:
            Scalar magnetic field strength in tesla.
        """
        R = self.coil_radius
        return (
            MU_0
            * self.n_turns
            * self.current
            * R**2
            / (2.0 * (R**2 + z**2) ** 1.5)
        )

    def field_at(self, position: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute the B-field vector at an arbitrary 3-D position.

        For on-axis points the exact Biot-Savart formula is used.
        For off-axis points a first-order gradient expansion is applied:
        because div(B)=0, the transverse gradient is -0.5 * dBz/dz.

        Args:
            position: Shape (3,) array [x, y, z] in metres.

        Returns:
            Shape (3,) magnetic field vector in tesla.
        """
        pos = np.asarray(position, dtype=np.float64)
        half_sep = self.separation / 2.0

        # Extract on-axis coordinate and transverse coordinates
        z_coord = pos[self.axis]
        transverse_mask = np.ones(3, dtype=bool)
        transverse_mask[self.axis] = False
        rho_vec = pos[transverse_mask]  # 2-element vector
        rho = np.linalg.norm(rho_vec)

        # On-axis field from each coil (coils at +half_sep and -half_sep)
        z1 = z_coord - half_sep   # distance from coil 1
        z2 = z_coord + half_sep   # distance from coil 2
        Bz1 = self._on_axis_field_magnitude(z1)
        Bz2 = self._on_axis_field_magnitude(z2)
        Bz_total = Bz1 + Bz2

        # Build the output field vector
        B = np.zeros(3, dtype=np.float64)
        B[self.axis] = Bz_total

        # Off-axis correction using first-order Taylor expansion.
        # From Maxwell's equations (div B = 0 in cylindrical symmetry):
        #   B_rho ≈ -rho/2 * dBz/dz
        if rho > 1e-15:
            # Compute dBz/dz analytically
            R = self.coil_radius
            n = self.n_turns
            I = self.current

            dBz_dz1 = (
                -3.0 * MU_0 * n * I * R**2 * z1
                / (2.0 * (R**2 + z1**2) ** 2.5)
            )
            dBz_dz2 = (
                -3.0 * MU_0 * n * I * R**2 * z2
                / (2.0 * (R**2 + z2**2) ** 2.5)
            )
            dBz_dz = dBz_dz1 + dBz_dz2

            # Transverse correction: B_rho = -rho/2 * dBz/dz
            B_rho = -0.5 * rho * dBz_dz

            # Project radial field back into Cartesian components
            rho_unit = rho_vec / rho
            idx = 0
            for i in range(3):
                if i != self.axis:
                    B[i] = B_rho * rho_unit[idx]
                    idx += 1

        return B

    def gradient_at(
        self,
        position: NDArray[np.floating],
        step: float = _FD_STEP,
    ) -> NDArray[np.floating]:
        """Compute the field gradient tensor dBi/dxj via finite differences.

        Args:
            position: Shape (3,) position in metres.
            step: Finite-difference step size.

        Returns:
            Shape (3, 3) Jacobian matrix where element [i, j] = dB_i/dx_j.
        """
        pos = np.asarray(position, dtype=np.float64)
        grad = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            pos_fwd = pos.copy()
            pos_bwd = pos.copy()
            pos_fwd[j] += step
            pos_bwd[j] -= step
            grad[:, j] = (self.field_at(pos_fwd) - self.field_at(pos_bwd)) / (
                2.0 * step
            )
        return grad


class CoilSystem:
    """A collection of Helmholtz coil pairs for multi-axis field control.

    Superimposes the fields of all constituent coils to produce an
    arbitrary field vector and gradient at any point.

    Args:
        coils: List of HelmholtzCoil instances.
    """

    def __init__(self, coils: list[HelmholtzCoil]) -> None:
        self.coils = list(coils)

    @classmethod
    def three_axis(
        cls,
        radius: float = 0.1,
        n_turns: int = 100,
        current: float = 1.0,
    ) -> "CoilSystem":
        """Create a 3-axis Helmholtz system (one pair per axis).

        Each pair uses the Helmholtz condition (separation = radius).

        Args:
            radius: Coil radius in metres.
            n_turns: Turns per coil.
            current: Current in amperes.

        Returns:
            CoilSystem with three orthogonal coil pairs.
        """
        coils = [
            HelmholtzCoil(
                coil_radius=radius,
                separation=radius,
                n_turns=n_turns,
                current=current,
                axis=ax,
            )
            for ax in range(3)
        ]
        return cls(coils)

    def field_at(self, position: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute superimposed B-field at position.

        Args:
            position: Shape (3,) position in metres.

        Returns:
            Shape (3,) field vector in tesla.
        """
        pos = np.asarray(position, dtype=np.float64)
        return sum(
            (coil.field_at(pos) for coil in self.coils),
            np.zeros(3, dtype=np.float64),
        )

    def gradient_at(
        self,
        position: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute superimposed field gradient at position.

        Args:
            position: Shape (3,) position in metres.

        Returns:
            Shape (3, 3) gradient tensor.
        """
        pos = np.asarray(position, dtype=np.float64)
        return sum(
            (coil.gradient_at(pos) for coil in self.coils),
            np.zeros((3, 3), dtype=np.float64),
        )


# ------------------------------------------------------------------
# Force and torque on a magnetic microbot
# ------------------------------------------------------------------


def magnetic_force(
    field_gradient: NDArray[np.floating],
    magnetic_moment: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Compute the magnetic gradient force on a point dipole.

    F_i = sum_j(m_j * dB_j / dx_i) = gradient^T @ moment

    Args:
        field_gradient: Shape (3, 3) Jacobian dB_i/dx_j.
        magnetic_moment: Shape (3,) dipole moment in A*m^2.
            Defaults to [0, 0, 1e-12] (typical 10um iron-oxide bot).

    Returns:
        Shape (3,) force vector in newtons.
    """
    if magnetic_moment is None:
        magnetic_moment = np.array([0.0, 0.0, 1e-12])
    grad = np.asarray(field_gradient, dtype=np.float64)
    m = np.asarray(magnetic_moment, dtype=np.float64)
    return grad.T @ m


def magnetic_torque(
    field: NDArray[np.floating],
    magnetic_moment: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Compute the magnetic torque on a point dipole.

    tau = m x B

    Args:
        field: Shape (3,) B-field vector in tesla.
        magnetic_moment: Shape (3,) dipole moment in A*m^2.
            Defaults to [0, 0, 1e-12].

    Returns:
        Shape (3,) torque vector in N*m.
    """
    if magnetic_moment is None:
        magnetic_moment = np.array([0.0, 0.0, 1e-12])
    B = np.asarray(field, dtype=np.float64)
    m = np.asarray(magnetic_moment, dtype=np.float64)
    return np.cross(m, B)
