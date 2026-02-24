"""Tests for the magnetic field physics module.

Validates Helmholtz coil field computation, gradient calculation,
microbot force/torque, and multi-coil superposition.
"""

from __future__ import annotations

import numpy as np
import pytest

from vascularsim.physics.magnetic import (
    MU_0,
    HelmholtzCoil,
    CoilSystem,
    magnetic_force,
    magnetic_torque,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_coil() -> HelmholtzCoil:
    """Standard Helmholtz pair along z-axis, radius=separation=0.1 m."""
    return HelmholtzCoil()


@pytest.fixture
def three_axis_system() -> CoilSystem:
    """Three orthogonal Helmholtz pairs with default parameters."""
    return CoilSystem.three_axis()


# ---------------------------------------------------------------------------
# Helmholtz coil field tests
# ---------------------------------------------------------------------------


class TestHelmholtzField:
    def test_helmholtz_center_field(self, default_coil: HelmholtzCoil) -> None:
        """At (0,0,0), the field should match the Helmholtz analytic formula.

        For Helmholtz condition (separation = R):
            B_center = mu_0 * n * I * 8 / (5 * sqrt(5) * R)
        """
        B = default_coil.field_at(np.array([0.0, 0.0, 0.0]))
        R = default_coil.coil_radius
        n = default_coil.n_turns
        I = default_coil.current

        B_analytic = MU_0 * n * I * 8.0 / (5.0 * np.sqrt(5.0) * R)

        # Field should be along z-axis only
        assert B[0] == pytest.approx(0.0, abs=1e-10)
        assert B[1] == pytest.approx(0.0, abs=1e-10)
        assert B[2] == pytest.approx(B_analytic, rel=1e-6)

    def test_field_symmetry(self, default_coil: HelmholtzCoil) -> None:
        """B(0,0,z) magnitude should equal B(0,0,-z) magnitude."""
        z_offset = 0.02  # 2 cm off-center
        B_pos = default_coil.field_at(np.array([0.0, 0.0, z_offset]))
        B_neg = default_coil.field_at(np.array([0.0, 0.0, -z_offset]))

        # Magnitudes should match (symmetry about midplane)
        assert np.linalg.norm(B_pos) == pytest.approx(
            np.linalg.norm(B_neg), rel=1e-10
        )

    def test_field_along_axis(self, default_coil: HelmholtzCoil) -> None:
        """Field at center should be >= field at off-center on-axis points."""
        B_center = np.linalg.norm(
            default_coil.field_at(np.array([0.0, 0.0, 0.0]))
        )
        B_offset = np.linalg.norm(
            default_coil.field_at(np.array([0.0, 0.0, 0.04]))
        )
        assert B_center >= B_offset

    def test_field_direction_on_axis(self, default_coil: HelmholtzCoil) -> None:
        """On-axis field should point purely along the coil axis (z)."""
        B = default_coil.field_at(np.array([0.0, 0.0, 0.01]))
        # Transverse components should be zero on-axis
        assert abs(B[0]) < 1e-12
        assert abs(B[1]) < 1e-12
        assert B[2] > 0  # Positive field for positive current

    def test_field_x_axis_coil(self) -> None:
        """Coil aligned along x-axis should produce field in x-direction."""
        coil = HelmholtzCoil(axis=0)
        B = coil.field_at(np.array([0.0, 0.0, 0.0]))
        assert abs(B[0]) > 1e-6
        assert abs(B[1]) < 1e-10
        assert abs(B[2]) < 1e-10

    def test_invalid_axis_raises(self) -> None:
        """Axis must be 0, 1, or 2."""
        with pytest.raises(ValueError, match="axis must be"):
            HelmholtzCoil(axis=3)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


class TestGradient:
    def test_gradient_shape(self, default_coil: HelmholtzCoil) -> None:
        """Gradient should be a 3x3 matrix."""
        grad = default_coil.gradient_at(np.array([0.0, 0.0, 0.0]))
        assert grad.shape == (3, 3)

    def test_gradient_finite_difference_accuracy(self) -> None:
        """Compare FD gradient to analytically expected dBz/dz at center.

        At the exact center of a Helmholtz pair, dBz/dz should be ~0
        (this is why the Helmholtz condition is used -- uniformity).
        """
        coil = HelmholtzCoil()
        grad = coil.gradient_at(np.array([0.0, 0.0, 0.0]))

        # dBz/dz at center should be near zero for Helmholtz condition
        assert abs(grad[2, 2]) < 1e-3  # Very small gradient at center

    def test_gradient_nonzero_offcenter(self, default_coil: HelmholtzCoil) -> None:
        """Gradient should be nonzero away from center."""
        grad = default_coil.gradient_at(np.array([0.0, 0.0, 0.04]))
        # At least one component should be nonzero
        assert np.linalg.norm(grad) > 1e-6

    def test_gradient_divergence_free(self, default_coil: HelmholtzCoil) -> None:
        """div(B) = dBx/dx + dBy/dy + dBz/dz should be ~0 (Maxwell)."""
        pos = np.array([0.01, 0.01, 0.02])
        grad = default_coil.gradient_at(pos)
        div_B = grad[0, 0] + grad[1, 1] + grad[2, 2]
        assert abs(div_B) < 1e-2  # Should be near zero


# ---------------------------------------------------------------------------
# Force tests
# ---------------------------------------------------------------------------


class TestMagneticForce:
    def test_magnetic_force_direction(self) -> None:
        """Force should point toward stronger field (gradient direction).

        A z-magnetised bot in a z-gradient should experience z-force.
        """
        # Construct a gradient where dBz/dz > 0 (field increasing in +z)
        grad = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],  # dBz/dz = 0.1 T/m
        ])
        moment = np.array([0.0, 0.0, 1e-12])
        F = magnetic_force(grad, moment)

        # Force should be in +z direction
        assert F[2] > 0
        assert abs(F[0]) < 1e-20
        assert abs(F[1]) < 1e-20

    def test_force_units_reasonable(self) -> None:
        """Force should be in pN--nN range for typical microbot parameters.

        Typical: gradient ~1 T/m, moment ~1e-12 A*m^2
        Expected: F ~ 1e-12 N = 1 pN
        """
        coil = HelmholtzCoil(current=10.0)
        # Evaluate gradient off-center where it's nonzero
        pos = np.array([0.0, 0.0, 0.04])
        grad = coil.gradient_at(pos)
        moment = np.array([0.0, 0.0, 1e-12])
        F = magnetic_force(grad, moment)
        F_mag = np.linalg.norm(F)

        # Should be in the pico-to-nano newton range
        assert 1e-18 < F_mag < 1e-6, (
            f"Force magnitude {F_mag:.2e} N outside expected pN-nN range"
        )

    def test_force_default_moment(self) -> None:
        """magnetic_force should work with default moment."""
        grad = np.eye(3) * 0.01
        F = magnetic_force(grad)
        assert F.shape == (3,)
        # Default moment is [0, 0, 1e-12], so F = grad.T @ m
        assert F[2] == pytest.approx(1e-14, rel=1e-6)

    def test_force_zero_gradient(self) -> None:
        """Zero gradient should produce zero force."""
        grad = np.zeros((3, 3))
        moment = np.array([0.0, 0.0, 1e-12])
        F = magnetic_force(grad, moment)
        np.testing.assert_allclose(F, [0.0, 0.0, 0.0], atol=1e-30)


# ---------------------------------------------------------------------------
# Torque tests
# ---------------------------------------------------------------------------


class TestMagneticTorque:
    def test_magnetic_torque_perpendicular(self) -> None:
        """Torque is zero when moment is parallel to field."""
        B = np.array([0.0, 0.0, 0.01])  # Field along z
        m = np.array([0.0, 0.0, 1e-12])  # Moment along z
        tau = magnetic_torque(B, m)
        np.testing.assert_allclose(tau, [0.0, 0.0, 0.0], atol=1e-25)

    def test_magnetic_torque_cross_product(self) -> None:
        """Torque = m x B should follow right-hand rule.

        m along z, B along x => tau along -y (z cross x = -y... wait:
        z x x = -y is wrong. Let's compute:  [0,0,mz] x [Bx,0,0]
        = [0*0 - mz*0, mz*Bx - 0*0, 0*0 - 0*Bx] = [0, mz*Bx, 0]
        """
        Bx = 0.01
        mz = 1e-12
        B = np.array([Bx, 0.0, 0.0])
        m = np.array([0.0, 0.0, mz])
        tau = magnetic_torque(B, m)

        expected = np.cross(m, B)
        np.testing.assert_allclose(tau, expected, atol=1e-30)
        # Specifically: tau_y = mz * Bx
        assert tau[1] == pytest.approx(mz * Bx, rel=1e-10)

    def test_torque_default_moment(self) -> None:
        """magnetic_torque should work with default moment."""
        B = np.array([0.01, 0.0, 0.0])
        tau = magnetic_torque(B)
        assert tau.shape == (3,)
        # Default moment [0, 0, 1e-12] x [0.01, 0, 0] = [0, 1e-14, 0]
        assert tau[1] == pytest.approx(1e-14, rel=1e-6)

    def test_torque_magnitude(self) -> None:
        """Torque magnitude should be |m||B|sin(theta)."""
        B = np.array([0.01, 0.0, 0.0])
        m = np.array([0.0, 0.0, 1e-12])
        tau = magnetic_torque(B, m)
        expected_mag = np.linalg.norm(m) * np.linalg.norm(B)  # sin(90) = 1
        assert np.linalg.norm(tau) == pytest.approx(expected_mag, rel=1e-10)


# ---------------------------------------------------------------------------
# CoilSystem tests
# ---------------------------------------------------------------------------


class TestCoilSystem:
    def test_three_axis_system(self, three_axis_system: CoilSystem) -> None:
        """3-axis system produces controllable field in any direction.

        With equal current in all three pairs, field at center should
        have equal magnitude along each axis.
        """
        B = three_axis_system.field_at(np.array([0.0, 0.0, 0.0]))
        # Each coil pair contributes the same magnitude along its axis
        assert abs(B[0]) > 1e-6
        assert abs(B[1]) > 1e-6
        assert abs(B[2]) > 1e-6
        # All components should be equal (symmetric construction)
        assert B[0] == pytest.approx(B[1], rel=1e-6)
        assert B[1] == pytest.approx(B[2], rel=1e-6)

    def test_coil_system_superposition(self) -> None:
        """Multiple coils superpose linearly.

        Sum of individual fields should equal system field.
        """
        coils = [
            HelmholtzCoil(axis=0, current=1.0),
            HelmholtzCoil(axis=2, current=2.0),
        ]
        system = CoilSystem(coils)
        pos = np.array([0.01, 0.02, 0.03])

        B_system = system.field_at(pos)
        B_individual = coils[0].field_at(pos) + coils[1].field_at(pos)

        np.testing.assert_allclose(B_system, B_individual, atol=1e-15)

    def test_coil_system_gradient_superposition(self) -> None:
        """System gradient should be the sum of individual gradients."""
        coils = [
            HelmholtzCoil(axis=0, current=1.5),
            HelmholtzCoil(axis=1, current=0.5),
        ]
        system = CoilSystem(coils)
        pos = np.array([0.01, 0.02, 0.01])

        grad_system = system.gradient_at(pos)
        grad_sum = coils[0].gradient_at(pos) + coils[1].gradient_at(pos)

        np.testing.assert_allclose(grad_system, grad_sum, atol=1e-15)

    def test_single_coil_system(self) -> None:
        """A system with one coil should match that coil directly."""
        coil = HelmholtzCoil(current=3.0, axis=1)
        system = CoilSystem([coil])
        pos = np.array([0.0, 0.0, 0.0])

        np.testing.assert_allclose(
            system.field_at(pos), coil.field_at(pos), atol=1e-15
        )

    def test_empty_system_zero_field(self) -> None:
        """A system with no coils should produce zero field."""
        system = CoilSystem([])
        B = system.field_at(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(B, [0.0, 0.0, 0.0], atol=1e-30)


# ---------------------------------------------------------------------------
# Integration: end-to-end microbot actuation
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_actuation_pipeline(self) -> None:
        """Complete workflow: coil -> field -> gradient -> force + torque."""
        system = CoilSystem.three_axis(current=5.0)
        pos = np.array([0.005, 0.005, 0.005])
        moment = np.array([0.0, 0.0, 1e-12])

        B = system.field_at(pos)
        grad = system.gradient_at(pos)
        F = magnetic_force(grad, moment)
        tau = magnetic_torque(B, moment)

        # All outputs should have correct shapes
        assert B.shape == (3,)
        assert grad.shape == (3, 3)
        assert F.shape == (3,)
        assert tau.shape == (3,)

        # Field should be nonzero
        assert np.linalg.norm(B) > 0
        # Force should be nonzero (off-center -> nonzero gradient)
        assert np.linalg.norm(F) > 0
