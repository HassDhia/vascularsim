"""VascularSim - Open-source vascular microbot simulation platform.

Provides tools for simulating autonomous microbot navigation through
vascular networks, including:

- Graph-based vascular anatomy representation
- Gymnasium RL environments (VascularNav-v0, FlowAwareNav-v0, MagneticNav-v0)
- Analytical hemodynamics (Poiseuille flow, Murray's law)
- Magnetic field models (Helmholtz coils, gradient force, torque)
- Neural flow surrogate for real-time prediction
- PPO training pipeline and benchmark suite
"""

__version__ = "0.1.1"

from vascularsim.graph import VascularGraph

# Register Gymnasium environments on package import
import vascularsim.envs  # noqa: F401

__all__ = [
    "__version__",
    "VascularGraph",
]
