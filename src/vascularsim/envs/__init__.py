"""Gymnasium environments for vascular microbot navigation."""

from gymnasium.envs.registration import register

register(
    id="VascularNav-v0",
    entry_point="vascularsim.envs.vascular_nav:VascularNavEnv",
)

register(
    id="FlowAwareNav-v0",
    entry_point="vascularsim.envs.flow_nav:FlowAwareNavEnv",
)

register(
    id="MagneticNav-v0",
    entry_point="vascularsim.envs.magnetic_nav:MagneticNavEnv",
)
