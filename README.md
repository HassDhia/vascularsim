# VascularSim

**An Open-Source Platform for Reinforcement Learning-Based Microbot Navigation in Vascular Networks**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/tests-139%20passing-brightgreen.svg)
[![PyPI version](https://img.shields.io/pypi/v/vascularsim.svg?v=1)](https://pypi.org/project/vascularsim/)

---

VascularSim provides a complete stack for training RL agents to navigate blood vessel graphs: TubeTK data ingestion, Gymnasium environments with physics-based observations, analytical hemodynamics and magnetic field models, a neural flow surrogate, and a benchmark suite across 5 difficulty tiers.

## Installation

```bash
pip install vascularsim
```

With RL training support (requires PyTorch):

```bash
pip install "vascularsim[train]"
```

Development (includes pytest):

```bash
pip install "vascularsim[dev]"
```

## Quick Start

### 1. Load a vascular graph and explore it

```python
from vascularsim import VascularGraph
from vascularsim.benchmarks import make_benchmark_graph

# Create a benchmark vascular graph (tier 1 = straight vessel, 20 nodes)
graph = make_benchmark_graph(tier=1, seed=42)

print(f"Nodes: {graph.num_nodes}")
print(f"Edges: {graph.num_edges}")

# Access node properties
for node in graph.nodes[:3]:
    pos = graph.get_node_pos(node)
    radius = graph.get_node_radius(node)
    print(f"  {node}: pos={pos}, radius={radius:.4f}")
```

### 2. Run the RL environment

```python
import gymnasium as gym
from vascularsim import VascularGraph
from vascularsim.benchmarks import make_benchmark_graph

# Create a graph and pass it to the environment
graph = make_benchmark_graph(tier=1, seed=42)
env = gym.make("VascularNav-v0", graph=graph)
obs, info = env.reset()
print(f"Observation keys: {list(obs.keys())}")

# Step through the environment
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print(f"Reached target!")
        break

env.close()

# Physics-aware environments (use default graph if none provided)
flow_env = gym.make("FlowAwareNav-v0")
mag_env = gym.make("MagneticNav-v0")
```

### 3. Train a PPO agent and evaluate

```bash
# Train PPO for 200k steps (CPU, ~68 seconds)
python -m vascularsim.training.train --timesteps 200000 --output-dir ./checkpoints

# Evaluate trained agent vs random baseline
python -m vascularsim.training.evaluate --model-path ./checkpoints/ppo_vascularnav.zip

# Run benchmark suite across all 5 difficulty tiers
python -m vascularsim.benchmarks.runner --agent random --tiers 1 2 3 4 5 --output results.json
```

## Physics Modules

### Hemodynamics

Analytical Hagen-Poiseuille flow with Murray's law bifurcation distribution:

```python
from vascularsim.physics import compute_hemodynamics

result = compute_hemodynamics(graph)
# result.edge_velocities   — flow velocity per edge (mm/s)
# result.edge_pressures    — pressure drop per edge (Pa)
# result.edge_wall_shear   — wall shear stress per edge (Pa)
# result.node_pressures    — pressure at each node (Pa)
```

### Magnetic Fields

Helmholtz coil pairs with gradient-based force and torque on magnetic microbots:

```python
from vascularsim.physics import CoilSystem, magnetic_force, magnetic_torque
import numpy as np

coils = CoilSystem.three_axis(coil_radius=0.05, current=1.0)
point = np.array([0.0, 0.0, 0.0])

field = coils.field_at(point)          # Tesla
gradient = coils.gradient_at(point)    # T/m (3x3 Jacobian)

moment = np.array([0.0, 0.0, 1e-12])  # A*m^2
force = magnetic_force(gradient, moment)
torque = magnetic_torque(field, moment)
```

### Neural Flow Surrogate

Pure NumPy MLP trained in log-space for fast flow prediction (<10% MRE):

```python
from vascularsim.physics import FlowSurrogate

surrogate = FlowSurrogate.from_graph(graph)
predicted_velocities = surrogate.predict(features)
```

## Benchmark Tiers

| Tier | Name | Nodes | Topology |
|------|------|-------|----------|
| 1 | Straight | 20 | Linear vessel |
| 2 | Bifurcation | 35 | Single branch point |
| 3 | Tree | 50 | Two-level branching |
| 4 | Ring | 43 | Vessel loops |
| 5 | Dense Mesh | 147 | Sub-branches and dead-ends |

## Architecture

```
vascularsim/
  data/          TubeTK .tre parser + Girder API downloader
  graph.py       VascularGraph (NetworkX DiGraph wrapper)
  envs/          Gymnasium environments
    vascular_nav.py    VascularNav-v0 (base discrete navigation)
    flow_nav.py        FlowAwareNav-v0 (+ hemodynamic observations)
    magnetic_nav.py    MagneticNav-v0 (+ magnetic field observations)
    wrappers.py        FlatNavObservation (Dict→Box for SB3)
  physics/       Analytical physics models
    hemodynamics.py    Poiseuille flow + Murray's law + WSS
    magnetic.py        Helmholtz coils + force/torque
    surrogate.py       Neural MLP flow surrogate
  training/      RL training pipeline
    train.py           PPO training CLI
    evaluate.py        Evaluation + comparison CLI
  benchmarks/    Benchmark suite
    environments.py    5 difficulty tier generators
    runner.py          Benchmark runner CLI
```

## Test Suite

```bash
pytest tests/ -v    # 139 tests, ~2 seconds
```

## Citation

If you use VascularSim in your research, please cite:

```bibtex
@article{dhia2026vascularsim,
  title={VascularSim: An Open-Source Platform for Reinforcement Learning-Based Microbot Navigation in Vascular Networks},
  author={Dhia, Hass},
  year={2026},
  note={Smart Technology Investments Research Institute}
}
```

## Contact

Smart Technology Investments Research Institute
partners@smarttechinvest.com

## License

MIT
