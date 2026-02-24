"""Neural flow surrogate for fast edge-velocity prediction.

A lightweight NumPy-only MLP that learns to map edge features
(radius, length, pressures, position) to Hagen-Poiseuille flow
velocity.  Eliminates the need for full hemodynamic re-computation
during RL rollouts.

Internally trains on log-transformed velocities to handle the wide
dynamic range (velocities can span 4+ orders of magnitude).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from vascularsim.graph import VascularGraph
from vascularsim.physics.hemodynamics import compute_hemodynamics


class FlowSurrogate:
    """Two-hidden-layer MLP trained to predict per-edge flow velocity.

    Uses ReLU activations for hidden layers and linear output.
    Trained with mini-batch SGD on MSE loss in log-space for
    numerically stable learning across wide velocity ranges.

    Args:
        hidden_sizes: Widths of the two hidden layers.
        learning_rate: Step size for SGD.
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.01,
    ) -> None:
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        self.hidden_sizes = list(hidden_sizes)
        self.learning_rate = learning_rate
        self._weights: list[tuple[NDArray, NDArray]] = []
        self._fitted = False
        # Normalisation stats (set during training)
        self._x_mean: NDArray | None = None
        self._x_std: NDArray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        # Whether to use log-space for targets (set in from_graph)
        self._log_targets: bool = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _forward(
        self, X: NDArray
    ) -> tuple[NDArray, list[NDArray], list[NDArray]]:
        """Full forward pass returning output, pre-activations, activations."""
        activations: list[NDArray] = [X]
        pre_acts: list[NDArray] = []

        h = X
        for i, (W, b) in enumerate(self._weights):
            z = h @ W + b
            pre_acts.append(z)
            if i < len(self._weights) - 1:
                h = np.maximum(0.0, z)  # ReLU
            else:
                h = z  # linear output
            activations.append(h)

        return h, pre_acts, activations

    def predict(self, X: NDArray) -> NDArray:
        """Forward pass returning predictions shape (N,).

        Args:
            X: Input features shape (N, n_features).

        Returns:
            Predicted flow velocities shape (N,).
        """
        if not self._fitted:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X_norm = self._normalise_x(X)
        out, _, _ = self._forward(X_norm)
        # De-normalise output
        raw = (out.ravel() * self._y_std) + self._y_mean
        if self._log_targets:
            return np.exp(raw)
        return raw

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _init_weights(self, n_features: int, seed: int | None = None) -> None:
        """He-initialise weights for each layer."""
        rng = np.random.default_rng(seed)
        layer_sizes = [n_features] + self.hidden_sizes + [1]
        self._weights = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)
            W = rng.normal(0, std, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self._weights.append((W, b))

    def _normalise_x(self, X: NDArray) -> NDArray:
        """Z-score normalise using stored training stats."""
        if self._x_mean is None:
            return X
        return (X - self._x_mean) / (self._x_std + 1e-8)

    def train(
        self,
        X: NDArray,
        y: NDArray,
        epochs: int = 500,
        batch_size: int = 32,
        seed: int | None = None,
    ) -> list[float]:
        """Train the MLP on (X, y) data.

        Args:
            X: Features shape (N, n_features).
            y: Targets shape (N,).
            epochs: Number of passes over the data.
            batch_size: Mini-batch size.
            seed: Random seed for reproducibility.

        Returns:
            List of MSE loss values per epoch.
        """
        rng = np.random.default_rng(seed)
        N, n_feat = X.shape
        y = y.ravel().copy()

        # Apply log transform if enabled (for wide-range velocity data)
        if self._log_targets:
            y = np.log(np.maximum(y, 1e-30))

        # Compute normalisation stats
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) if y.std() > 1e-12 else 1.0

        X_norm = self._normalise_x(X)
        y_norm = (y - self._y_mean) / self._y_std

        self._init_weights(n_feat, seed=seed)

        losses: list[float] = []

        for epoch in range(epochs):
            # Shuffle
            perm = rng.permutation(N)
            X_shuf = X_norm[perm]
            y_shuf = y_norm[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end].reshape(-1, 1)
                bs = Xb.shape[0]

                # Forward
                out, pre_acts, activations = self._forward(Xb)

                # MSE loss
                residual = out - yb  # (bs, 1)
                batch_loss = float(np.mean(residual ** 2))
                epoch_loss += batch_loss
                n_batches += 1

                # Backward (manual chain rule)
                # dL/dout = 2 * residual / bs
                d_out = (2.0 / bs) * residual  # (bs, 1)

                d = d_out
                for i in reversed(range(len(self._weights))):
                    W, b = self._weights[i]
                    a_prev = activations[i]

                    # Gradients for this layer
                    dW = a_prev.T @ d  # (fan_in, fan_out)
                    db = d.sum(axis=0, keepdims=True)  # (1, fan_out)

                    # Propagate gradient to previous layer
                    if i > 0:
                        d = d @ W.T
                        # ReLU derivative
                        d = d * (pre_acts[i - 1] > 0).astype(float)

                    # Update weights
                    self._weights[i] = (
                        W - self.learning_rate * dW,
                        b - self.learning_rate * db,
                    )

            losses.append(epoch_loss / max(n_batches, 1))

        self._fitted = True
        return losses

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_graph(
        cls,
        graph: VascularGraph,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        seed: int = 42,
    ) -> tuple["FlowSurrogate", dict[str, Any]]:
        """Train a surrogate directly from a VascularGraph.

        Runs ``compute_hemodynamics`` to get ground-truth velocities, then
        extracts per-edge features and trains an MLP.  Uses log-space
        training to handle the wide dynamic range of Poiseuille velocities.

        Features per edge:
            [mean_radius, length, upstream_pressure, downstream_pressure,
             position_x, position_y, position_z]

        Args:
            graph: Populated VascularGraph.
            hidden_sizes: MLP hidden layer widths.
            learning_rate: SGD learning rate.
            epochs: Training epochs.
            seed: Random seed.

        Returns:
            (trained_surrogate, metrics_dict) where metrics_dict
            contains ``train_loss`` and ``mean_relative_error``.
        """
        result = compute_hemodynamics(graph)
        g = graph._graph  # noqa: SLF001

        # Extract features for every directed edge
        features_list: list[list[float]] = []
        targets: list[float] = []

        for u, v in g.edges:
            edata = g.edges[u, v]
            mean_radius = edata["mean_radius"]
            length = edata["length"]

            p_u = result.node_pressures.get(u, 0.0)
            p_v = result.node_pressures.get(v, 0.0)

            pos_u = g.nodes[u]["pos"]
            pos_v = g.nodes[v]["pos"]
            mid_pos = (
                np.asarray(pos_u, dtype=float)
                + np.asarray(pos_v, dtype=float)
            ) / 2.0

            features_list.append([
                mean_radius,
                length,
                p_u,
                p_v,
                float(mid_pos[0]),
                float(mid_pos[1]),
                float(mid_pos[2]),
            ])

            velocity = result.edge_velocities.get((u, v), 0.0)
            targets.append(velocity)

        X = np.array(features_list, dtype=np.float64)
        y = np.array(targets, dtype=np.float64)

        # Clamp zero/near-zero velocities to a small positive floor
        # so log-transform is safe
        y = np.maximum(y, 1e-30)

        surrogate = cls(
            hidden_sizes=hidden_sizes or [64, 64],
            learning_rate=learning_rate,
        )
        surrogate._log_targets = True
        losses = surrogate.train(X, y, epochs=epochs, seed=seed)

        # Compute mean relative error in original space
        predictions = surrogate.predict(X)
        abs_errors = np.abs(predictions - y)
        # Use mean of targets as denominator scale to avoid
        # instability from tiny values
        denom = np.maximum(np.abs(y), np.mean(np.abs(y)) * 0.01)
        relative_errors = abs_errors / denom
        mean_rel_error = float(np.mean(relative_errors))

        metrics = {
            "train_loss": losses[-1] if losses else float("inf"),
            "mean_relative_error": mean_rel_error,
        }

        return surrogate, metrics
