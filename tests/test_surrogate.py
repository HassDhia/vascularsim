"""Tests for the neural flow surrogate (FlowSurrogate).

Validates MLP forward/backward, training convergence, from_graph factory,
and <10% relative error on training data.
"""

from __future__ import annotations

import numpy as np
import pytest

from vascularsim.benchmarks.environments import make_benchmark_graph
from vascularsim.physics.surrogate import FlowSurrogate


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_toy_data(
    n: int = 100, n_features: int = 7, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple data: y = sum(X, axis=1) + noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = X.sum(axis=1) + 0.1 * rng.standard_normal(n)
    return X, y


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestFlowSurrogateMLP:
    """Unit tests for the raw MLP mechanics."""

    def test_mlp_forward_shape(self) -> None:
        """Output shape matches (N,) after predict."""
        X, y = _make_toy_data(50)
        model = FlowSurrogate(hidden_sizes=[16, 16], learning_rate=0.01)
        model.train(X, y, epochs=10, seed=0)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_mlp_train_loss_decreases(self) -> None:
        """Training loss should generally decrease over epochs."""
        X, y = _make_toy_data(100)
        model = FlowSurrogate(hidden_sizes=[32, 32], learning_rate=0.01)
        losses = model.train(X, y, epochs=200, seed=0)
        # First loss should be larger than last
        assert losses[0] > losses[-1], (
            f"Loss did not decrease: first={losses[0]:.6f}, last={losses[-1]:.6f}"
        )

    def test_predict_shape(self) -> None:
        """predict() returns ndarray with correct length."""
        X, y = _make_toy_data(30, n_features=5)
        model = FlowSurrogate(hidden_sizes=[8, 8])
        model.train(X, y, epochs=5, seed=1)
        out = model.predict(X[:10])
        assert isinstance(out, np.ndarray)
        assert out.shape == (10,)

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces identical training results."""
        X, y = _make_toy_data(60)

        m1 = FlowSurrogate(hidden_sizes=[16, 16])
        losses1 = m1.train(X, y, epochs=50, seed=42)

        m2 = FlowSurrogate(hidden_sizes=[16, 16])
        losses2 = m2.train(X, y, epochs=50, seed=42)

        np.testing.assert_allclose(losses1, losses2, rtol=1e-10)

    def test_different_hidden_sizes(self) -> None:
        """Both [16,16] and [64,64] train without error."""
        X, y = _make_toy_data(40)

        for sizes in ([16, 16], [64, 64]):
            model = FlowSurrogate(hidden_sizes=sizes)
            losses = model.train(X, y, epochs=10, seed=0)
            assert len(losses) == 10
            preds = model.predict(X)
            assert preds.shape == (40,)

    def test_predict_before_train_raises(self) -> None:
        """Calling predict before train raises RuntimeError."""
        model = FlowSurrogate()
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(np.zeros((5, 7)))


class TestFlowSurrogateFromGraph:
    """Integration tests using from_graph on benchmark graphs."""

    def test_from_graph_returns_surrogate(self) -> None:
        """from_graph returns a FlowSurrogate and metrics dict."""
        graph = make_benchmark_graph(tier=1)
        surrogate, metrics = FlowSurrogate.from_graph(graph, epochs=500, seed=42)
        assert isinstance(surrogate, FlowSurrogate)
        assert "train_loss" in metrics
        assert "mean_relative_error" in metrics

    def test_surrogate_error_under_ten_percent(self) -> None:
        """Mean relative error on training data should be < 10%."""
        graph = make_benchmark_graph(tier=1)
        _, metrics = FlowSurrogate.from_graph(
            graph, hidden_sizes=[64, 64], epochs=2000, seed=42, learning_rate=0.005
        )
        assert metrics["mean_relative_error"] < 0.10, (
            f"MRE too high: {metrics['mean_relative_error']:.4f}"
        )

    @pytest.mark.parametrize("tier", [1, 2, 3])
    def test_works_on_benchmark_graphs(self, tier: int) -> None:
        """from_graph runs without error on tier 1-3 benchmarks."""
        graph = make_benchmark_graph(tier=tier)
        surrogate, metrics = FlowSurrogate.from_graph(graph, epochs=200, seed=42)
        assert isinstance(surrogate, FlowSurrogate)
        assert metrics["train_loss"] < float("inf")
        assert 0.0 <= metrics["mean_relative_error"] < 1.0
