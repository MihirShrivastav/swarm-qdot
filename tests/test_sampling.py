import torch

from numerics.sampling import jittered_grid, monte_carlo_batch, uniform_grid


def test_uniform_grid_shape_and_weights():
    b = uniform_grid(2.0, 1.0, nq=10, device=torch.device("cpu"), dtype=torch.float64)
    assert b.coords.shape == (100, 2)
    assert b.weights.shape == (100,)
    assert b.x is not None and b.y is not None


def test_jittered_grid_bounds():
    X, Y = 2.0, 1.5
    b = jittered_grid(X, Y, nq=16, device=torch.device("cpu"), dtype=torch.float64, jitter_frac=0.3)
    assert b.coords.shape == (256, 2)
    assert torch.all(b.coords[:, 0] <= X) and torch.all(b.coords[:, 0] >= -X)
    assert torch.all(b.coords[:, 1] <= Y) and torch.all(b.coords[:, 1] >= -Y)
    assert b.x is None and b.y is None


def test_monte_carlo_batch_bounds_and_weights():
    X, Y = 3.0, 2.0
    n_points = 2000
    b = monte_carlo_batch(X, Y, n_points=n_points, device=torch.device("cpu"), dtype=torch.float32)
    assert b.coords.shape == (n_points, 2)
    assert torch.all(b.coords[:, 0] <= X) and torch.all(b.coords[:, 0] >= -X)
    assert torch.all(b.coords[:, 1] <= Y) and torch.all(b.coords[:, 1] >= -Y)
    expected_w = (4.0 * X * Y) / n_points
    assert torch.allclose(b.weights, torch.full_like(b.weights, expected_w))
