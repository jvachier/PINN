"""Tests for the finite-difference Fokker-Planck residual scheme.

The ``_FokkerPlanckPropagator._fp_loss`` uses second-order central
finite differences with symmetric (zero-flux) boundary padding.  When
fed the exact Gaussian solution at two consecutive time steps the
residual should be small, confirming the discrete scheme is consistent
with the continuous PDE.

These tests are pure numpy — no TensorFlow graph compilation is used —
so they run fast even without GPU hardware.
"""

import numpy as np


def _gaussian_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Normalised 1-D Gaussian evaluated on x."""
    out = np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)
    out[out < 1e-12] = 0.0
    return out.astype(np.float64)


def _fp_residual(
    p_t: np.ndarray,
    p_tp1: np.ndarray,
    dt: float,
    dx: float,
    drift: float,
    diff: float,
) -> np.ndarray:
    """Compute the discrete FP residual using the same scheme as _fp_loss."""
    p_pad = np.pad(p_tp1, 1, mode="symmetric")
    p_r = p_pad[2:]
    p_l = p_pad[:-2]
    dpdx = (p_r - p_l) / (2.0 * dx)
    d2pdx2 = (p_r - 2.0 * p_tp1 + p_l) / dx**2
    dpdt = (p_tp1 - p_t) / dt
    return dpdt + drift * dpdx - diff * d2pdx2


class TestFPResidual:
    """Numerical scheme tests — no TF/Keras dependency."""

    def test_residual_small_for_exact_solution(self):
        """FP residual must be small when p_t and p_tp1 are exact Gaussians."""
        dx = 0.05
        x = np.arange(-15.0, 15.0, dx)
        drift, diff, dt = 0.1, 0.5, 1.0

        mean0, var0 = 0.0, 1.0
        mean1 = mean0 + drift * dt
        var1 = var0 + 2.0 * diff * dt

        p_t = _gaussian_pdf(x, mean0, var0)
        p_tp1 = _gaussian_pdf(x, mean1, var1)

        residual = _fp_residual(p_t, p_tp1, dt, dx, drift, diff)
        # Relative residual normalised by the peak amplitude
        rel = np.mean(np.abs(residual)) / np.max(p_tp1)
        assert rel < 0.02, f"relative FP residual too large: {rel:.4f}"

    def test_residual_large_for_wrong_diff(self):
        """Using the wrong diffusion coefficient should yield a large residual."""
        dx = 0.05
        x = np.arange(-15.0, 15.0, dx)
        drift, diff, dt = 0.0, 0.5, 1.0

        p_t = _gaussian_pdf(x, 0.0, 1.0)
        p_tp1 = _gaussian_pdf(x, 0.0, 1.0 + 2.0 * diff * dt)

        wrong_diff = diff * 10.0  # deliberately wrong
        residual = _fp_residual(p_t, p_tp1, dt, dx, drift, wrong_diff)
        rel = np.mean(np.abs(residual)) / np.max(p_tp1)
        assert rel > 0.05, f"expected large residual for wrong diff, got {rel:.4f}"

    def test_residual_large_for_wrong_drift(self):
        """Using the wrong drift should yield a large residual."""
        dx = 0.05
        x = np.arange(-15.0, 15.0, dx)
        drift, diff, dt = 0.5, 0.5, 1.0

        p_t = _gaussian_pdf(x, 0.0, 1.0)
        p_tp1 = _gaussian_pdf(x, drift * dt, 1.0 + 2.0 * diff * dt)

        wrong_drift = -drift  # opposite sign
        residual = _fp_residual(p_t, p_tp1, dt, dx, wrong_drift, diff)
        rel = np.mean(np.abs(residual)) / np.max(p_tp1)
        assert rel > 0.05, f"expected large residual for wrong drift, got {rel:.4f}"

    def test_residual_zero_drift_pure_diffusion(self):
        """Pure diffusion (drift=0) residual must be small with the exact solution."""
        dx = 0.02
        x = np.arange(-20.0, 20.0, dx)
        diff, dt = 1.0, 0.1  # small dt keeps the FD truncation error low

        p_t = _gaussian_pdf(x, 0.0, 1.0)
        p_tp1 = _gaussian_pdf(x, 0.0, 1.0 + 2.0 * diff * dt)

        residual = _fp_residual(p_t, p_tp1, dt, dx, drift=0.0, diff=diff)
        rel = np.mean(np.abs(residual)) / np.max(p_tp1)
        assert rel < 0.02, f"pure-diffusion residual too large: {rel:.4f}"
