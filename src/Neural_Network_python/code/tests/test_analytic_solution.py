"""Tests for the analytic Fokker-Planck solution in data_analytic.Analytic.

These tests verify that _funct(t) produces a correctly normalised Gaussian
with the mean and variance predicted by the 1-D drift-diffusion equation:
    mean(t)     =  vs * delta * t
    variance(t) =  2 * Dt * delta * t
"""

import numpy as np
import pandas as pd
import pytest
from modules.data_analytic import Analytic


@pytest.fixture(scope="module")
def ana():
    """Analytic instance with a minimal dummy DataFrame (no file I/O)."""
    df = pd.DataFrame({"time": [100, 500, 1000]})
    return Analytic(data=df)


class TestAnalyticSolution:
    def test_normalisation(self, ana):
        """The PDF must integrate to 1 (up to boundary truncation)."""
        result = ana._funct(500)
        dx = float(ana.x[1] - ana.x[0])
        integral = float(np.sum(result) * dx)
        assert abs(integral - 1.0) < 0.01, f"norm={integral:.4f}"

    def test_mean(self, ana):
        """Mean of the PDF must equal vs * delta * t_int."""
        t_int = 500
        result = ana._funct(t_int)
        dx = float(ana.x[1] - ana.x[0])
        computed_mean = float(np.sum(ana.x * result) * dx)
        expected_mean = ana.drift * ana.scaling_time * t_int
        assert (
            abs(computed_mean - expected_mean) < 0.02
        ), f"mean={computed_mean:.4f}, expected={expected_mean:.4f}"

    def test_variance(self, ana):
        """Variance of the PDF must equal 2 * Dt * delta * t_int."""
        t_int = 500
        result = ana._funct(t_int)
        dx = float(ana.x[1] - ana.x[0])
        mean = float(np.sum(ana.x * result) * dx)
        var = float(np.sum((ana.x - mean) ** 2 * result) * dx)
        expected_var = 2.0 * ana.diffusion * ana.scaling_time * t_int
        assert (
            abs(var - expected_var) < 0.05
        ), f"var={var:.4f}, expected={expected_var:.4f}"

    def test_non_negative(self, ana):
        """PDF values must be non-negative everywhere."""
        result = ana._funct(200)
        assert np.all(result >= 0.0)

    def test_peak_location(self, ana):
        """The peak must be close to the expected drift position."""
        t_int = 1000
        result = ana._funct(t_int)
        peak_x = float(ana.x[np.argmax(result)])
        expected_peak = ana.drift * ana.scaling_time * t_int
        assert (
            abs(peak_x - expected_peak) < 0.05
        ), f"peak_x={peak_x:.4f}, expected={expected_peak:.4f}"

    def test_wider_at_later_time(self, ana):
        """Distribution must broaden over time (larger variance at t2 > t1)."""
        dx = float(ana.x[1] - ana.x[0])

        def variance(t_int):
            r = ana._funct(t_int)
            m = float(np.sum(ana.x * r) * dx)
            return float(np.sum((ana.x - m) ** 2 * r) * dx)

        assert variance(1000) > variance(500)
