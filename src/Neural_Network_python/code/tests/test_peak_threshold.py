"""Tests for NN._threshold_main_peak.

This method zeroes tail bins and removes isolated satellite spikes so
that only the connected region around the global maximum survives.
It is a pure numpy function — no I/O or model weights are required.
"""

import numpy as np
from modules.neural_network import NN


def _gaussian(n: int, center: int, sigma: float) -> np.ndarray:
    x = np.arange(n, dtype=float)
    g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return g / g.max()


class TestThresholdMainPeak:
    def test_isolated_satellite_is_zeroed(self):
        """A spike disconnected from the main peak must be zeroed out."""
        arr = np.zeros(100)
        arr[50] = 1.0
        arr[49] = 0.8
        arr[51] = 0.8
        arr[20] = 0.5  # isolated — not connected to peak at 50

        result = NN._threshold_main_peak(arr, frac=0.10)

        assert result[20] == 0.0
        assert result[50] > 0.0

    def test_connected_segment_is_preserved(self):
        """All bins that form a contiguous above-threshold region survive."""
        arr = _gaussian(80, center=40, sigma=5.0)
        result = NN._threshold_main_peak(arr, frac=0.10)

        # Bins near the peak should be non-zero
        assert np.all(result[36:45] > 0.0)

    def test_far_tail_bins_are_zeroed(self):
        """Bins far from the peak that fall below the threshold are zeroed."""
        arr = _gaussian(200, center=100, sigma=3.0)
        result = NN._threshold_main_peak(arr, frac=0.10)

        # Bins at the very edges of the domain are in the deep tail
        assert result[0] == 0.0
        assert result[-1] == 0.0

    def test_two_satellites_both_removed(self):
        """Multiple isolated spikes on both sides are all zeroed."""
        arr = np.zeros(120)
        arr[60] = 1.0  # main peak
        arr[59] = 0.9
        arr[61] = 0.9
        arr[10] = 0.5  # left satellite
        arr[110] = 0.4  # right satellite

        result = NN._threshold_main_peak(arr, frac=0.10)

        assert result[10] == 0.0
        assert result[110] == 0.0
        assert result[60] > 0.0

    def test_zero_array_returns_zeros(self):
        """An all-zero input should return an all-zero output."""
        arr = np.zeros(50)
        result = NN._threshold_main_peak(arr, frac=0.10)
        assert np.all(result == 0.0)

    def test_output_shape_unchanged(self):
        """Output shape must match input shape."""
        arr = _gaussian(300, center=150, sigma=10.0)
        result = NN._threshold_main_peak(arr, frac=0.05)
        assert result.shape == arr.shape

    def test_frac_zero_keeps_everything_connected(self):
        """With frac=0 the entire array counts as one connected component."""
        arr = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.3, 0.0])
        result = NN._threshold_main_peak(arr, frac=0.0)
        # frac=0 → threshold=0 → every non-negative value satisfies mask.
        # The connected walk from peak (index 2) extends left until it hits
        # a False mask entry.  Since all values >= 0, the whole array is
        # connected.  result should equal arr element-wise.
        np.testing.assert_array_equal(result, arr)
