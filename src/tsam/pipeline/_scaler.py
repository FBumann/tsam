"""Lightweight min-max scaler without a scikit-learn dependency.

Replaces ``sklearn.preprocessing.MinMaxScaler`` on the hot path so a default
``aggregate`` need not load the full scikit-learn stack (~600 ms import). The
arithmetic mirrors ``MinMaxScaler`` with the default ``feature_range`` of
``(0, 1)`` exactly — same ``scale``/``min`` factors applied in the same
multiply-then-add order — so the scaled values are bit-identical, including
scikit-learn's convention of treating a zero-range (constant) column as scale
``1`` so it maps to ``0`` without dividing by zero.
"""

from __future__ import annotations

import numpy as np


class MinMaxScaler:
    """Scale each column to ``[0, 1]``; a numpy stand-in for sklearn's scaler.

    Attributes:
        scale: Per-column multiplicative factor (``1 / range``, with a
            zero range treated as one).
        offset: Per-column additive term so the column minimum maps to zero.
    """

    scale: np.ndarray
    offset: np.ndarray

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Learn the per-column scale/offset and return the scaled data."""
        data = np.asarray(data, dtype=float)
        data_min = data.min(axis=0)
        data_range = data.max(axis=0) - data_min
        # Match sklearn: a zero range would divide by zero, so use 1 instead.
        data_range[data_range == 0.0] = 1.0
        self.scale = 1.0 / data_range
        self.offset = -data_min * self.scale
        return data * self.scale + self.offset

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Map scaled values back to the original units."""
        return (np.asarray(data, dtype=float) - self.offset) / self.scale
