"""Fuzz tests pinning the sklearn-free rewrites to scikit-learn as the oracle.

To keep a plain ``import tsam`` and the default ``aggregate`` off scikit-learn's
~600 ms import, three hot-path pieces were reimplemented on numpy/scipy:

* :class:`tsam.pipeline._scaler.MinMaxScaler` replaces
  ``sklearn.preprocessing.MinMaxScaler``,
* :func:`tsam.algorithms.representations._euclidean_distances` replaces
  ``sklearn.metrics.pairwise.euclidean_distances``,
* :func:`tsam.algorithms.clustering._ward_labels` replaces the default
  ``sklearn.cluster.AgglomerativeClustering(linkage="ward")``.

Each is asserted equivalent to its scikit-learn counterpart across many random
inputs so any future numerical drift is caught. scikit-learn is a hard project
dependency, so it is always importable here as the reference implementation.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances as sk_euclidean_distances
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from tsam.algorithms.clustering import _ward_labels
from tsam.algorithms.representations import _euclidean_distances
from tsam.pipeline._scaler import MinMaxScaler

SEEDS = list(range(40))


def _random_matrix(rng: np.random.Generator) -> np.ndarray:
    """A random 2-D matrix with a shape and scale drawn from broad ranges."""
    n_rows = int(rng.integers(2, 200))
    n_cols = int(rng.integers(1, 96))
    scale = 10.0 ** rng.integers(-3, 4)
    offset = rng.uniform(-1000, 1000)
    return rng.normal(size=(n_rows, n_cols)) * scale + offset


class TestMinMaxScaler:
    """numpy MinMaxScaler must match sklearn's fit_transform and inverse."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_sklearn_forward_and_inverse(self, seed: int):
        # Bit-identical: same scale/min factors applied in sklearn's own
        # multiply-then-add order, so no downstream distance tie can break
        # differently than it did on the scikit-learn backend.
        rng = np.random.default_rng(seed)
        data = _random_matrix(rng)

        transformed = MinMaxScaler().fit_transform(data)
        sk_scaler = SkMinMaxScaler()
        sk_transformed = sk_scaler.fit_transform(data)
        np.testing.assert_array_equal(transformed, sk_transformed)

        # Inverse must also match sklearn's inverse bit-for-bit and round-trip.
        scaler = MinMaxScaler()
        scaler.fit_transform(data)
        recovered = scaler.inverse_transform(transformed)
        np.testing.assert_array_equal(
            recovered, sk_scaler.inverse_transform(sk_transformed)
        )
        np.testing.assert_allclose(recovered, data, rtol=1e-9, atol=1e-9)

    def test_constant_columns_map_to_zero_like_sklearn(self):
        # Zero-range columns are sklearn's tricky case: scale is forced to 1 so
        # the column maps to 0 rather than dividing by zero.
        data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        got = MinMaxScaler().fit_transform(data)
        want = SkMinMaxScaler().fit_transform(data)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
        assert np.all(got[:, 0] == 0.0)

    def test_integer_input_does_not_truncate(self):
        data = np.arange(12).reshape(6, 2)
        got = MinMaxScaler().fit_transform(data)
        want = SkMinMaxScaler().fit_transform(data.astype(float))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


class TestEuclideanDistances:
    """The numpy euclidean distance must be bit-identical to sklearn's."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_asymmetric_matches_sklearn(self, seed: int):
        rng = np.random.default_rng(seed)
        n_cols = int(rng.integers(1, 96))
        x = rng.normal(size=(int(rng.integers(1, 150)), n_cols))
        y = rng.normal(size=(int(rng.integers(1, 150)), n_cols))

        got = _euclidean_distances(x, y)
        want = sk_euclidean_distances(x, y)
        # Bit-identical: same BLAS expansion in the same accumulation order.
        np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_symmetric_matches_sklearn(self, seed: int):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(int(rng.integers(2, 150)), int(rng.integers(1, 96))))

        got = _euclidean_distances(x, x, symmetric=True)
        want = sk_euclidean_distances(x)
        np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_medoid_selection_is_invariant(self, seed: int):
        # The distance feeds an argmin over row sums; pin that the selected
        # index matches what sklearn's distances would pick.
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(int(rng.integers(2, 120)), int(rng.integers(1, 64))))
        got = int(np.argmin(_euclidean_distances(x, x, symmetric=True).sum(axis=0)))
        want = int(np.argmin(sk_euclidean_distances(x).sum(axis=0)))
        assert got == want


class TestWardLabels:
    """scipy-based Ward labels must equal sklearn's AgglomerativeClustering."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_labels_equal_sklearn(self, seed: int):
        rng = np.random.default_rng(seed)
        n_rows = int(rng.integers(4, 200))
        candidates = rng.normal(size=(n_rows, int(rng.integers(2, 96))))

        for n_clusters in (2, 3, 5, 8, min(20, n_rows - 1)):
            if n_clusters < 2 or n_clusters >= n_rows:
                continue
            sk = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward"
            ).fit_predict(candidates)
            mine = _ward_labels(candidates, n_clusters)
            # Exact cluster-id equality (not just partition equality): downstream
            # results are ordered by cluster id, so the numbering must match too.
            np.testing.assert_array_equal(
                mine, sk, err_msg=f"seed={seed} n_clusters={n_clusters}"
            )

    def test_single_cluster_trivial(self):
        candidates = np.random.default_rng(0).normal(size=(10, 4))
        # n_clusters == 1 is short-circuited upstream; the tree cut still yields
        # a single group when asked directly.
        labels = _ward_labels(candidates, 1)
        assert set(labels.tolist()) == {0}

    def test_duplicate_rows_tie_break_matches_sklearn(self):
        # Duplicate rows create zero-distance ties — the hardest case for the
        # tree cut to number identically to sklearn.
        rng = np.random.default_rng(1)
        base = rng.normal(size=(12, 5))
        candidates = np.vstack([base, base[:6], base[:3]])
        for n_clusters in (2, 5, 9):
            sk = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward"
            ).fit_predict(candidates)
            np.testing.assert_array_equal(_ward_labels(candidates, n_clusters), sk)


def test_default_aggregate_never_imports_sklearn():
    """The whole point: a default run must not pull in scikit-learn."""
    code = (
        "import sys, numpy as np, pandas as pd, tsam\n"
        "idx = pd.date_range('2020-01-01', periods=10 * 24, freq='h')\n"
        "rng = np.random.default_rng(0)\n"
        "df = pd.DataFrame(rng.random((len(idx), 3)), index=idx, columns=list('abc'))\n"
        "tsam.aggregate(df, 3)\n"
        "leaked = [m for m in sys.modules if m == 'sklearn' or m.startswith('sklearn.')]\n"
        "assert not leaked, f'default aggregate imported sklearn: {leaked}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
