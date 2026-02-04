import warnings

import numpy as np
import pandas as pd
import pytest

import tsam
import tsam.timeseriesaggregation as tsam_legacy
from conftest import TESTDATA_CSV
from tsam.config import ExtremeConfig


def test_extremePeriods():
    hoursPerPeriod = 24

    noTypicalPeriods = 8

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation1 = tsam_legacy.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        rescaleClusterPeriods=False,
        extremePeriodMethod="new_cluster_center",
        addPeakMax=["GHI"],
    )

    aggregation2 = tsam_legacy.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        rescaleClusterPeriods=False,
        extremePeriodMethod="append",
        addPeakMax=["GHI"],
    )

    aggregation3 = tsam_legacy.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        rescaleClusterPeriods=False,
        extremePeriodMethod="replace_cluster_center",
        addPeakMax=["GHI"],
    )

    # make sure that the RMSE for new cluster centers (reassigning points to the exxtreme point if the distance to it is
    # smaller)is bigger than for appending just one extreme period
    np.testing.assert_array_less(
        aggregation1.accuracyIndicators().loc["GHI", "RMSE"],
        aggregation2.accuracyIndicators().loc["GHI", "RMSE"],
    )

    # make sure that the RMSE for appending the extreme period is smaller than for replacing the cluster center by the
    # extreme period (conservative assumption)
    np.testing.assert_array_less(
        aggregation2.accuracyIndicators().loc["GHI", "RMSE"],
        aggregation3.accuracyIndicators().loc["GHI", "RMSE"],
    )

    # check if addMeanMax and addMeanMin are working
    aggregation4 = tsam_legacy.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        rescaleClusterPeriods=False,
        extremePeriodMethod="append",
        addMeanMax=["GHI"],
        addMeanMin=["GHI"],
    )

    origData = aggregation4.predictOriginalData()

    np.testing.assert_array_almost_equal(
        raw.groupby(np.arange(len(raw)) // 24).mean().max().loc["GHI"],
        origData.groupby(np.arange(len(origData)) // 24).mean().max().loc["GHI"],
        decimal=6,
    )

    np.testing.assert_array_almost_equal(
        raw.groupby(np.arange(len(raw)) // 24).mean().min().loc["GHI"],
        origData.groupby(np.arange(len(origData)) // 24).mean().min().loc["GHI"],
        decimal=6,
    )


def test_preserve_n_clusters_exact_clusters_append():
    """Final n_clusters equals requested when preserve_n_clusters=True with append method."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    n_clusters = 10
    result = tsam.aggregate(
        raw,
        n_clusters=n_clusters,
        extremes=ExtremeConfig(
            method="append",
            max_value=["GHI"],
            min_value=["T"],
            preserve_n_clusters=True,
        ),
    )

    # With preserve_n_clusters=True, final cluster count should equal n_clusters
    assert result.n_clusters == n_clusters


def test_preserve_n_clusters_exact_clusters_new_cluster():
    """Final n_clusters equals requested when preserve_n_clusters=True with new_cluster method."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    n_clusters = 10
    result = tsam.aggregate(
        raw,
        n_clusters=n_clusters,
        extremes=ExtremeConfig(
            method="new_cluster",
            max_value=["GHI"],
            preserve_n_clusters=True,
        ),
    )

    # With preserve_n_clusters=True, final cluster count should equal n_clusters
    assert result.n_clusters == n_clusters


def test_preserve_n_clusters_false_adds_extra_clusters():
    """Default behavior: extremes are added on top of n_clusters."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    n_clusters = 10
    result = tsam.aggregate(
        raw,
        n_clusters=n_clusters,
        extremes=ExtremeConfig(
            method="append",
            max_value=["GHI"],
            min_value=["T"],
            preserve_n_clusters=False,  # Default
        ),
    )

    # With preserve_n_clusters=False (default), extremes are added on top
    # So final count should be > n_clusters (n_clusters + n_extremes)
    assert result.n_clusters > n_clusters


def test_preserve_n_clusters_validation_error():
    """Error if n_clusters <= n_extremes when preserve_n_clusters=True."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    with pytest.raises(ValueError, match="must be greater than"):
        tsam.aggregate(
            raw,
            n_clusters=2,
            extremes=ExtremeConfig(
                max_value=["GHI", "T", "Wind"],  # 3 extremes
                preserve_n_clusters=True,
            ),
        )


def test_preserve_n_clusters_preserves_extremes():
    """Extreme values are still preserved with preserve_n_clusters=True."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    result = tsam.aggregate(
        raw,
        n_clusters=10,
        extremes=ExtremeConfig(
            method="append",
            max_value=["GHI"],
            preserve_n_clusters=True,
        ),
        preserve_column_means=False,  # Don't rescale to check raw extreme preservation
    )

    # The maximum GHI value should be preserved in the typical periods
    orig_max = raw["GHI"].max()
    typical_max = result.cluster_representatives["GHI"].max()

    np.testing.assert_almost_equal(orig_max, typical_max, decimal=5)


def test_preserve_n_clusters_serialization():
    """ExtremeConfig with preserve_n_clusters serializes correctly."""
    config = ExtremeConfig(
        method="append",
        max_value=["Load"],
        preserve_n_clusters=True,
    )

    d = config.to_dict()
    assert d["preserve_n_clusters"] is True

    config2 = ExtremeConfig.from_dict(d)
    assert config2.preserve_n_clusters is True


def test_preserve_n_clusters_default_none_with_future_warning():
    """Default value of preserve_n_clusters is None with FutureWarning."""
    # Creating ExtremeConfig with extremes but without explicit preserve_n_clusters
    # should emit a FutureWarning
    with pytest.warns(FutureWarning, match="preserve_n_clusters currently defaults"):
        config = ExtremeConfig(max_value=["Load"])

    # The raw value should be None
    assert config.preserve_n_clusters is None

    # But effective value should be False (current default behavior)
    assert config._effective_preserve_n_clusters is False

    # to_dict should not include it when None
    d = config.to_dict()
    assert "preserve_n_clusters" not in d


def test_preserve_n_clusters_explicit_false_no_warning():
    """Setting preserve_n_clusters=False explicitly should not warn."""
    # No warning when explicitly set
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        config = ExtremeConfig(max_value=["Load"], preserve_n_clusters=False)

    assert config.preserve_n_clusters is False
    assert config._effective_preserve_n_clusters is False

    # to_dict should include it when explicitly False
    d = config.to_dict()
    assert d["preserve_n_clusters"] is False


# =============================================================================
# Comprehensive tests for preserve_n_clusters with multiple columns and methods
# =============================================================================


def assert_actual_cluster_count(result, expected_n_clusters):
    """Verify the actual cluster count matches expected from multiple data sources.

    Checks:
    1. result.n_clusters property
    2. Unique cluster indices in cluster_representatives
    3. Number of keys in cluster_weights
    4. Unique values in cluster_assignments
    """
    # 1. The n_clusters property
    assert result.n_clusters == expected_n_clusters, (
        f"n_clusters property: {result.n_clusters} != {expected_n_clusters}"
    )

    # 2. Actual unique clusters in representatives DataFrame
    actual_clusters_in_repr = result.cluster_representatives.index.get_level_values(
        0
    ).nunique()
    assert actual_clusters_in_repr == expected_n_clusters, (
        f"Unique clusters in representatives: {actual_clusters_in_repr} != {expected_n_clusters}"
    )

    # 3. Number of cluster_weights entries
    assert len(result.cluster_weights) == expected_n_clusters, (
        f"cluster_weights entries: {len(result.cluster_weights)} != {expected_n_clusters}"
    )

    # 4. Unique cluster indices in assignments
    unique_assignments = len(set(result.cluster_assignments))
    assert unique_assignments == expected_n_clusters, (
        f"Unique cluster assignments: {unique_assignments} != {expected_n_clusters}"
    )


class TestPreserveNClustersMultipleColumns:
    """Test preserve_n_clusters with multiple extreme columns."""

    @pytest.fixture
    def raw_data(self):
        """Load test data."""
        return pd.read_csv(TESTDATA_CSV, index_col=0)

    # -------------------------------------------------------------------------
    # Tests for APPEND method
    # -------------------------------------------------------------------------

    def test_append_multiple_max_value_columns(self, raw_data):
        """Append method with multiple max_value columns preserves n_clusters."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T", "Wind"],  # 3 max_value extremes
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_append_multiple_min_value_columns(self, raw_data):
        """Append method with multiple min_value columns preserves n_clusters."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                min_value=["GHI", "T", "Wind"],  # 3 min_value extremes
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_append_multiple_max_period_columns(self, raw_data):
        """Append method with multiple max_period columns preserves n_clusters."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_period=["GHI", "T", "Wind"],  # 3 max_period extremes
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_append_multiple_min_period_columns(self, raw_data):
        """Append method with multiple min_period columns preserves n_clusters."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                min_period=["GHI", "T", "Wind"],  # 3 min_period extremes
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_append_mixed_extreme_types(self, raw_data):
        """Append method with all extreme types mixed preserves n_clusters."""
        n_clusters = 15
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T"],
                min_value=["Wind"],
                max_period=["GHI"],
                min_period=["T"],
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_append_same_column_different_types(self, raw_data):
        """Append method: same column with max/min value and period."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI"],
                min_value=["GHI"],
                max_period=["GHI"],
                min_period=["GHI"],
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    # -------------------------------------------------------------------------
    # Tests for NEW_CLUSTER method
    # -------------------------------------------------------------------------

    def test_new_cluster_multiple_max_value_columns(self, raw_data):
        """New_cluster method with multiple max_value columns preserves n_clusters."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_multiple_min_value_columns(self, raw_data):
        """New_cluster method with multiple min_value columns preserves n_clusters."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                min_value=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_multiple_max_period_columns(self, raw_data):
        """New_cluster method with multiple max_period columns preserves n_clusters."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_period=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_multiple_min_period_columns(self, raw_data):
        """New_cluster method with multiple min_period columns preserves n_clusters."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                min_period=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_mixed_extreme_types(self, raw_data):
        """New_cluster method with all extreme types mixed preserves n_clusters."""
        n_clusters = 15
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T"],
                min_value=["Wind"],
                max_period=["GHI"],
                min_period=["T"],
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_same_column_different_types(self, raw_data):
        """New_cluster method: same column with max/min value and period."""
        n_clusters = 12
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI"],
                min_value=["GHI"],
                max_period=["GHI"],
                min_period=["GHI"],
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)


class TestPreserveNClustersExtremesPreserved:
    """Test that extreme values are actually preserved with multiple columns."""

    @pytest.fixture
    def raw_data(self):
        """Load test data."""
        return pd.read_csv(TESTDATA_CSV, index_col=0)

    def test_append_multiple_max_values_preserved(self, raw_data):
        """Append method: multiple max values are preserved in output."""
        result = tsam.aggregate(
            raw_data,
            n_clusters=12,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
            preserve_column_means=False,
        )

        # Both max values should be preserved
        for col in ["GHI", "T"]:
            orig_max = raw_data[col].max()
            typical_max = result.cluster_representatives[col].max()
            np.testing.assert_almost_equal(orig_max, typical_max, decimal=5)

    def test_append_multiple_min_values_preserved(self, raw_data):
        """Append method: multiple min values are preserved in output."""
        result = tsam.aggregate(
            raw_data,
            n_clusters=12,
            extremes=ExtremeConfig(
                method="append",
                min_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
            preserve_column_means=False,
        )

        # Both min values should be preserved
        for col in ["GHI", "T"]:
            orig_min = raw_data[col].min()
            typical_min = result.cluster_representatives[col].min()
            np.testing.assert_almost_equal(orig_min, typical_min, decimal=5)

    def test_append_max_period_values_preserved(self, raw_data):
        """Append method: max period (mean) values are preserved."""
        result = tsam.aggregate(
            raw_data,
            n_clusters=12,
            period_duration=24,
            extremes=ExtremeConfig(
                method="append",
                max_period=["GHI"],
                preserve_n_clusters=True,
            ),
            preserve_column_means=False,
        )

        # The period with max mean should be preserved
        orig_period_means = (
            raw_data["GHI"].groupby(np.arange(len(raw_data)) // 24).mean()
        )
        orig_max_mean = orig_period_means.max()

        typical_period_means = (
            result.cluster_representatives["GHI"]
            .groupby(np.arange(len(result.cluster_representatives)) // 24)
            .mean()
        )
        typical_max_mean = typical_period_means.max()

        np.testing.assert_almost_equal(orig_max_mean, typical_max_mean, decimal=5)

    def test_append_min_period_values_preserved(self, raw_data):
        """Append method: min period (mean) values are preserved."""
        result = tsam.aggregate(
            raw_data,
            n_clusters=12,
            period_duration=24,
            extremes=ExtremeConfig(
                method="append",
                min_period=["GHI"],
                preserve_n_clusters=True,
            ),
            preserve_column_means=False,
        )

        # The period with min mean should be preserved
        orig_period_means = (
            raw_data["GHI"].groupby(np.arange(len(raw_data)) // 24).mean()
        )
        orig_min_mean = orig_period_means.min()

        typical_period_means = (
            result.cluster_representatives["GHI"]
            .groupby(np.arange(len(result.cluster_representatives)) // 24)
            .mean()
        )
        typical_min_mean = typical_period_means.min()

        np.testing.assert_almost_equal(orig_min_mean, typical_min_mean, decimal=5)

    def test_new_cluster_multiple_max_values_preserved(self, raw_data):
        """New_cluster method: multiple max values are preserved in output."""
        result = tsam.aggregate(
            raw_data,
            n_clusters=12,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
            preserve_column_means=False,
        )

        # Both max values should be preserved
        for col in ["GHI", "T"]:
            orig_max = raw_data[col].max()
            typical_max = result.cluster_representatives[col].max()
            np.testing.assert_almost_equal(orig_max, typical_max, decimal=5)

    def test_new_cluster_multiple_min_values_preserved(self, raw_data):
        """New_cluster method: multiple min values are preserved in output."""
        result = tsam.aggregate(
            raw_data,
            n_clusters=12,
            extremes=ExtremeConfig(
                method="new_cluster",
                min_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
            preserve_column_means=False,
        )

        # Both min values should be preserved
        for col in ["GHI", "T"]:
            orig_min = raw_data[col].min()
            typical_min = result.cluster_representatives[col].min()
            np.testing.assert_almost_equal(orig_min, typical_min, decimal=5)


class TestPreserveNClustersDeduplication:
    """Test that overlapping extreme periods are correctly deduplicated."""

    @pytest.fixture
    def raw_data(self):
        """Load test data."""
        return pd.read_csv(TESTDATA_CSV, index_col=0)

    def test_append_deduplication_with_false_more_clusters(self, raw_data):
        """preserve_n_clusters=False should add more clusters than True."""
        n_clusters = 10

        # With preserve_n_clusters=True
        result_true = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )

        # With preserve_n_clusters=False
        result_false = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T", "Wind"],
                preserve_n_clusters=False,
            ),
        )

        assert_actual_cluster_count(result_true, n_clusters)
        assert result_false.n_clusters > n_clusters

    def test_new_cluster_deduplication_with_false_more_clusters(self, raw_data):
        """New_cluster: preserve_n_clusters=False should add more clusters than True."""
        n_clusters = 10

        # With preserve_n_clusters=True
        result_true = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )

        # With preserve_n_clusters=False
        result_false = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T", "Wind"],
                preserve_n_clusters=False,
            ),
        )

        assert_actual_cluster_count(result_true, n_clusters)
        assert result_false.n_clusters > n_clusters


class TestPreserveNClustersEdgeCases:
    """Test edge cases for preserve_n_clusters."""

    @pytest.fixture
    def raw_data(self):
        """Load test data."""
        return pd.read_csv(TESTDATA_CSV, index_col=0)

    def test_append_n_clusters_equals_extremes_plus_one(self, raw_data):
        """Minimum viable n_clusters: exactly extremes + 1."""
        # Use 4 extremes, so n_clusters must be at least 5
        n_clusters = 5
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T", "Wind"],
                min_value=["GHI"],  # 4 potential extremes (may dedupe)
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_n_clusters_equals_extremes_plus_one(self, raw_data):
        """New_cluster: minimum viable n_clusters: exactly extremes + 1."""
        n_clusters = 5
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T", "Wind"],
                min_value=["GHI"],
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_append_validation_error_with_many_extremes(self, raw_data):
        """Append: error if n_clusters <= n_extremes with many extremes."""
        # 6 potential extremes (may be less due to deduplication)
        with pytest.raises(ValueError, match="must be greater than"):
            tsam.aggregate(
                raw_data,
                n_clusters=3,
                extremes=ExtremeConfig(
                    method="append",
                    max_value=["GHI", "T", "Wind"],
                    min_value=["GHI", "T", "Wind"],  # 6 potential extremes
                    preserve_n_clusters=True,
                ),
            )

    def test_new_cluster_validation_error_with_many_extremes(self, raw_data):
        """New_cluster: error if n_clusters <= n_extremes with many extremes."""
        with pytest.raises(ValueError, match="must be greater than"):
            tsam.aggregate(
                raw_data,
                n_clusters=3,
                extremes=ExtremeConfig(
                    method="new_cluster",
                    max_value=["GHI", "T", "Wind"],
                    min_value=["GHI", "T", "Wind"],
                    preserve_n_clusters=True,
                ),
            )

    def test_append_all_columns_all_types(self, raw_data):
        """Append: all columns with all extreme types."""
        columns = list(raw_data.columns)
        n_clusters = 20  # Enough for many extremes
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=columns,
                min_value=columns,
                max_period=columns,
                min_period=columns,
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_all_columns_all_types(self, raw_data):
        """New_cluster: all columns with all extreme types."""
        columns = list(raw_data.columns)
        n_clusters = 20
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=columns,
                min_value=columns,
                max_period=columns,
                min_period=columns,
                preserve_n_clusters=True,
            ),
        )

        assert_actual_cluster_count(result, n_clusters)


def create_multi_column_data(n_columns=10, n_days=365, hours_per_day=24, seed=42):
    """Create synthetic data with many columns to test extreme collisions."""
    np.random.seed(seed)
    n_timesteps = n_days * hours_per_day

    # Create index
    index = pd.date_range("2020-01-01", periods=n_timesteps, freq="h")

    # Create columns with different patterns
    data = {}
    for i in range(n_columns):
        # Mix of patterns: some correlated, some independent
        base = np.sin(np.linspace(0, 4 * np.pi * n_days / 365, n_timesteps))
        noise = np.random.randn(n_timesteps) * 0.3
        seasonal = np.sin(np.linspace(0, 2 * np.pi, n_timesteps)) * (i % 3)
        data[f"col_{i}"] = base + noise + seasonal + i * 0.1

    return pd.DataFrame(data, index=index)


class TestPreserveNClustersExtremeAsClusterCenter:
    """Test edge case where extreme period is already a cluster center.

    This tests the bug where _countExtremePeriods counts ALL extremes,
    but _addExtremePeriods skips extremes that are already cluster centers.
    This can cause n_clusters mismatch.
    """

    @pytest.fixture
    def raw_data(self):
        """Load test data."""
        return pd.read_csv(TESTDATA_CSV, index_col=0)

    @pytest.fixture
    def multi_column_data(self):
        """Create data with many columns."""
        return create_multi_column_data(n_columns=10)

    def test_append_high_n_clusters_increases_collision_chance(self, raw_data):
        """Higher n_clusters = more chance extreme is already a cluster center.

        With many clusters, there's higher probability that an extreme period
        gets selected as a cluster center, which would cause _addExtremePeriods
        to skip it (but we already reserved a slot for it).
        """
        # Use high n_clusters to increase chance of collision
        n_clusters = 50
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_high_n_clusters_increases_collision_chance(self, raw_data):
        """New_cluster: higher n_clusters = more chance extreme is already a cluster center."""
        n_clusters = 50
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_multiple_extremes_high_n_clusters(self, raw_data):
        """Multiple extremes with high n_clusters."""
        n_clusters = 100
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T", "Wind"],
                min_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_multiple_extremes_high_n_clusters(self, raw_data):
        """New_cluster: multiple extremes with high n_clusters."""
        n_clusters = 100
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T", "Wind"],
                min_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_kmedoids_extreme_likely_center(self, raw_data):
        """kmedoids selects actual data points as centers, increasing collision chance."""
        n_clusters = 20
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            cluster=tsam.ClusterConfig(method="kmedoids"),
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_kmedoids_extreme_likely_center(self, raw_data):
        """New_cluster with kmedoids: actual data points as centers."""
        n_clusters = 20
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            cluster=tsam.ClusterConfig(method="kmedoids"),
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    # -------------------------------------------------------------------------
    # Tests with MULTI-COLUMN DATA (10 columns)
    # -------------------------------------------------------------------------

    def test_append_many_columns_all_max_value(self, multi_column_data):
        """Append: max_value for all 10 columns."""
        n_clusters = 15
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_many_columns_all_max_value(self, multi_column_data):
        """New_cluster: max_value for all 10 columns."""
        n_clusters = 15
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_many_columns_all_min_value(self, multi_column_data):
        """Append: min_value for all 10 columns."""
        n_clusters = 15
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                min_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_many_columns_all_min_value(self, multi_column_data):
        """New_cluster: min_value for all 10 columns."""
        n_clusters = 15
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                min_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_many_columns_max_and_min_value(self, multi_column_data):
        """Append: max_value AND min_value for all 10 columns (20 potential extremes)."""
        n_clusters = 25
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=columns,
                min_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_many_columns_max_and_min_value(self, multi_column_data):
        """New_cluster: max_value AND min_value for all 10 columns."""
        n_clusters = 25
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=columns,
                min_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_many_columns_all_extreme_types(self, multi_column_data):
        """Append: all 4 extreme types for all 10 columns (40 potential extremes)."""
        n_clusters = 50
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=columns,
                min_value=columns,
                max_period=columns,
                min_period=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_many_columns_all_extreme_types(self, multi_column_data):
        """New_cluster: all 4 extreme types for all 10 columns."""
        n_clusters = 50
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=columns,
                min_value=columns,
                max_period=columns,
                min_period=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_many_columns_high_n_clusters(self, multi_column_data):
        """Append: many columns with very high n_clusters (max collision chance)."""
        n_clusters = 200
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=columns,
                min_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_many_columns_high_n_clusters(self, multi_column_data):
        """New_cluster: many columns with very high n_clusters."""
        n_clusters = 200
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=columns,
                min_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_many_columns_kmedoids(self, multi_column_data):
        """Append with kmedoids: many columns (medoids are actual data points)."""
        n_clusters = 20
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            cluster=tsam.ClusterConfig(method="kmedoids"),
            extremes=ExtremeConfig(
                method="append",
                max_value=columns,
                min_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_many_columns_kmedoids(self, multi_column_data):
        """New_cluster with kmedoids: many columns."""
        n_clusters = 20
        columns = list(multi_column_data.columns)
        result = tsam.aggregate(
            multi_column_data,
            n_clusters=n_clusters,
            cluster=tsam.ClusterConfig(method="kmedoids"),
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=columns,
                min_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)


class TestPreserveNClustersAdjacentPeriods:
    """Test preserve_n_clusters with adjacent_periods (contiguous) clustering."""

    @pytest.fixture
    def raw_data(self):
        """Load test data."""
        return pd.read_csv(TESTDATA_CSV, index_col=0)

    def test_adjacent_periods_warns_about_preserve_n_clusters(self, raw_data):
        """adjacent_periods with preserve_n_clusters=True should warn."""
        with pytest.warns(UserWarning, match="adjacent_periods.*contiguity"):
            tsam.aggregate(
                raw_data,
                n_clusters=10,
                cluster=tsam.ClusterConfig(method="contiguous"),
                extremes=ExtremeConfig(
                    method="append",
                    max_value=["GHI"],
                    preserve_n_clusters=True,
                ),
            )

    def test_adjacent_periods_still_works_without_preserve(self, raw_data):
        """adjacent_periods without preserve_n_clusters should work normally."""
        result = tsam.aggregate(
            raw_data,
            n_clusters=10,
            cluster=tsam.ClusterConfig(method="contiguous"),
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI"],
                preserve_n_clusters=False,
            ),
        )
        # Should have at least n_clusters (extremes added on top)
        assert result.n_clusters >= 10


class TestPreserveNClustersMultipleExtremesDetailed:
    """Detailed tests to verify exact extreme counting with multiple columns."""

    @pytest.fixture
    def raw_data(self):
        """Load test data."""
        return pd.read_csv(TESTDATA_CSV, index_col=0)

    # -------------------------------------------------------------------------
    # APPEND method - detailed tests
    # -------------------------------------------------------------------------

    def test_append_two_max_value_columns_preserve(self, raw_data):
        """Append: 2 max_value columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_three_max_value_columns_preserve(self, raw_data):
        """Append: 3 max_value columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_four_max_value_columns_preserve(self, raw_data):
        """Append: 4 max_value columns (all) with preserve_n_clusters=True."""
        n_clusters = 10
        columns = list(raw_data.columns)
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_two_min_value_columns_preserve(self, raw_data):
        """Append: 2 min_value columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                min_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_three_min_value_columns_preserve(self, raw_data):
        """Append: 3 min_value columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                min_value=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_two_max_period_columns_preserve(self, raw_data):
        """Append: 2 max_period columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_period=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_two_min_period_columns_preserve(self, raw_data):
        """Append: 2 min_period columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                min_period=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_max_and_min_value_same_column_preserve(self, raw_data):
        """Append: max_value and min_value for same column."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI"],
                min_value=["GHI"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_max_and_min_value_different_columns_preserve(self, raw_data):
        """Append: max_value and min_value for different columns."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI"],
                min_value=["T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_max_value_and_max_period_same_column_preserve(self, raw_data):
        """Append: max_value and max_period for same column."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI"],
                max_period=["GHI"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_all_four_types_single_column_preserve(self, raw_data):
        """Append: all 4 extreme types for single column."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI"],
                min_value=["GHI"],
                max_period=["GHI"],
                min_period=["GHI"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_append_multiple_columns_multiple_types_preserve(self, raw_data):
        """Append: multiple columns with multiple extreme types."""
        n_clusters = 15
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="append",
                max_value=["GHI", "T"],
                min_value=["GHI", "Wind"],
                max_period=["T"],
                min_period=["Wind"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    # -------------------------------------------------------------------------
    # NEW_CLUSTER method - detailed tests
    # -------------------------------------------------------------------------

    def test_new_cluster_two_max_value_columns_preserve(self, raw_data):
        """New_cluster: 2 max_value columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_three_max_value_columns_preserve(self, raw_data):
        """New_cluster: 3 max_value columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_four_max_value_columns_preserve(self, raw_data):
        """New_cluster: 4 max_value columns (all) with preserve_n_clusters=True."""
        n_clusters = 10
        columns = list(raw_data.columns)
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=columns,
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_two_min_value_columns_preserve(self, raw_data):
        """New_cluster: 2 min_value columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                min_value=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_three_min_value_columns_preserve(self, raw_data):
        """New_cluster: 3 min_value columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                min_value=["GHI", "T", "Wind"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_two_max_period_columns_preserve(self, raw_data):
        """New_cluster: 2 max_period columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_period=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_two_min_period_columns_preserve(self, raw_data):
        """New_cluster: 2 min_period columns with preserve_n_clusters=True."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                min_period=["GHI", "T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_max_and_min_value_same_column_preserve(self, raw_data):
        """New_cluster: max_value and min_value for same column."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI"],
                min_value=["GHI"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_max_and_min_value_different_columns_preserve(self, raw_data):
        """New_cluster: max_value and min_value for different columns."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI"],
                min_value=["T"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_max_value_and_max_period_same_column_preserve(self, raw_data):
        """New_cluster: max_value and max_period for same column."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI"],
                max_period=["GHI"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_all_four_types_single_column_preserve(self, raw_data):
        """New_cluster: all 4 extreme types for single column."""
        n_clusters = 10
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI"],
                min_value=["GHI"],
                max_period=["GHI"],
                min_period=["GHI"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)

    def test_new_cluster_multiple_columns_multiple_types_preserve(self, raw_data):
        """New_cluster: multiple columns with multiple extreme types."""
        n_clusters = 15
        result = tsam.aggregate(
            raw_data,
            n_clusters=n_clusters,
            extremes=ExtremeConfig(
                method="new_cluster",
                max_value=["GHI", "T"],
                min_value=["GHI", "Wind"],
                max_period=["T"],
                min_period=["Wind"],
                preserve_n_clusters=True,
            ),
        )
        assert_actual_cluster_count(result, n_clusters)


if __name__ == "__main__":
    test_extremePeriods()
