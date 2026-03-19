"""Tests that deprecated camelCase names still work for backward compatibility."""

import warnings

import pandas as pd
import pytest

from conftest import TESTDATA_CSV


@pytest.fixture
def raw():
    return pd.read_csv(TESTDATA_CSV, index_col=0)


class TestTimeSeriesAggregationCompat:
    """Verify old camelCase kwargs, methods, and properties still work."""

    def test_old_kwargs_accepted_with_warning(self, raw):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from tsam.timeseriesaggregation import TimeSeriesAggregation

            agg = TimeSeriesAggregation(
                timeSeries=raw,
                noTypicalPeriods=8,
                hoursPerPeriod=24,
                clusterMethod="hierarchical",
            )

        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        # Should have warnings for timeSeries, noTypicalPeriods, hoursPerPeriod, clusterMethod
        assert len(future_warnings) >= 4
        assert agg.no_typical_periods == 8
        assert agg.hours_per_period == 24
        assert agg.cluster_method == "hierarchical"

    def test_unexpected_kwarg_raises(self, raw):
        from tsam.timeseriesaggregation import TimeSeriesAggregation

        with pytest.raises(TypeError, match="Unexpected keyword arguments"):
            TimeSeriesAggregation(raw, bogusKwarg=42)

    def test_old_method_names(self, raw):
        from tsam.timeseriesaggregation import TimeSeriesAggregation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agg = TimeSeriesAggregation(raw, noTypicalPeriods=8)

        # Old method names should work
        typical = agg.createTypicalPeriods()
        assert typical is not None

        predicted = agg.predictOriginalData()
        assert predicted is not None

        acc = agg.accuracyIndicators()
        assert acc is not None

        total_acc = agg.totalAccuracyIndicators()
        assert total_acc is not None

        matching = agg.indexMatching()
        assert matching is not None

    def test_old_property_names(self, raw):
        from tsam.timeseriesaggregation import TimeSeriesAggregation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agg = TimeSeriesAggregation(raw, noTypicalPeriods=8)
            agg.createTypicalPeriods()

        # Old property names should return the same as new ones
        assert agg.stepIdx == agg.step_idx
        assert list(agg.clusterPeriodIdx) == list(agg.cluster_period_idx)
        assert list(agg.clusterOrder) == list(agg.cluster_order)
        assert agg.clusterPeriodNoOccur == agg.cluster_period_no_occur
        assert agg.clusterPeriodDict == agg.cluster_period_dict
