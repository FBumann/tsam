"""End-to-end workflows, as downstream code actually runs them.

`test_bench.py` isolates one dimension per benchmark, which is what makes it
plottable but also means no single number there answers "how long does my job
take". These benchmarks do the opposite: each one is a whole task a real
caller performs, including the result fields that task reads afterwards.

Two things they exist to catch that the dimensional suite cannot:

- **Lazy work.** ``accuracy`` is a cached property and ``disaggregate()`` is
  called by the caller, so an `aggregate()`-only benchmark charges for
  neither. A workflow that reports metrics or expands an optimization result
  pays for both.
- **Per-call overhead.** Tuning sweeps and scenario batches call
  ``aggregate()`` tens of times on small inputs, where fixed overhead
  dominates and a win on one large call may not show up at all.

Usage::

    pytest benchmarks/test_workflows.py --benchmark-only
    pytest benchmarks/test_workflows.py --benchmark-only --large
"""

from __future__ import annotations

import pandas as pd
import pytest
from test_bench import _fine_data, _testdata, _wide_columns

from tsam import (
    ClusterConfig,
    Distribution,
    ExtremeConfig,
    SegmentConfig,
    aggregate,
)

HEAVY = {"rounds": 3, "iterations": 1, "warmup_rounds": 1}
LIGHT = {"rounds": 10, "iterations": 1, "warmup_rounds": 1}


def _fine_production_data() -> pd.DataFrame:
    """Two years x 400 columns of tiled FINE profiles (the issue #49 shape)."""
    wide = pd.concat([_fine_data().add_suffix(f"_{i}") for i in range(10)], axis=1)
    data = pd.concat([wide] * 2, ignore_index=True)
    data.index = pd.date_range("2020-01-01", periods=len(data), freq="h")
    return data


def _read_like_fine(result) -> tuple:
    """The four things FINE reads off the aggregation and nothing else.

    ``fine/energySystemModel.py`` touches ``clusterPeriodDict``,
    ``segmentDurationDict``, ``clusterPeriodIdx`` and ``clusterOrder``. It
    never asks for the accuracy indicators, and never reads the reconstructed
    series back.
    """
    representatives = result.cluster_representatives
    return (
        representatives.to_dict(),
        representatives.index.get_level_values(-1).tolist(),
        result.n_clusters,
        list(result.cluster_assignments),
    )


@pytest.mark.benchmark(group="workflow")
def test_workflow_fine_esm(benchmark):
    """FINE's ``aggregateTemporally()`` at its own example's scale (8760 x 40).

    40 typical days, hierarchical clustering with duration representation,
    12 segments, no rescaling — then the fields FINE reads.
    """
    data = _fine_data()
    benchmark.pedantic(
        lambda: _read_like_fine(
            aggregate(
                data,
                40,
                cluster=ClusterConfig(
                    method="hierarchical", representation="distribution"
                ),
                segments=SegmentConfig(n_segments=12),
                preserve_column_means=False,
            )
        ),
        **HEAVY,
    )


@pytest.mark.large
@pytest.mark.benchmark(group="workflow")
def test_workflow_fine_esm_production(benchmark):
    """The same workflow at production scale: two years x 400 columns.

    Global-scope duration representation and peak-demand extreme periods, as
    a multi-region model with per-region peaks would configure it.
    """
    data = _fine_production_data()
    peaks = [c for c in data.columns if c.startswith("ElecDemand")]
    benchmark.extra_info["n_timesteps"] = len(data)
    benchmark.extra_info["n_columns"] = data.shape[1]
    benchmark.pedantic(
        lambda: _read_like_fine(
            aggregate(
                data,
                40,
                cluster=ClusterConfig(
                    method="hierarchical", representation=Distribution(scope="global")
                ),
                segments=SegmentConfig(n_segments=12),
                extremes=ExtremeConfig(method="append", max_value=peaks),
                preserve_column_means=False,
            )
        ),
        **HEAVY,
    )


@pytest.mark.benchmark(group="workflow")
def test_workflow_optimization_roundtrip(benchmark):
    """Aggregate, then expand 50 solved variables back to the full year.

    The shape of an energy-system run: cluster once, hand the typical periods
    to a solver, then map every optimization variable back onto the original
    time index. Only the expansion is repeated per variable.
    """
    data = _wide_columns(20)
    n_variables = 50

    def run():
        result = aggregate(data, 36)
        solution = result.cluster_representatives
        return [result.disaggregate(solution) for _ in range(n_variables)]

    benchmark.extra_info["n_variables"] = n_variables
    benchmark.pedantic(run, **HEAVY)


@pytest.mark.benchmark(group="workflow")
def test_workflow_accuracy_report(benchmark):
    """Aggregate and report the error metrics, as a config comparison does.

    ``accuracy`` is lazy, so this is the only workflow that pays for it.
    """
    data = _wide_columns(48)

    def run():
        result = aggregate(data, 8)
        metrics = result.accuracy
        return metrics.rmse, metrics.mae, metrics.rmse_duration

    benchmark.pedantic(run, **LIGHT)


@pytest.mark.benchmark(group="workflow")
def test_workflow_scenario_batch(benchmark):
    """Eight scenario years aggregated in one go, metrics collected per run.

    Batch jobs multiply per-call overhead by the number of scenarios rather
    than amortizing it, so a fixed cost matters here that a single large
    aggregation would hide.
    """
    scenarios = [_wide_columns(64).iloc[:, i * 8 : (i + 1) * 8] for i in range(8)]

    def run():
        return [aggregate(s, 8).accuracy.rmse.mean() for s in scenarios]

    benchmark.extra_info["n_scenarios"] = 8
    benchmark.pedantic(run, **HEAVY)


@pytest.mark.benchmark(group="workflow")
def test_workflow_tuning_sweep(benchmark):
    """A hyperparameter sweep: 12 cluster counts scored by duration-curve error.

    ``tsam.tuning`` drives this internally; the loop is written out here so the
    workflow runs identically on every version being compared.
    """
    data = _testdata()
    candidates = [4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 60]

    def run():
        scored = []
        for k in candidates:
            result = aggregate(data, k)
            scored.append((k, float(result.accuracy.rmse_duration.mean())))
        return min(scored, key=lambda item: item[1])

    benchmark.extra_info["n_configs"] = len(candidates)
    benchmark.pedantic(run, **HEAVY)
