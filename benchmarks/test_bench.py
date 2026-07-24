"""Performance benchmarks for :func:`tsam.aggregate`.

Star design: one baseline configuration (hourly data, 8 typical days,
hierarchical clustering with medoid representation) and one dimension varied
per test function. Every parametrize value is a scalar, so each one becomes a
plottable pytest-benchmem dim; with one axis per function, ``benchmem plot
--x <dim>`` slices out exactly that function's scaling curve.

Usage::

    pytest benchmarks/ --benchmark-only                       # quick tier, timing
    pytest benchmarks/ --benchmark-only --slow                # + k-medoids, 96 columns
    pytest benchmarks/ --benchmark-only --large               # + production-sized cases
    pytest benchmarks/ --benchmark-only --benchmark-memory    # + memray peak memory
    pytest benchmarks/ --benchmark-only --benchmark-save=dev  # snapshot to .benchmarks/

    benchmem compare '*base*' '*dev*' --columns peak --diff
    benchmem plot .benchmarks/*/0001_base.json --columns peak --x n_timesteps
    benchmem plot .benchmarks/*/0001_base.json --columns peak --x n_columns
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from tsam import ClusterConfig, Distribution, ExtremeConfig, SegmentConfig, aggregate

ROOT = Path(__file__).resolve().parent.parent
TESTDATA_CSV = ROOT / "docs" / "data" / "testdata.csv"
WIDE_CSV = ROOT / "test" / "data" / "wide.csv"

N_CLUSTERS = 8
ROUNDS = 10


@lru_cache
def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _testdata() -> pd.DataFrame:
    """One year of hourly GHI/T/Wind/Load data (8760 x 4)."""
    return _load(str(TESTDATA_CSV))


def _wide_columns(n_columns: int) -> pd.DataFrame:
    """wide.csv tiled with unique column names up to ``n_columns`` (8760 rows)."""
    base = _load(str(WIDE_CSV))
    repeats = -(-n_columns // base.shape[1])
    tiled = pd.concat([base.add_suffix(f"_{i}") for i in range(repeats)], axis=1)
    return tiled.iloc[:, :n_columns]


def _years(n: int) -> pd.DataFrame:
    """testdata tiled to ``n`` years with a continuous hourly index."""
    base = _testdata()
    df = pd.concat([base] * n, ignore_index=True)
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="h")
    return df


def _quarter_hourly() -> pd.DataFrame:
    """testdata linearly interpolated to 15-min resolution (35,040 rows)."""
    base = _testdata()
    idx = pd.date_range(base.index[0], periods=len(base) * 4, freq="15min")
    return base.reindex(idx).interpolate(method="linear").ffill()


def _bench(
    benchmark, data: pd.DataFrame, n_clusters: int = N_CLUSTERS, **kwargs
) -> None:
    benchmark.pedantic(
        lambda: aggregate(data, n_clusters, **kwargs),
        rounds=ROUNDS,
        warmup_rounds=1,
    )


@pytest.mark.benchmark(group="scale-timesteps")
@pytest.mark.parametrize("n_timesteps", [672, 2190, 4380, 8760])
def test_scale_timesteps(benchmark, n_timesteps):
    """Baseline config over a growing slice of the year."""
    _bench(benchmark, _testdata().iloc[:n_timesteps])


@pytest.mark.benchmark(group="scale-columns")
@pytest.mark.parametrize(
    "n_columns",
    [
        4,
        12,
        48,
        pytest.param(96, marks=pytest.mark.slow),
        pytest.param(400, marks=pytest.mark.slow),
    ],
)
def test_scale_columns(benchmark, n_columns):
    """Baseline config over a growing number of columns (full year)."""
    _bench(benchmark, _wide_columns(n_columns))


@pytest.mark.benchmark(group="method")
@pytest.mark.parametrize(
    "method", ["averaging", "kmeans", "kmaxoids", "hierarchical", "contiguous"]
)
def test_method(benchmark, method):
    """Clustering methods on the full year, default representation each."""
    _bench(benchmark, _testdata(), cluster=ClusterConfig(method=method))


@pytest.mark.slow
@pytest.mark.benchmark(group="method")
def test_method_kmedoids(benchmark):
    """Exact k-medoids (MILP): truncated to 8 weeks to keep the solve tractable.

    Not size-comparable with :func:`test_method`; the extra_info dims record
    the reduced input so plots and tables show it.
    """
    n_timesteps = 8 * 168
    benchmark.extra_info["method"] = "kmedoids"
    benchmark.extra_info["n_timesteps"] = n_timesteps
    _bench(
        benchmark,
        _testdata().iloc[:n_timesteps],
        cluster=ClusterConfig(method="kmedoids"),
    )


@pytest.mark.benchmark(group="representation")
@pytest.mark.parametrize(
    "representation",
    [
        "mean",
        "medoid",
        "maxoid",
        "distribution",
        "distribution_global",
        "minmax_mean",
    ],
)
def test_representation(benchmark, representation):
    """Representations on hierarchical clustering, full year."""
    rep = (
        Distribution(scope="global")
        if representation == "distribution_global"
        else representation
    )
    _bench(
        benchmark,
        _testdata(),
        cluster=ClusterConfig(method="hierarchical", representation=rep),
    )


@pytest.mark.benchmark(group="feature")
@pytest.mark.parametrize(
    "feature", ["baseline", "segmentation", "extremes", "no_rescale"]
)
def test_feature(benchmark, feature):
    """Cost of each pipeline feature toggled onto the baseline, one at a time."""
    kwargs = {}
    if feature == "segmentation":
        kwargs["segments"] = SegmentConfig(n_segments=12)
    elif feature == "extremes":
        kwargs["extremes"] = ExtremeConfig(
            method="new_cluster", max_value=["Load"], min_value=["T"]
        )
    elif feature == "no_rescale":
        kwargs["preserve_column_means"] = False
    _bench(benchmark, _testdata(), **kwargs)


@pytest.mark.benchmark(group="resolution")
def test_resolution_15min(benchmark):
    """One year at 15-min resolution: 96 steps per daily period stress the reshapes."""
    benchmark.extra_info["steps_per_period"] = 96
    _bench(benchmark, _quarter_hourly())


@pytest.mark.benchmark(group="extremes-scale")
def test_extremes_3y(benchmark):
    """new_cluster extreme handling over 3 years; reassignment scales with period count."""
    benchmark.extra_info["n_timesteps"] = 3 * 8760
    _bench(
        benchmark,
        _years(3),
        extremes=ExtremeConfig(
            method="new_cluster", max_value=["Load"], min_value=["T"]
        ),
    )


LARGE_OPTS = {"rounds": 3, "iterations": 1, "warmup_rounds": 0}


def _wide_years(n_columns: int, n_years: int) -> pd.DataFrame:
    """wide.csv tiled to ``n_columns`` and ``n_years`` with a continuous index."""
    df = pd.concat([_wide_columns(n_columns)] * n_years, ignore_index=True)
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="h")
    return df


@pytest.mark.large
@pytest.mark.benchmark(group="large")
def test_large_wide(benchmark):
    """Production-sized single frame: two years hourly x 256 columns."""
    benchmark.extra_info["n_timesteps"] = 2 * 8760
    benchmark.extra_info["n_columns"] = 256
    data = _wide_years(256, 2)
    benchmark.pedantic(lambda: aggregate(data, N_CLUSTERS), **LARGE_OPTS)


@pytest.mark.large
@pytest.mark.benchmark(group="large")
def test_large_scenarios(benchmark):
    """Eight sequential year x 64-column aggregations (multi-scenario workload)."""
    benchmark.extra_info["n_slices"] = 8
    data = _wide_columns(64)
    benchmark.pedantic(
        lambda: [aggregate(data, N_CLUSTERS) for _ in range(8)], **LARGE_OPTS
    )


def _fine_data() -> pd.DataFrame:
    """FINE 8-region example: 5 profiles x 8 regions, hourly year (8760 x 40)."""
    return _load(str(ROOT / "benchmarks" / "data" / "fine_multiregional.csv.gz"))


@pytest.mark.benchmark(group="headline")
def test_fine_default(benchmark):
    """FINE's aggregateTemporally() defaults: 40 typical days, hierarchical +
    duration representation, 12 segments, no rescale."""
    _bench(
        benchmark,
        _fine_data(),
        n_clusters=40,
        cluster=ClusterConfig(method="hierarchical", representation="distribution"),
        segments=SegmentConfig(n_segments=12),
        preserve_column_means=False,
    )


@pytest.mark.benchmark(group="headline")
def test_fine_extremes(benchmark):
    """FINE defaults plus appended peak-demand extreme periods per region."""
    data = _fine_data()
    peak_columns = [c for c in data.columns if c.startswith("ElecDemand")]
    _bench(
        benchmark,
        data,
        n_clusters=40,
        cluster=ClusterConfig(method="hierarchical", representation="distribution"),
        segments=SegmentConfig(n_segments=12),
        extremes=ExtremeConfig(method="append", max_value=peak_columns),
        preserve_column_means=False,
    )


@pytest.mark.large
@pytest.mark.benchmark(group="headline")
def test_fine_production(benchmark):
    """Production-scale FINE workload (the issue #49 shape): two years x 400
    columns of tiled FINE profiles, 40 typical days, global-scope
    distribution, 12 segments, peak-demand extremes. ~6 s on tsam 3.4.2."""
    base = _fine_data()
    wide = pd.concat([base.add_suffix(f"_{i}") for i in range(10)], axis=1)
    data = pd.concat([wide] * 2, ignore_index=True)
    data.index = pd.date_range("2020-01-01", periods=len(data), freq="h")
    benchmark.extra_info["n_timesteps"] = len(data)
    benchmark.extra_info["n_columns"] = data.shape[1]
    benchmark.pedantic(
        lambda: aggregate(
            data,
            40,
            cluster=ClusterConfig(
                method="hierarchical", representation=Distribution(scope="global")
            ),
            segments=SegmentConfig(n_segments=12),
            extremes=ExtremeConfig(
                method="append",
                max_value=[c for c in data.columns if c.startswith("ElecDemand")],
            ),
            preserve_column_means=False,
        ),
        **LARGE_OPTS,
    )


@pytest.mark.benchmark(group="accuracy")
def test_accuracy(benchmark):
    """Lazy accuracy metrics on a fresh 48-column result each round."""
    benchmark.extra_info["n_columns"] = 48
    data = _wide_columns(48)
    benchmark.pedantic(
        lambda result: result.accuracy,
        setup=lambda: ((aggregate(data, N_CLUSTERS),), {}),
        rounds=ROUNDS,
        warmup_rounds=1,
    )


@pytest.mark.benchmark(group="disaggregate")
def test_disaggregate(benchmark):
    """Expanding typical-period data back to the full datetime index.

    Reconstruction of the input is eager inside ``aggregate()`` and therefore
    already timed by every other benchmark; this times the standalone
    expansion of external (e.g. optimization) results.
    """
    benchmark.extra_info["phase"] = "disaggregate"
    result = aggregate(_testdata(), N_CLUSTERS)
    benchmark.pedantic(
        lambda: result.disaggregate(result.cluster_representatives),
        rounds=ROUNDS,
        warmup_rounds=1,
    )
