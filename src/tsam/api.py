"""New simplified API for tsam aggregation."""

from __future__ import annotations

import pandas as pd

from tsam.config import (
    EXTREME_METHOD_MAPPING,
    METHOD_MAPPING,
    REPRESENTATION_MAPPING,
    ClusterConfig,
    ClusteringResult,
    ExtremeConfig,
    SegmentConfig,
)
from tsam.result import AccuracyMetrics, AggregationResult
from tsam.timeseriesaggregation import TimeSeriesAggregation


def aggregate(
    data: pd.DataFrame,
    n_clusters: int,
    *,
    period_duration: int = 24,
    timestep_duration: float | None = None,
    cluster: ClusterConfig | None = None,
    segments: SegmentConfig | None = None,
    extremes: ExtremeConfig | None = None,
    rescale: bool = True,
    round_decimals: int | None = None,
    numerical_tolerance: float = 1e-13,
) -> AggregationResult:
    """Aggregate time series data into typical periods.

    This function reduces a time series dataset to a smaller set of
    representative "typical periods" using clustering algorithms.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series data with a datetime index.
        Each column represents a different variable (e.g., solar, wind, demand).
        The index should be a DatetimeIndex with regular intervals.

    n_clusters : int
        Number of typical periods (clusters) to create.
        Higher values = more accuracy but less data reduction.
        Typical range: 4-20 for energy system models.

    period_duration : int, default 24
        Length of each period in hours.
        Common values:
        - 24: Daily periods (most common)
        - 168: Weekly periods
        - 1: Hourly periods (for sub-hourly data)

    timestep_duration : float, optional
        Time resolution of input data in hours.
        If not provided, inferred from the datetime index.
        Examples: 1.0 (hourly), 0.25 (15-minute), 0.5 (30-minute)

    cluster : ClusterConfig, optional
        Clustering configuration. If not provided, uses defaults:
        - method: "hierarchical"
        - representation: "medoid"

    segments : SegmentConfig, optional
        Segmentation configuration for reducing temporal resolution
        within periods. If not provided, no segmentation is applied.

    extremes : ExtremeConfig, optional
        Configuration for preserving extreme periods.
        If not provided, no extreme period handling is applied.

    rescale : bool, default True
        Rescale typical periods to match the original data's mean.
        Preserves total energy/load values across the aggregation.

    round_decimals : int, optional
        Round output values to this many decimal places.
        If not provided, no rounding is applied.

    numerical_tolerance : float, default 1e-13
        Tolerance for numerical precision issues.
        Controls when warnings are raised for aggregated values exceeding
        the original time series bounds. Increase this value to silence
        warnings caused by floating-point precision errors.

    Returns
    -------
    AggregationResult
        Object containing:
        - cluster_representatives: DataFrame with aggregated periods
        - cluster_assignments: Which cluster each original period belongs to
        - cluster_weights: Occurrence count per cluster
        - accuracy: RMSE, MAE metrics
        - Methods: reconstruct(), to_dict()

    Raises
    ------
    ValueError
        If input data is invalid or parameters are inconsistent.
    TypeError
        If parameter types are incorrect.

    Examples
    --------
    Basic usage with defaults:

    >>> import tsam
    >>> result = tsam.aggregate(df, n_clusters=8)
    >>> typical = result.cluster_representatives

    With custom clustering:

    >>> from tsam import aggregate, ClusterConfig
    >>> result = aggregate(
    ...     df,
    ...     n_clusters=8,
    ...     cluster=ClusterConfig(method="kmeans", representation="mean"),
    ... )

    With segmentation (reduce to 12 timesteps per period):

    >>> from tsam import aggregate, SegmentConfig
    >>> result = aggregate(
    ...     df,
    ...     n_clusters=8,
    ...     segments=SegmentConfig(n_segments=12),
    ... )

    Preserving peak demand periods:

    >>> from tsam import aggregate, ExtremeConfig
    >>> result = aggregate(
    ...     df,
    ...     n_clusters=8,
    ...     extremes=ExtremeConfig(max_value=["demand"]),
    ... )

    Transferring assignments to new data:

    >>> result1 = aggregate(df_wind, n_clusters=8)
    >>> result2 = result1.clustering.apply(df_all)

    See Also
    --------
    ClusterConfig : Clustering algorithm configuration
    SegmentConfig : Temporal segmentation configuration
    ExtremeConfig : Extreme period preservation configuration
    AggregationResult : Result object with all outputs
    """
    # Validate input
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")

    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError(f"n_clusters must be a positive integer, got {n_clusters}")

    if not isinstance(period_duration, int) or period_duration < 1:
        raise ValueError(
            f"period_duration must be a positive integer, got {period_duration}"
        )

    # Apply defaults
    if cluster is None:
        cluster = ClusterConfig()

    # Validate segments against data
    if segments is not None:
        # Calculate timesteps per period
        if timestep_duration is not None:
            timesteps_per_period = int(period_duration / timestep_duration)
        else:
            # Infer resolution from data index
            if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
                inferred_resolution = (
                    data.index[1] - data.index[0]
                ).total_seconds() / 3600
                timesteps_per_period = int(period_duration / inferred_resolution)
            else:
                # Fall back to assuming hourly resolution
                timesteps_per_period = period_duration

        if segments.n_segments > timesteps_per_period:
            raise ValueError(
                f"n_segments ({segments.n_segments}) cannot exceed "
                f"timesteps per period ({timesteps_per_period})"
            )

    # Validate extreme columns exist in data
    if extremes is not None:
        all_extreme_cols = (
            extremes.max_value
            + extremes.min_value
            + extremes.max_period
            + extremes.min_period
        )
        missing = set(all_extreme_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Extreme period columns not found in data: {missing}")

    # Validate weight columns exist
    if cluster.weights is not None:
        missing = set(cluster.weights.keys()) - set(data.columns)
        if missing:
            raise ValueError(f"Weight columns not found in data: {missing}")

    # Build old API parameters
    old_params = _build_old_params(
        data=data,
        n_clusters=n_clusters,
        period_duration=period_duration,
        timestep_duration=timestep_duration,
        cluster=cluster,
        segments=segments,
        extremes=extremes,
        rescale=rescale,
        round_decimals=round_decimals,
        numerical_tolerance=numerical_tolerance,
    )

    # Run aggregation using old implementation
    agg = TimeSeriesAggregation(**old_params)
    cluster_representatives = agg.createTypicalPeriods()

    # Build accuracy metrics
    accuracy_df = agg.accuracyIndicators()

    # Build rescale deviations DataFrame
    rescale_deviations_dict = getattr(agg, "_rescaleDeviations", {})
    if rescale_deviations_dict:
        rescale_deviations = pd.DataFrame.from_dict(
            rescale_deviations_dict, orient="index"
        )
        rescale_deviations.index.name = "column"
    else:
        rescale_deviations = pd.DataFrame(
            columns=["deviation_pct", "converged", "iterations"]
        )

    accuracy = AccuracyMetrics(
        rmse=accuracy_df["RMSE"],
        mae=accuracy_df["MAE"],
        rmse_duration=accuracy_df["RMSE_duration"],
        rescale_deviations=rescale_deviations,
    )

    # Build ClusteringResult
    clustering_result = _build_clustering_result(
        agg=agg,
        n_segments=segments.n_segments if segments else None,
        cluster_config=cluster,
        segment_config=segments,
        extremes_config=extremes,
        rescale=rescale,
        timestep_duration=timestep_duration,
    )

    # Build result object
    return AggregationResult(
        cluster_representatives=cluster_representatives,
        cluster_weights=dict(agg.clusterPeriodNoOccur),
        n_timesteps_per_period=agg.timeStepsPerPeriod,
        segment_durations=agg.segmentDurationDict if segments else None,
        accuracy=accuracy,
        clustering_duration=getattr(agg, "clusteringDuration", 0.0),
        clustering=clustering_result,
        _aggregation=agg,
    )


def _build_clustering_result(
    agg: TimeSeriesAggregation,
    n_segments: int | None,
    cluster_config: ClusterConfig,
    segment_config: SegmentConfig | None,
    extremes_config: ExtremeConfig | None,
    rescale: bool,
    timestep_duration: float | None,
) -> ClusteringResult:
    """Build ClusteringResult from a TimeSeriesAggregation object."""
    # Get cluster centers (convert to Python ints for JSON serialization)
    cluster_centers: tuple[int, ...] | None = None
    if agg.clusterCenterIndices is not None:
        cluster_centers = tuple(int(x) for x in agg.clusterCenterIndices)

    # Compute segment data if segmentation was used
    segment_order: tuple[tuple[int, ...], ...] | None = None
    segment_durations: tuple[tuple[int, ...], ...] | None = None
    segment_centers: tuple[tuple[int, ...], ...] | None = None

    if n_segments is not None and hasattr(agg, "segmentedNormalizedTypicalPeriods"):
        segmented_df = agg.segmentedNormalizedTypicalPeriods
        segment_order_list = []
        segment_durations_list = []

        for period_idx in segmented_df.index.get_level_values(0).unique():
            period_data = segmented_df.loc[period_idx]
            # Index levels: Segment Step, Segment Duration, Original Start Step
            assignments = []
            durations = []
            for seg_step, seg_dur, _orig_start in period_data.index:
                assignments.extend([int(seg_step)] * int(seg_dur))
                durations.append(int(seg_dur))
            segment_order_list.append(tuple(assignments))
            segment_durations_list.append(tuple(durations))

        segment_order = tuple(segment_order_list)
        segment_durations = tuple(segment_durations_list)

        # Extract segment center indices (only available for medoid/maxoid representations)
        if (
            hasattr(agg, "segmentCenterIndices")
            and agg.segmentCenterIndices is not None
        ):
            # Check if any period has center indices (None for mean representation)
            if all(pc is not None for pc in agg.segmentCenterIndices):
                segment_centers = tuple(
                    tuple(int(x) for x in period_centers)
                    for period_centers in agg.segmentCenterIndices
                )

    # Extract representation from configs
    representation = cluster_config.get_representation()
    segment_representation = segment_config.representation if segment_config else None

    return ClusteringResult(
        period_duration=agg.hoursPerPeriod,
        cluster_assignments=tuple(int(x) for x in agg.clusterOrder),
        cluster_centers=cluster_centers,
        segment_assignments=segment_order,
        segment_durations=segment_durations,
        segment_centers=segment_centers,
        rescale=rescale,
        representation=representation,
        segment_representation=segment_representation,
        timestep_duration=timestep_duration,
        cluster_config=cluster_config,
        segment_config=segment_config,
        extremes_config=extremes_config,
    )


def _build_old_params(
    data: pd.DataFrame,
    n_clusters: int,
    period_duration: int,
    timestep_duration: float | None,
    cluster: ClusterConfig,
    segments: SegmentConfig | None,
    extremes: ExtremeConfig | None,
    rescale: bool,
    round_decimals: int | None,
    numerical_tolerance: float,
    *,
    # Predefined parameters (used internally by ClusteringResult.apply())
    predef_cluster_order: tuple[int, ...] | None = None,
    predef_cluster_centers: tuple[int, ...] | None = None,
    predef_segment_order: tuple[tuple[int, ...], ...] | None = None,
    predef_segment_durations: tuple[tuple[int, ...], ...] | None = None,
    predef_segment_centers: tuple[tuple[int, ...], ...] | None = None,
) -> dict:
    """Build parameters for the old TimeSeriesAggregation API."""
    params: dict = {
        "timeSeries": data,
        "noTypicalPeriods": n_clusters,
        "hoursPerPeriod": period_duration,
        "rescaleClusterPeriods": rescale,
        "numericalTolerance": numerical_tolerance,
    }

    if timestep_duration is not None:
        params["resolution"] = timestep_duration

    if round_decimals is not None:
        params["roundOutput"] = round_decimals

    # Cluster config
    method = METHOD_MAPPING.get(cluster.method)
    if method is None:
        raise ValueError(
            f"Unknown cluster method: {cluster.method!r}. "
            f"Valid options: {list(METHOD_MAPPING.keys())}"
        )
    params["clusterMethod"] = method

    representation = cluster.get_representation()
    rep_mapped = REPRESENTATION_MAPPING.get(representation)
    if rep_mapped is None:
        raise ValueError(
            f"Unknown representation method: {representation!r}. "
            f"Valid options: {list(REPRESENTATION_MAPPING.keys())}"
        )
    params["representationMethod"] = rep_mapped
    params["sortValues"] = cluster.use_duration_curves
    params["sameMean"] = cluster.normalize_means
    params["evalSumPeriods"] = cluster.include_period_sums
    params["solver"] = cluster.solver

    if cluster.weights is not None:
        params["weightDict"] = cluster.weights

    # Predefined cluster parameters (from ClusteringResult)
    if predef_cluster_order is not None:
        params["predefClusterOrder"] = list(predef_cluster_order)

    if predef_cluster_centers is not None:
        params["predefClusterCenterIndices"] = list(predef_cluster_centers)

    # Segmentation config
    if segments is not None:
        params["segmentation"] = True
        params["noSegments"] = segments.n_segments
        params["segmentRepresentationMethod"] = REPRESENTATION_MAPPING.get(
            segments.representation, "meanRepresentation"
        )

        # Predefined segment parameters (from ClusteringResult)
        if predef_segment_order is not None:
            params["predefSegmentOrder"] = [list(s) for s in predef_segment_order]
        if predef_segment_durations is not None:
            params["predefSegmentDurations"] = [
                list(s) for s in predef_segment_durations
            ]
        if predef_segment_centers is not None:
            params["predefSegmentCenters"] = [list(s) for s in predef_segment_centers]
    else:
        params["segmentation"] = False

    # Extreme config
    if extremes is not None and extremes.has_extremes():
        params["extremePeriodMethod"] = EXTREME_METHOD_MAPPING[extremes.method]
        params["addPeakMax"] = extremes.max_value
        params["addPeakMin"] = extremes.min_value
        params["addMeanMax"] = extremes.max_period
        params["addMeanMin"] = extremes.min_period
    else:
        params["extremePeriodMethod"] = "None"

    return params
