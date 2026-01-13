"""Plotting utilities for tsam using Plotly Express.

This module provides interactive visualizations for time series aggregation results.
Uses Plotly Express for clean, declarative plotting with automatic faceting and colors.

Two usage patterns are supported:

1. Module-level functions for exploring raw data:
   >>> import tsam
   >>> tsam.plot.heatmap(df, column="Load")
   >>> tsam.plot.duration_curve(df)
   >>> tsam.plot.time_slice(df, start="2010-02-01", end="2010-02-07")
   >>> tsam.plot.compare({"Method1": df1, "Method2": df2}, column="Load")

2. Accessor pattern on results for validation and visualization:
   >>> result = tsam.aggregate(df, n_clusters=8)
   >>> result.plot.compare()  # Compare original vs reconstructed
   >>> result.plot.residuals()  # View reconstruction errors
   >>> result.plot.cluster_representatives()
   >>> result.plot.cluster_weights()
   >>> result.plot.accuracy()

Note: This module requires the 'plotly' optional dependency.
Install with: pip install tsam[plot]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError(
        "The tsam.plot module requires plotly. Install it with: pip install tsam[plot]"
    ) from e

if TYPE_CHECKING:
    from tsam.result import AggregationResult


def heatmap(
    data: pd.DataFrame,
    column: str | None = None,
    period_duration: int | float | str = 24,
    title: str | None = None,
    color_continuous_scale: str = "Viridis",
) -> go.Figure:
    """Create a heatmap of time series data organized by periods.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data to plot.
    column : str, optional
        Column to plot. If None, uses the first column.
    period_duration : int, float, or str, default 24
        Length of each period. Accepts:
        - int/float: hours (e.g., 24 for daily, 168 for weekly)
        - str: pandas Timedelta string (e.g., '24h', '1d', '1w')
    title : str, optional
        Plot title.
    color_continuous_scale : str, default "Viridis"
        Plotly color scale name.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import tsam
    >>> tsam.plot.heatmap(df, column="Temperature", period_duration=24)
    """
    from tsam.api import _parse_duration_hours
    from tsam.timeseriesaggregation import unstackToPeriods

    if column is None:
        column = data.columns[0]

    period_duration = int(_parse_duration_hours(period_duration, "period_duration"))
    stacked, _ = unstackToPeriods(data[[column]].copy(), period_duration)

    fig = px.imshow(
        stacked[column].values.T,
        labels={"x": "Period (Day)", "y": "Timestep (Hour)", "color": column},
        title=title or f"{column} Heatmap",
        color_continuous_scale=color_continuous_scale,
        aspect="auto",
    )

    return fig


def duration_curve(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Duration Curve",
) -> go.Figure:
    """Plot duration curves (sorted descending values).

    Parameters
    ----------
    data : pd.DataFrame
        Time series data to plot.
    columns : list[str], optional
        Columns to plot. If None, plots all.
    title : str, default "Duration Curve"
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import tsam
    >>> tsam.plot.duration_curve(df, columns=["Load", "GHI"])
    """
    if columns is None:
        columns = list(data.columns)

    # Build long-form data with sorted values using vectorized operations
    frames = []
    for col in columns:
        sorted_vals = data[col].sort_values(ascending=False).reset_index(drop=True)
        df_col = pd.DataFrame(
            {
                "Hour": range(len(sorted_vals)),
                "Value": sorted_vals.values,
                "Column": col,
            }
        )
        frames.append(df_col)
    long_df = pd.concat(frames, ignore_index=True)

    fig = px.line(
        long_df,
        x="Hour",
        y="Value",
        color="Column",
        title=title,
    )

    return fig


def time_slice(
    data: pd.DataFrame,
    start: str,
    end: str,
    columns: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot a time slice of the data.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index.
    start : str
        Start date/time string.
    end : str
        End date/time string.
    columns : list[str], optional
        Columns to plot. If None, plots all.
    title : str, optional
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import tsam
    >>> tsam.plot.time_slice(df, start="20100210", end="20100218", columns=["Load"])
    """
    sliced = data.loc[start:end]  # type: ignore[misc]

    if columns is None:
        columns = list(sliced.columns)

    sliced_subset = sliced[columns].copy()
    sliced_subset = sliced_subset.reset_index()
    sliced_subset.columns = pd.Index(["Time", *columns])

    long_df = sliced_subset.melt(
        id_vars=["Time"], var_name="Column", value_name="Value"
    )

    fig = px.line(
        long_df,
        x="Time",
        y="Value",
        color="Column",
        title=title or f"Time Series: {start} to {end}",
    )

    return fig


def compare(
    results: dict[str, pd.DataFrame],
    column: str,
    plot_type: str = "duration_curve",
    start: str | None = None,
    end: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Compare multiple DataFrames (e.g., from different aggregation methods).

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Dictionary mapping names to DataFrames.
        Example: {"Original": raw, "K-means": result1.reconstruct()}
    column : str
        Column to compare.
    plot_type : str, default "duration_curve"
        Type of plot: "duration_curve" or "time_slice".
    start : str, optional
        Start time (required for time_slice).
    end : str, optional
        End time (required for time_slice).
    title : str, optional
        Plot title.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> import tsam
    >>> result1 = tsam.aggregate(df, n_clusters=8, cluster=ClusterConfig(method="kmeans"))
    >>> result2 = tsam.aggregate(df, n_clusters=8, cluster=ClusterConfig(method="hierarchical"))
    >>> fig = tsam.plot.compare(
    ...     {"Original": df, "K-means": result1.reconstruct(), "Hierarchical": result2.reconstruct()},
    ...     column="Load",
    ...     plot_type="duration_curve"
    ... )
    """
    if plot_type == "duration_curve":
        frames = []
        for name, data in results.items():
            sorted_vals = (
                data[column].sort_values(ascending=False).reset_index(drop=True)
            )
            frames.append(
                pd.DataFrame(
                    {
                        "Hour": range(len(sorted_vals)),
                        "Value": sorted_vals.values,
                        "Method": name,
                    }
                )
            )
        long_df = pd.concat(frames, ignore_index=True)
        fig = px.line(
            long_df,
            x="Hour",
            y="Value",
            color="Method",
            line_dash="Method",
            title=title or f"Duration Curve Comparison - {column}",
        )

    elif plot_type == "time_slice":
        if start is None or end is None:
            raise ValueError("start and end are required for time_slice plot")

        frames = []
        for name, data in results.items():
            sliced = data.loc[start:end]  # type: ignore[misc]
            frames.append(
                pd.DataFrame(
                    {
                        "Time": sliced.index,
                        "Value": sliced[column].values,
                        "Method": name,
                    }
                )
            )
        long_df = pd.concat(frames, ignore_index=True)
        fig = px.line(
            long_df,
            x="Time",
            y="Value",
            color="Method",
            line_dash="Method",
            title=title or f"Time Slice Comparison - {column}",
        )

    else:
        raise ValueError(
            f"Unknown plot_type: {plot_type}. Use 'duration_curve' or 'time_slice'."
        )

    return fig


class ResultPlotAccessor:
    """Plotting accessor for AggregationResult.

    Provides convenient plotting methods directly on the result object.

    Examples
    --------
    >>> result = tsam.aggregate(df, n_clusters=8)
    >>> result.plot.compare()  # Compare original vs reconstructed
    >>> result.plot.residuals()  # View reconstruction errors
    >>> result.plot.cluster_representatives()
    >>> result.plot.cluster_weights()
    """

    def __init__(self, result: AggregationResult):
        self._result = result

    def cluster_representatives(
        self,
        columns: list[str] | None = None,
        title: str = "Cluster Representatives",
    ) -> go.Figure:
        """Plot all cluster representatives (typical periods).

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot.
        title : str, default "Cluster Representatives"
            Plot title.

        Returns
        -------
        go.Figure
        """
        typ = self._result.cluster_representatives
        weights = self._result.cluster_weights

        all_columns = [c for c in typ.columns if c not in ["cluster", "timestep"]]
        if columns is None:
            columns = all_columns
        else:
            columns = [c for c in columns if c in all_columns]

        # Reset index to get period/timestep as columns
        df = typ[columns].reset_index()
        df.columns = pd.Index(["Period", "Timestep", *columns])

        # Map period IDs to labels with weights
        df["Period"] = df["Period"].map(lambda p: f"Period {p} (n={weights.get(p, 1)})")

        long_df = df.melt(
            id_vars=["Period", "Timestep"],
            var_name="Column",
            value_name="Value",
        )

        fig = px.line(
            long_df,
            x="Timestep",
            y="Value",
            color="Period",
            facet_col="Column" if len(columns) > 1 else None,
            title=title,
        )

        return fig

    def cluster_weights(self, title: str = "Cluster Weights") -> go.Figure:
        """Plot cluster weight distribution.

        Parameters
        ----------
        title : str, default "Cluster Weights"
            Plot title.

        Returns
        -------
        go.Figure
        """
        weights = self._result.cluster_weights
        df = pd.DataFrame(
            {
                "Period": [f"Period {p}" for p in weights],
                "Count": list(weights.values()),
            }
        )

        fig = px.bar(
            df,
            x="Period",
            y="Count",
            title=title,
            text="Count",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig.update_traces(textposition="auto")
        fig.update_layout(showlegend=False)

        return fig

    def accuracy(self, title: str = "Accuracy Metrics") -> go.Figure:
        """Plot accuracy metrics by column.

        Parameters
        ----------
        title : str, default "Accuracy Metrics"
            Plot title.

        Returns
        -------
        go.Figure
        """
        acc = self._result.accuracy
        columns = list(acc.rmse.index)

        records = []
        for col in columns:
            records.append({"Column": col, "Metric": "RMSE", "Value": acc.rmse[col]})
            records.append({"Column": col, "Metric": "MAE", "Value": acc.mae[col]})
            records.append(
                {
                    "Column": col,
                    "Metric": "RMSE (Duration)",
                    "Value": acc.rmse_duration[col],
                }
            )

        df = pd.DataFrame(records)

        fig = px.bar(
            df,
            x="Column",
            y="Value",
            color="Metric",
            barmode="group",
            title=title,
        )

        return fig

    def segment_durations(self, title: str = "Segment Durations") -> go.Figure:
        """Plot segment durations (if segmentation was used).

        Parameters
        ----------
        title : str, default "Segment Durations"
            Plot title.

        Returns
        -------
        go.Figure

        Raises
        ------
        ValueError
            If no segmentation was used.
        """
        if self._result.segment_durations is None:
            raise ValueError("No segmentation was used in this aggregation")

        # segment_durations is tuple[tuple[int, ...], ...] - one tuple per period
        # Average durations across all typical periods for the bar chart
        durations = self._result.segment_durations
        n_segments = len(durations[0])
        avg_durations = [
            sum(period[s] for period in durations) / len(durations)
            for s in range(n_segments)
        ]

        df = pd.DataFrame(
            {
                "Segment": [f"Segment {s}" for s in range(n_segments)],
                "Duration": avg_durations,
            }
        )

        fig = px.bar(
            df,
            x="Segment",
            y="Duration",
            title=title,
            text="Duration",
            color="Duration",
            color_continuous_scale="Viridis",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="auto")
        fig.update_layout(showlegend=False, yaxis_title="Duration (timesteps)")

        return fig

    def compare(
        self,
        columns: list[str] | None = None,
        mode: str = "overlay",
        start: str | None = None,
        end: str | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Compare original vs reconstructed time series.

        Parameters
        ----------
        columns : list[str], optional
            Columns to compare. If None, uses first column.
        mode : str, default "overlay"
            Comparison mode:
            - "overlay": Both series on same axes
            - "side_by_side": Separate subplots
            - "duration_curve": Compare sorted values
        start : str, optional
            Start time for time slice (overlay/side_by_side modes).
        end : str, optional
            End time for time slice.
        title : str, optional
            Plot title.

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> result.plot.compare()  # Quick overlay of first column
        >>> result.plot.compare(mode="duration_curve")
        >>> result.plot.compare(start="2010-02-01", end="2010-02-07")
        """
        orig = self._result.original
        recon = self._result.reconstructed

        if columns is None:
            columns = [orig.columns[0]]

        if start is not None and end is not None:
            orig = orig.loc[start:end]  # type: ignore[misc]
            recon = recon.loc[start:end]  # type: ignore[misc]

        if mode == "duration_curve":
            return compare(
                {"Original": orig, "Reconstructed": recon},
                column=columns[0],
                plot_type="duration_curve",
                title=title or f"Duration Curve Comparison - {columns[0]}",
            )

        elif mode in ("overlay", "side_by_side"):
            # Build long-form data with Source (Original/Reconstructed) and Column
            orig_df = orig[columns].copy()
            orig_df["Source"] = "Original"
            recon_df = recon[columns].copy()
            recon_df["Source"] = "Reconstructed"

            combined = pd.concat([orig_df, recon_df])
            combined.index.name = "Time"
            long_df = combined.reset_index().melt(
                id_vars=["Time", "Source"],
                var_name="Column",
                value_name="Value",
            )

            if mode == "overlay":
                # Color by Column, dash by Source (Original/Reconstructed)
                fig = px.line(
                    long_df,
                    x="Time",
                    y="Value",
                    color="Column",
                    line_dash="Source",
                    facet_row="Column" if len(columns) > 1 else None,
                    title=title or "Original vs Reconstructed",
                )
            else:  # side_by_side
                fig = px.line(
                    long_df,
                    x="Time",
                    y="Value",
                    color="Column",
                    facet_row="Source",
                    title=title or "Original vs Reconstructed",
                )
                fig.update_layout(height=600)

            return fig

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'overlay', 'side_by_side', or 'duration_curve'."
            )

    def residuals(
        self,
        columns: list[str] | None = None,
        mode: str = "time_series",
        start: str | None = None,
        end: str | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Plot residuals (original - reconstructed).

        Parameters
        ----------
        columns : list[str], optional
            Columns to plot. If None, plots all.
        mode : str, default "time_series"
            Display mode:
            - "time_series": Residuals over time
            - "histogram": Distribution of residuals
            - "by_period": Mean absolute error per period (bar chart)
            - "by_timestep": Mean absolute error by timestep within period
        start : str, optional
            Start time for time slice (time_series mode only).
        end : str, optional
            End time for time slice.
        title : str, optional
            Plot title.

        Returns
        -------
        go.Figure

        Examples
        --------
        >>> result.plot.residuals()  # Time series of residuals
        >>> result.plot.residuals(mode="histogram")  # Error distribution
        >>> result.plot.residuals(mode="by_period")  # Which periods have highest error
        >>> result.plot.residuals(mode="by_timestep")  # Error pattern within day
        """
        resid = self._result.residuals
        if columns is None:
            columns = list(resid.columns)

        if start is not None and end is not None:
            resid = resid.loc[start:end]  # type: ignore[misc]

        if mode == "time_series":
            df_plot = resid[columns].copy()
            df_plot.index.name = "Time"
            long_df = df_plot.reset_index().melt(
                id_vars=["Time"],
                var_name="Column",
                value_name="Residual",
            )
            fig = px.line(
                long_df,
                x="Time",
                y="Residual",
                color="Column",
                title=title or "Residuals Over Time",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            return fig

        elif mode == "histogram":
            long_df = resid[columns].melt(var_name="Column", value_name="Residual")
            fig = px.histogram(
                long_df,
                x="Residual",
                color="Column",
                barmode="overlay",
                opacity=0.7,
                title=title or "Residual Distribution",
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            return fig

        elif mode == "by_period":
            n_timesteps = self._result.n_timesteps_per_period
            full_resid = self._result.residuals[columns].abs().copy()
            full_resid["Period"] = np.arange(len(full_resid)) // n_timesteps

            df = full_resid.groupby("Period")[columns].mean().reset_index()
            long_df = df.melt(id_vars="Period", var_name="Column", value_name="MAE")

            fig = px.bar(
                long_df,
                x="Period",
                y="MAE",
                color="Column",
                barmode="group",
                title=title or "Mean Absolute Error by Period",
            )
            return fig

        elif mode == "by_timestep":
            n_timesteps = self._result.n_timesteps_per_period
            full_resid = self._result.residuals[columns].abs().copy()
            full_resid["Timestep"] = np.arange(len(full_resid)) % n_timesteps

            df = full_resid.groupby("Timestep")[columns].mean().reset_index()
            long_df = df.melt(id_vars="Timestep", var_name="Column", value_name="MAE")

            fig = px.line(
                long_df,
                x="Timestep",
                y="MAE",
                color="Column",
                title=title or "Mean Absolute Error by Timestep",
            )
            return fig

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'time_series', 'histogram', 'by_period', or 'by_timestep'."
            )
