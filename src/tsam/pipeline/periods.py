"""Period unstacking and feature augmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsam.pipeline.types import PeriodProfiles


def unstack_to_periods(
    normalized_ts: pd.DataFrame,
    n_timesteps_per_period: int,
) -> PeriodProfiles:
    """Reshape the flat time series into a (period x timestep-feature) matrix.

    Clustering groups whole periods, so the flat series must first become a
    matrix where each row is one period and each column is an
    ``(attribute, timestep)`` pair.

    **Example.** 365 days of hourly data for 3 columns is an ``(8760, 3)``
    DataFrame. Unstacking with ``n_timesteps_per_period=24`` yields a
    ``(365, 72)`` matrix — each row is a 72-dimensional point
    (3 columns x 24 hours). Each rows first contains all time steps
    from the first column of respective period, then all time steps from the
    second column, and so on. For the example above that means:

    period_1_ (a1_t1,.....a1_t24, a2_t1,...,a2_t24, a3_t1,...,a3_t24)
    period_2_ (a1_t25,.....a1_t48, a2_t25,...,a2_t48, a3_t25,...,a3_t48)
    ...

    If the series length is not an integer multiple of the period length, the
    last period is padded by repeating the first rows so the reshape succeeds;
    the padded period's weight is corrected later during post-processing.

    Args:
        normalized_ts: Normalized flat time series (output of `normalize`).
        n_timesteps_per_period: Timesteps in one period, e.g. ``24`` for daily
            periods of hourly data.

    Returns:
        The candidate matrix plus the column ``MultiIndex`` and original time
        index needed to reshape and reconstruct later.

    Raises:
        ValueError: If the reshaped data contains NaN (indicates malformed
            input).

    Note:
        `add_period_sum_features` optionally appends per-period column sums as
        extra clustering features.
    """
    # Extend to integer multiple of period length
    padded = normalized_ts
    if len(normalized_ts) % n_timesteps_per_period != 0:
        attached_timesteps = (
            n_timesteps_per_period - len(normalized_ts) % n_timesteps_per_period
        )
        padded = pd.concat([normalized_ts, normalized_ts.head(attached_timesteps)])

    time_index = padded.index.copy(deep=True)
    n_periods = len(padded) // n_timesteps_per_period
    n_columns = len(normalized_ts.columns)

    dtypes = padded.dtypes.unique()
    if len(dtypes) == 1 and isinstance(dtypes[0], np.dtype):
        # Regular grid with one plain dtype: the unstack is a numpy reshape.
        # The index and columns come from pandas unstacking a single period,
        # so the labels are what `unstack` would produce, by construction.
        values = (
            padded.to_numpy()
            .reshape(n_periods, n_timesteps_per_period, n_columns)
            .transpose(0, 2, 1)
            .reshape(n_periods, n_columns * n_timesteps_per_period)
        )
        one_period = _unstack_with_pandas(
            padded.iloc[:n_timesteps_per_period], n_timesteps_per_period
        )
        unstacked = pd.DataFrame(
            values,
            index=pd.Index(np.arange(n_periods), name="PeriodNum"),
            columns=one_period.columns,
        )
    else:
        unstacked = _unstack_with_pandas(padded, n_timesteps_per_period)

    # Check for NaN
    if unstacked.isnull().values.any():
        raise ValueError(
            "Pre processed data includes NaN. Please check the time_series input data."
        )

    return PeriodProfiles(
        column_index=unstacked.columns,  # type: ignore[arg-type]
        time_index=time_index,
        profiles_dataframe=unstacked,
        n_timesteps_per_period=n_timesteps_per_period,
        n_columns=n_columns,
        n_periods=n_periods,
    )


def _unstack_with_pandas(
    padded: pd.DataFrame,
    n_timesteps_per_period: int,
) -> pd.DataFrame:
    """Unstack via pandas, preserving per-column dtypes (mixed-dtype fallback)."""
    unstacked = padded.copy()
    period_index = [ii // n_timesteps_per_period for ii in range(len(unstacked))]
    step_index = [ii % n_timesteps_per_period for ii in range(len(unstacked))]
    unstacked.index = pd.MultiIndex.from_arrays(
        [step_index, period_index], names=["TimeStep", "PeriodNum"]
    )
    return unstacked.unstack(level="TimeStep")  # type: ignore[return-value]


def add_period_sum_features(
    profiles_df: pd.DataFrame,
    candidates: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Append each period's per-column sum as extra clustering features.

    Optional stage, enabled by ``ClusterConfig.include_period_sums``. The
    per-column sum of each period is appended as extra columns so that periods
    with similar totals are pulled together, not just periods with similar
    shapes.

    These extra columns influence **only** which periods get grouped — they are
    stripped from the cluster centers during post-processing (the trim step) so
    they never reach the representation logic, which expects the original
    columns. When per-column weights are active they are already baked into
    ``candidates``, so the sums are appended to the weighted candidates.

    Args:
        profiles_df: The unstacked period profiles (used to compute per-period
            sums).
        candidates: Current candidate matrix (possibly already weighted) to
            augment.

    Returns:
        ``(augmented_candidates, n_extra_features)`` — the second value is the
        number of appended columns, kept so the trim step can remove them.

    Note:
        `cluster_periods` consumes the (possibly augmented) candidate matrix.
    """
    evaluation_values = (
        profiles_df.stack(future_stack=True, level=0).sum(axis=1).unstack(level=1)  # type: ignore[arg-type]
    )
    n_extra = len(evaluation_values.columns)
    augmented = np.concatenate(
        (candidates, evaluation_values.values),
        axis=1,
    )
    return augmented, n_extra
