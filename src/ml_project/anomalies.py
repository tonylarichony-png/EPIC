"""Transparent statistical outlier overview for EDA notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NumericAnomalyReport:
    """Read-only IQR/MAD overview calculated from one reference dataset."""

    reference_dataset: str
    iqr_multiplier: float
    robust_z_threshold: float
    feature_bounds: pd.DataFrame
    dataset_summary: pd.DataFrame
    flagged_values: pd.DataFrame
    row_summary: pd.DataFrame


_BOUND_COLUMNS = [
    "feature",
    "reference_non_missing",
    "reference_min",
    "q1",
    "median",
    "q3",
    "reference_max",
    "iqr",
    "lower_bound",
    "upper_bound",
    "mad",
    "iqr_available",
    "mad_available",
]

_DATASET_SUMMARY_COLUMNS = [
    "dataset",
    "feature",
    "non_missing",
    "iqr_outliers",
    "mad_outliers",
    "any_outliers",
    "outlier_share",
    "outside_reference_range",
]

_FLAGGED_COLUMNS = [
    "dataset",
    "row_position",
    "row_index",
    "feature",
    "value",
    "iqr_outlier",
    "mad_outlier",
    "iqr_distance",
    "robust_z",
    "severity",
]

_ROW_SUMMARY_COLUMNS = [
    "dataset",
    "row_position",
    "row_index",
    "flagged_features",
    "feature_count",
    "max_severity",
]


def _validate_inputs(
    datasets: Mapping[str, pd.DataFrame],
    reference_dataset: str,
    columns: Sequence[str],
) -> tuple[str, ...]:
    if reference_dataset not in datasets:
        raise KeyError(f"Reference dataset {reference_dataset!r} is unavailable")
    if not datasets:
        raise ValueError("At least one dataset is required")

    selected = tuple(dict.fromkeys(map(str, columns)))
    for dataset_name, frame in datasets.items():
        missing = [column for column in selected if column not in frame.columns]
        if missing:
            raise KeyError(
                f"Numeric columns are absent from {dataset_name!r}: {missing}"
            )
        non_numeric = [
            column
            for column in selected
            if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if non_numeric:
            raise TypeError(
                f"Configured numeric columns are not numeric in "
                f"{dataset_name!r}: {non_numeric}"
            )
    return selected


def _feature_bounds(
    frame: pd.DataFrame,
    columns: Sequence[str],
    iqr_multiplier: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in columns:
        series = pd.to_numeric(frame[feature], errors="coerce")
        finite = series[np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))]
        if finite.empty:
            q1 = median = q3 = minimum = maximum = mad = np.nan
        else:
            q1 = float(finite.quantile(0.25))
            median = float(finite.median())
            q3 = float(finite.quantile(0.75))
            minimum = float(finite.min())
            maximum = float(finite.max())
            mad = float((finite - median).abs().median())
        iqr = q3 - q1 if np.isfinite(q1) and np.isfinite(q3) else np.nan
        iqr_available = bool(np.isfinite(iqr) and iqr > 0)
        mad_available = bool(np.isfinite(mad) and mad > 0)
        rows.append(
            {
                "feature": feature,
                "reference_non_missing": int(series.notna().sum()),
                "reference_min": minimum,
                "q1": q1,
                "median": median,
                "q3": q3,
                "reference_max": maximum,
                "iqr": iqr,
                "lower_bound": q1 - iqr_multiplier * iqr if iqr_available else np.nan,
                "upper_bound": q3 + iqr_multiplier * iqr if iqr_available else np.nan,
                "mad": mad,
                "iqr_available": iqr_available,
                "mad_available": mad_available,
            }
        )
    return pd.DataFrame(rows, columns=_BOUND_COLUMNS)


def _score_feature(
    series: pd.Series,
    bounds: pd.Series,
    robust_z_threshold: float,
) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    valid = values.notna()

    iqr_outlier = pd.Series(False, index=values.index)
    iqr_distance = pd.Series(np.nan, index=values.index, dtype=float)
    if bool(bounds["iqr_available"]):
        lower = float(bounds["lower_bound"])
        upper = float(bounds["upper_bound"])
        iqr = float(bounds["iqr"])
        iqr_outlier = valid & ((values < lower) | (values > upper))
        iqr_distance = pd.Series(0.0, index=values.index)
        iqr_distance.loc[values < lower] = (lower - values.loc[values < lower]) / iqr
        iqr_distance.loc[values > upper] = (values.loc[values > upper] - upper) / iqr
        iqr_distance.loc[~valid] = np.nan

    robust_z = pd.Series(np.nan, index=values.index, dtype=float)
    mad_outlier = pd.Series(False, index=values.index)
    if bool(bounds["mad_available"]):
        robust_z = 0.6744897501960817 * (
            values - float(bounds["median"])
        ) / float(bounds["mad"])
        mad_outlier = valid & robust_z.abs().gt(robust_z_threshold)

    normalized_mad = robust_z.abs() / robust_z_threshold
    severity = pd.concat([iqr_distance, normalized_mad], axis=1).max(
        axis=1,
        skipna=True,
    )
    severity.loc[~(iqr_outlier | mad_outlier)] = 0.0
    return pd.DataFrame(
        {
            "value": values,
            "iqr_outlier": iqr_outlier,
            "mad_outlier": mad_outlier,
            "iqr_distance": iqr_distance,
            "robust_z": robust_z,
            "severity": severity,
        }
    )


def analyze_numeric_anomalies(
    datasets: Mapping[str, pd.DataFrame],
    *,
    reference_dataset: str,
    columns: Sequence[str],
    iqr_multiplier: float = 1.5,
    robust_z_threshold: float = 3.5,
) -> NumericAnomalyReport:
    """Flag numeric outlier candidates without modifying any input frame.

    IQR bounds, median and MAD are fitted only on ``reference_dataset`` and
    then reused for every comparison dataset.
    """

    if iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be positive")
    if robust_z_threshold <= 0:
        raise ValueError("robust_z_threshold must be positive")
    selected = _validate_inputs(datasets, reference_dataset, columns)
    bounds = _feature_bounds(
        datasets[reference_dataset],
        selected,
        iqr_multiplier,
    )
    bounds_by_feature = bounds.set_index("feature") if selected else bounds

    summary_rows: list[dict[str, Any]] = []
    flagged_rows: list[dict[str, Any]] = []
    for dataset_name, frame in datasets.items():
        for feature in selected:
            feature_bounds = bounds_by_feature.loc[feature]
            scores = _score_feature(
                frame[feature].reset_index(drop=True),
                feature_bounds,
                robust_z_threshold,
            )
            any_outlier = scores["iqr_outlier"] | scores["mad_outlier"]
            values = scores["value"]
            valid = values.notna()
            reference_min = float(feature_bounds["reference_min"])
            reference_max = float(feature_bounds["reference_max"])
            outside_reference = (
                valid
                & (
                    values.lt(reference_min)
                    | values.gt(reference_max)
                )
            ) if np.isfinite(reference_min) and np.isfinite(reference_max) else pd.Series(
                False, index=values.index
            )
            non_missing = int(valid.sum())
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "feature": feature,
                    "non_missing": non_missing,
                    "iqr_outliers": int(scores["iqr_outlier"].sum()),
                    "mad_outliers": int(scores["mad_outlier"].sum()),
                    "any_outliers": int(any_outlier.sum()),
                    "outlier_share": (
                        float(any_outlier.sum() / non_missing)
                        if non_missing
                        else 0.0
                    ),
                    "outside_reference_range": int(outside_reference.sum()),
                }
            )
            for position in np.flatnonzero(any_outlier.to_numpy()):
                score = scores.iloc[position]
                flagged_rows.append(
                    {
                        "dataset": dataset_name,
                        "row_position": int(position),
                        "row_index": frame.index[position],
                        "feature": feature,
                        "value": float(score["value"]),
                        "iqr_outlier": bool(score["iqr_outlier"]),
                        "mad_outlier": bool(score["mad_outlier"]),
                        "iqr_distance": float(score["iqr_distance"]),
                        "robust_z": float(score["robust_z"]),
                        "severity": float(score["severity"]),
                    }
                )

    dataset_summary = pd.DataFrame(
        summary_rows,
        columns=_DATASET_SUMMARY_COLUMNS,
    )
    flagged = pd.DataFrame(flagged_rows, columns=_FLAGGED_COLUMNS)
    if flagged.empty:
        row_summary = pd.DataFrame(columns=_ROW_SUMMARY_COLUMNS)
    else:
        row_summary = (
            flagged.groupby(
                ["dataset", "row_position", "row_index"],
                as_index=False,
                dropna=False,
                sort=False,
            )
            .agg(
                flagged_features=(
                    "feature",
                    lambda values: ", ".join(dict.fromkeys(map(str, values))),
                ),
                feature_count=("feature", "nunique"),
                max_severity=("severity", "max"),
            )
            .sort_values(
                ["feature_count", "max_severity"],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )

    return NumericAnomalyReport(
        reference_dataset=reference_dataset,
        iqr_multiplier=float(iqr_multiplier),
        robust_z_threshold=float(robust_z_threshold),
        feature_bounds=bounds,
        dataset_summary=dataset_summary,
        flagged_values=flagged.sort_values(
            ["severity", "dataset", "feature"],
            ascending=[False, True, True],
        ).reset_index(drop=True),
        row_summary=row_summary,
    )


def plot_anomaly_summary(
    report: NumericAnomalyReport,
    *,
    ax: Any | None = None,
) -> Any:
    """Plot the share of flagged values by feature and dataset."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    if report.dataset_summary.empty:
        ax.text(0.5, 0.5, "Нет числовых признаков", ha="center", va="center")
        ax.axis("off")
        return ax
    pivot = report.dataset_summary.pivot(
        index="feature",
        columns="dataset",
        values="outlier_share",
    )
    pivot.plot.bar(ax=ax)
    ax.set_title("Доля значений, отмеченных IQR или MAD")
    ax.set_xlabel("")
    ax.set_ylabel("Доля")
    ax.tick_params(axis="x", rotation=0)
    return ax


def plot_feature_anomalies(
    frame: pd.DataFrame,
    report: NumericAnomalyReport,
    *,
    dataset: str,
    feature: str,
    ax: Any | None = None,
) -> Any:
    """Show one numeric distribution and highlight its flagged observations."""

    import matplotlib.pyplot as plt

    if feature not in frame:
        raise KeyError(f"Feature {feature!r} is absent from the frame")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.8))
    values = pd.to_numeric(frame[feature], errors="coerce")
    finite = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
    if finite.empty:
        ax.text(0.5, 0.5, "Нет конечных значений", ha="center", va="center")
        ax.axis("off")
        return ax

    ax.boxplot(finite.to_numpy(), vert=False, widths=0.45)
    flagged = report.flagged_values[
        report.flagged_values["dataset"].eq(dataset)
        & report.flagged_values["feature"].eq(feature)
    ]
    if not flagged.empty:
        ax.scatter(
            flagged["value"],
            np.ones(len(flagged)),
            color="#d62728",
            marker="x",
            s=45,
            label="IQR/MAD candidate",
            zorder=3,
        )
        ax.legend(loc="upper right")
    feature_bounds = report.feature_bounds.set_index("feature").loc[feature]
    if bool(feature_bounds["iqr_available"]):
        ax.axvline(
            feature_bounds["lower_bound"],
            color="#ff7f0e",
            linestyle="--",
            linewidth=1,
        )
        ax.axvline(
            feature_bounds["upper_bound"],
            color="#ff7f0e",
            linestyle="--",
            linewidth=1,
        )
    ax.set_title(f"{dataset}: {feature}")
    ax.set_yticks([])
    ax.set_xlabel(feature)
    return ax


__all__ = [
    "NumericAnomalyReport",
    "analyze_numeric_anomalies",
    "plot_anomaly_summary",
    "plot_feature_anomalies",
]
