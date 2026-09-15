"""Reusable dataframe profiling used by Data and EDA notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


FEATURE_GROUP_NAMES = (
    "numeric",
    "count",
    "categorical",
    "ordinal",
    "text",
    "datetime",
    "identifier",
    "ignored",
)


def validate_feature_groups(
    frame: pd.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
    *,
    target: str | None = None,
) -> dict[str, list[str]]:
    """Validate and normalize the explicit feature schema from config.py."""
    unknown_groups = sorted(set(feature_groups) - set(FEATURE_GROUP_NAMES))
    normalized = {
        group: list(dict.fromkeys(feature_groups.get(group, ())))
        for group in FEATURE_GROUP_NAMES
    }

    assignments: dict[str, list[str]] = {}
    for group, columns in normalized.items():
        for column in columns:
            assignments.setdefault(column, []).append(group)

    duplicate_assignments = {
        column: groups
        for column, groups in assignments.items()
        if len(groups) > 1
    }
    missing_columns = sorted(set(assignments) - set(frame.columns))
    expected_columns = set(frame.columns)
    if target is not None:
        expected_columns.discard(target)
    unassigned_columns = sorted(expected_columns - set(assignments))

    non_numeric = [
        column
        for group in ("numeric", "count")
        for column in normalized[group]
        if column in frame.columns
        and not pd.api.types.is_numeric_dtype(frame[column])
    ]

    errors: list[str] = []
    if unknown_groups:
        errors.append(f"неизвестные группы: {', '.join(unknown_groups)}")
    if duplicate_assignments:
        formatted = ", ".join(
            f"{column} ({'/'.join(groups)})"
            for column, groups in sorted(duplicate_assignments.items())
        )
        errors.append(f"признаки назначены несколько раз: {formatted}")
    if missing_columns:
        errors.append(
            "указанные признаки отсутствуют в train: "
            + ", ".join(missing_columns)
        )
    if target is not None and target in assignments:
        errors.append(
            f"target {target!r} не должен входить в FEATURE_GROUPS"
        )
    if unassigned_columns:
        errors.append(
            "не распределены по группам: " + ", ".join(unassigned_columns)
        )
    if non_numeric:
        errors.append(
            "в numeric/count указаны нечисловые столбцы: "
            + ", ".join(sorted(set(non_numeric)))
        )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(
            "Некорректный FEATURE_GROUPS в src/ml_project/config.py:\n"
            f"{details}"
        )

    return normalized


def grouped_target_report(
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    ci_level: float = 95,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate grouped target means and nonparametric bootstrap intervals.

    For a binary ``0/1`` target, the estimate is named ``target_rate``.
    Other numeric targets use ``target_mean``. The returned CI describes
    uncertainty of the group mean, not the spread of individual observations.
    """
    missing = [
        column for column in (feature, target) if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Columns are absent: {missing}")
    if not pd.api.types.is_numeric_dtype(frame[target]):
        raise TypeError("grouped_target_report requires a numeric target.")
    if not 0 < ci_level < 100:
        raise ValueError("ci_level must be between 0 and 100.")
    if n_boot < 1:
        raise ValueError("n_boot must be positive.")

    unique_target = (
        pd.to_numeric(frame[target], errors="coerce")
        .dropna()
        .unique()
        .astype(float)
    )
    is_binary = (
        len(unique_target) > 0
        and len(unique_target) <= 2
        and bool(np.isin(unique_target, [0.0, 1.0]).all())
    )
    estimate_column = "target_rate" if is_binary else "target_mean"
    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile

    rows: list[dict[str, object]] = []
    grouped = frame.groupby(
        feature,
        dropna=False,
        sort=False,
        observed=True,
    )
    for group_number, (group_value, group_frame) in enumerate(grouped):
        values = (
            pd.to_numeric(group_frame[target], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if values.size == 0:
            estimate = np.nan
            ci_low = np.nan
            ci_high = np.nan
        elif values.size == 1:
            estimate = float(values[0])
            ci_low = estimate
            ci_high = estimate
        else:
            estimate = float(values.mean())
            rng = np.random.default_rng(seed + group_number)
            bootstrap_means = np.empty(n_boot, dtype=float)
            # Limit each temporary index matrix to roughly one million values.
            batch_size = max(1, min(n_boot, 1_000_000 // values.size))
            for start in range(0, n_boot, batch_size):
                stop = min(start + batch_size, n_boot)
                indices = rng.integers(
                    0,
                    values.size,
                    size=(stop - start, values.size),
                )
                bootstrap_means[start:stop] = values[indices].mean(axis=1)
            ci_low, ci_high = np.percentile(
                bootstrap_means,
                [lower_percentile, upper_percentile],
            )

        rows.append(
            {
                feature: "<NA>" if pd.isna(group_value) else group_value,
                "rows": len(group_frame),
                estimate_column: estimate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )

    return pd.DataFrame(rows).set_index(feature)


@dataclass
class DatasetProfiler:
    """Profile one dataframe without embedding implementation in a notebook."""

    df: pd.DataFrame
    name: str
    key: str | None = None
    target: str | None = None

    def schema_report(self) -> pd.DataFrame:
        """Return structural metadata only; value quality lives in other reports."""
        return pd.DataFrame(
            {
                "field": self.df.columns,
                "dtype": [str(dtype) for dtype in self.df.dtypes],
            }
        )

    def missing_report(self, *, include_zero: bool = False) -> pd.DataFrame:
        """Return missing counts and shares, already filtered when requested."""
        report = pd.DataFrame(
            {
                "dataset": self.name,
                "field": self.df.columns,
                "missing": self.df.isna().sum().to_numpy(),
                "missing_share": self.df.isna().mean().to_numpy(),
            }
        )
        if not include_zero:
            report = report[report["missing"] > 0]
        return report.sort_values(
            ["missing_share", "field"],
            ascending=[False, True],
        ).reset_index(drop=True)

    def duplicate_report(self) -> pd.DataFrame:
        """Return full-row and configured-key duplicate checks."""
        key_exists = self.key is not None and self.key in self.df.columns
        return pd.DataFrame(
            [
                {
                    "dataset": self.name,
                    "rows": len(self.df),
                    "full_row_duplicates": int(self.df.duplicated().sum()),
                    "key": self.key or "not configured",
                    "key_missing": (
                        int(self.df[self.key].isna().sum()) if key_exists else np.nan
                    ),
                    "key_duplicates": (
                        int(self.df[self.key].duplicated().sum())
                        if key_exists
                        else np.nan
                    ),
                    "key_unique": (
                        bool(self.df[self.key].is_unique) if key_exists else np.nan
                    ),
                }
            ]
        )

    def target_report(self) -> pd.DataFrame:
        """Return counts and shares for the configured target."""
        if self.target is None:
            raise ValueError(f"Target is not configured for dataset {self.name!r}.")
        if self.target not in self.df.columns:
            raise KeyError(
                f"Target {self.target!r} is absent from dataset {self.name!r}."
            )

        counts = self.df[self.target].value_counts(dropna=False).sort_index()
        report = counts.rename("count").to_frame()
        report["share"] = report["count"] / len(self.df)
        report.index = report.index.map(lambda value: "<NA>" if pd.isna(value) else value)
        report.index.name = self.target
        return report.reset_index()

    def numeric_report(
        self,
        *,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return descriptive statistics for numeric columns."""
        selected = (
            list(dict.fromkeys(columns))
            if columns is not None
            else self.df.select_dtypes(include="number").columns.tolist()
        )
        missing = [column for column in selected if column not in self.df.columns]
        if missing:
            raise KeyError(f"Numeric columns are absent: {missing}")
        non_numeric = [
            column
            for column in selected
            if not pd.api.types.is_numeric_dtype(self.df[column])
        ]
        if non_numeric:
            raise TypeError(f"Configured numeric columns are not numeric: {non_numeric}")
        if not selected:
            return pd.DataFrame()
        return self.df[selected].describe().T.reset_index(names="field")

    def categorical_report(
        self,
        *,
        columns: Sequence[str] | None = None,
        max_unique: int = 30,
    ) -> pd.DataFrame:
        """Return compact cardinality and mode information."""
        if columns is None:
            selected = [
                column
                for column in self.df.columns
                if self.df[column].nunique(dropna=True) <= max_unique
                and column != self.target
            ]
        else:
            selected = list(dict.fromkeys(columns))
            missing = [
                column
                for column in selected
                if column not in self.df.columns
            ]
            if missing:
                raise KeyError(f"Categorical columns are absent: {missing}")

        rows: list[dict[str, object]] = []
        for column in selected:
            mode = self.df[column].mode(dropna=True)
            counts = self.df[column].value_counts(dropna=True)
            rows.append(
                {
                    "field": column,
                    "unique": int(self.df[column].nunique(dropna=True)),
                    "most_frequent": mode.iloc[0] if not mode.empty else np.nan,
                    "frequency": int(counts.iloc[0]) if not counts.empty else 0,
                }
            )
        return pd.DataFrame(rows)

    def plot_target(self):
        """Plot the configured target distribution."""
        import seaborn as sns

        if self.target is None or self.target not in self.df.columns:
            raise KeyError(f"Configured target is unavailable in {self.name!r}.")
        axis = sns.countplot(data=self.df, x=self.target)
        axis.set(
            title=f"Target distribution — {self.name}",
            ylabel="Rows",
        )
        return axis

    def plot_numeric_histograms(
        self,
        *,
        columns: Sequence[str] | None = None,
        bins: int = 25,
    ):
        """Plot numeric features, excluding configured key and target."""
        import matplotlib.pyplot as plt

        if columns is None:
            excluded = {value for value in (self.key, self.target) if value}
            selected = [
                column
                for column in self.df.select_dtypes(include="number").columns
                if column not in excluded
            ]
        else:
            selected = list(dict.fromkeys(columns))
            missing = [
                column
                for column in selected
                if column not in self.df.columns
            ]
            if missing:
                raise KeyError(f"Numeric columns are absent: {missing}")
            non_numeric = [
                column
                for column in selected
                if not pd.api.types.is_numeric_dtype(self.df[column])
            ]
            if non_numeric:
                raise TypeError(
                    f"Configured numeric columns are not numeric: {non_numeric}"
                )

        if not selected:
            return None
        axes = self.df[selected].hist(figsize=(12, 8), bins=bins)
        plt.suptitle(f"Numeric distributions — {self.name}", y=1.02)
        plt.tight_layout()
        return axes

    @staticmethod
    def compare_missing(*profiles: "DatasetProfiler") -> pd.DataFrame:
        """Compare missing-value shares across multiple datasets."""
        if not profiles:
            return pd.DataFrame()
        series = {
            profile.name: profile.df.isna().mean()
            for profile in profiles
        }
        report = pd.DataFrame(series).fillna(0)
        report = report[(report > 0).any(axis=1)]
        report.index.name = "field"
        return report.reset_index()
