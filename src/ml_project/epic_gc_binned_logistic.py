"""Exact fixed-bin GC201 logistic model for EXP-007.

EXP-007 changes one thing relative to EXP-005: it adds a categorical GC201
effect.  The linear GC201 column from EXP-006 is deliberately absent.  The
module is inert on import; counting, fitting, and evaluation happen only after
explicit calls from the experiment notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import warnings
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .epic_baseline import _dense_rank_correlation
from .epic_gc_logistic import (
    GCWindowEvaluation,
    GCWindowLogisticConfig,
    KmerGCCounts,
    _average_precision_from_rows,
    _joint_summary,
    _normalise_contigs,
)
from .epic_kmer_logistic import HierarchicalKmerDesign, hierarchical_kmer_design
from .sequence_composition import gc_fraction


GC_BIN_EDGES: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
GC_BIN_LABELS: tuple[str, ...] = (
    "<0.30",
    "[0.30,0.35)",
    "[0.35,0.40)",
    "[0.40,0.45)",
    "[0.45,0.50)",
    "[0.50,0.55)",
    "[0.55,0.60)",
    ">=0.60",
    "missing",
)
GC_REFERENCE_BIN = 2
GC_MISSING_BIN = 8


@dataclass(frozen=True)
class GCBinnedLogisticConfig(GCWindowLogisticConfig):
    """Every setting capable of changing the EXP-007 candidate."""

    bin_edges: tuple[float, ...] = GC_BIN_EDGES
    reference_bin: int = GC_REFERENCE_BIN

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bin_edges != GC_BIN_EDGES:
            raise ValueError("EXP-007 uses the preregistered fixed GC201 boundaries")
        if self.reference_bin != GC_REFERENCE_BIN:
            raise ValueError("EXP-007 reference must be [0.35,0.40)")

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result["bin_edges"] = list(self.bin_edges)
        result["bin_labels"] = list(GC_BIN_LABELS)
        result["reference_bin"] = self.reference_bin
        result["reference_bin_label"] = GC_BIN_LABELS[self.reference_bin]
        result["gc_missing_policy"] = "separate categorical bin"
        result.pop("gc_scaling", None)
        result["feature_encoding"] = (
            "hierarchical 2/4/6-mer one-hot + eight GC201 contrasts; "
            "reference=[0.35,0.40)"
        )
        result["training_population"] = "all train whitelist positions"
        return result


@dataclass(frozen=True)
class AggregateGCBinnedLikelihood:
    """Exact class rows for hierarchical k-mers plus categorical GC201."""

    matrix: sparse.csr_matrix
    target: np.ndarray
    sample_weight: np.ndarray
    rows: pd.DataFrame


@dataclass(frozen=True)
class GCBinnedLogisticFit:
    """One fitted EXP-007 candidate."""

    estimator: LogisticRegression
    design: HierarchicalKmerDesign
    aggregate: AggregateGCBinnedLikelihood
    coefficient_table: pd.DataFrame
    fit_report: pd.DataFrame


def gc_bin_codes(
    gc_count: np.ndarray | pd.Series,
    valid_count: np.ndarray | pd.Series,
    *,
    edges: tuple[float, ...] = GC_BIN_EDGES,
) -> np.ndarray:
    """Map exact integer sufficient statistics to fixed GC201 categories."""

    fraction = gc_fraction(np.asarray(gc_count), np.asarray(valid_count))
    result = np.searchsorted(np.asarray(edges), fraction, side="right").astype(
        np.int8
    )
    result[~np.isfinite(fraction)] = GC_MISSING_BIN
    return result


def gc_bin_feature_table(
    config: GCBinnedLogisticConfig = GCBinnedLogisticConfig(),
) -> pd.DataFrame:
    """Describe the eight reference-coded GC201 columns."""

    records: list[dict[str, object]] = []
    feature_index = 0
    for code, label in enumerate(GC_BIN_LABELS):
        if code == config.reference_bin:
            continue
        records.append(
            {
                "bin_code": code,
                "bin_label": label,
                "feature_index_within_gc": feature_index,
                "feature_name": f"gc201_bin={label}",
                "feature_group": "GC201 bin",
            }
        )
        feature_index += 1
    return pd.DataFrame(records)


def _gc_bin_matrix(
    codes: np.ndarray,
    config: GCBinnedLogisticConfig,
) -> sparse.csr_matrix:
    codes = np.asarray(codes, dtype=np.int8)
    if ((codes < 0) | (codes >= len(GC_BIN_LABELS))).any():
        raise ValueError("GC bin code outside the fixed vocabulary")
    table = gc_bin_feature_table(config)
    column_for_code = np.full(len(GC_BIN_LABELS), -1, dtype=np.int16)
    column_for_code[table["bin_code"].to_numpy(dtype=np.int8)] = table[
        "feature_index_within_gc"
    ].to_numpy(dtype=np.int16)
    columns = column_for_code[codes]
    active = columns >= 0
    return sparse.csr_matrix(
        (
            np.ones(int(active.sum()), dtype=np.float64),
            (np.flatnonzero(active), columns[active]),
        ),
        shape=(len(codes), len(table)),
    )


def aggregate_binned_gc_binary_likelihood(
    counts: KmerGCCounts,
    design: HierarchicalKmerDesign,
    *,
    config: GCBinnedLogisticConfig = GCBinnedLogisticConfig(),
    contigs: Iterable[str] | None = None,
) -> AggregateGCBinnedLikelihood:
    """Build the exact repeated-row likelihood without the EXP-006 linear GC."""

    if counts.split_name != "train":
        raise ValueError("training aggregate requires train counts")
    summary = _joint_summary(counts, contigs=contigs)
    population = int(summary["total_positions"].sum())
    positive_population = int(summary["positive_positions"].sum())
    negative_population = population - positive_population
    if population <= 0 or positive_population <= 0 or negative_population <= 0:
        raise ValueError("selected population must contain both classes")

    tuple_indices = np.repeat(np.arange(len(summary), dtype=np.int64), 2)
    targets = np.tile(np.array([0, 1], dtype=np.int8), len(summary))
    population_counts = np.column_stack(
        (
            summary["negative_positions"].to_numpy(dtype=np.int64),
            summary["positive_positions"].to_numpy(dtype=np.int64),
        )
    ).reshape(-1)
    keep = population_counts > 0
    tuple_indices = tuple_indices[keep]
    targets = targets[keep]
    population_counts = population_counts[keep]
    if config.balance_classes:
        multipliers = np.array(
            [
                population / (2 * negative_population),
                population / (2 * positive_population),
            ],
            dtype=np.float64,
        )
    else:
        multipliers = np.ones(2, dtype=np.float64)
    weights = population_counts.astype(np.float64) * multipliers[targets]

    rows = summary.iloc[tuple_indices].reset_index(drop=True).assign(
        target=targets,
        population_count=population_counts,
        class_multiplier=multipliers[targets],
        sample_weight=weights,
    )
    codes = gc_bin_codes(rows["gc_count"], rows["valid_count"])
    rows["gc_bin_code"] = codes
    rows["gc_bin"] = pd.Categorical(
        np.asarray(GC_BIN_LABELS, dtype=object)[codes],
        categories=GC_BIN_LABELS,
        ordered=True,
    )
    tuple_matrix = _matrix_for_summary(summary, design, config)
    matrix = tuple_matrix[tuple_indices]
    if int(rows["population_count"].sum()) != population:
        raise AssertionError("aggregate rows do not preserve the population")
    if config.balance_classes:
        sums = rows.groupby("target")["sample_weight"].sum()
        if not np.isclose(sums.loc[0], sums.loc[1]):
            raise AssertionError("balanced class weights must have equal sums")
    return AggregateGCBinnedLikelihood(
        matrix=matrix,
        target=targets,
        sample_weight=weights,
        rows=rows,
    )


def _matrix_for_summary(
    summary: pd.DataFrame,
    design: HierarchicalKmerDesign,
    config: GCBinnedLogisticConfig,
) -> sparse.csr_matrix:
    codes = gc_bin_codes(summary["gc_count"], summary["valid_count"])
    kmer = design.matrix[summary["kmer_6_code"].to_numpy(dtype=np.int32)]
    return sparse.hstack([kmer, _gc_bin_matrix(codes, config)], format="csr")


def _coefficient_table(
    estimator: LogisticRegression,
    design: HierarchicalKmerDesign,
    config: GCBinnedLogisticConfig,
) -> pd.DataFrame:
    kmer = design.feature_table.copy()
    kmer["feature_group"] = kmer["level_k"].map(
        lambda value: f"{int(value)}-mer"
    )
    bins = gc_bin_feature_table(config).copy()
    bins["feature_index"] = len(kmer) + bins["feature_index_within_gc"]
    bins["level_k"] = pd.NA
    bins["category_code"] = bins["bin_code"]
    bins["category"] = bins["bin_label"]
    bins = bins[
        [
            "feature_index",
            "level_k",
            "category_code",
            "category",
            "feature_name",
            "feature_group",
        ]
    ]
    table = pd.concat([kmer, bins], ignore_index=True)
    table["coefficient"] = estimator.coef_[0]
    table["absolute_coefficient"] = np.abs(estimator.coef_[0])
    return table


def _fit_from_aggregate(
    aggregate: AggregateGCBinnedLikelihood,
    design: HierarchicalKmerDesign,
    *,
    C: float,
    config: GCBinnedLogisticConfig,
) -> GCBinnedLogisticFit:
    estimator = LogisticRegression(
        penalty=config.penalty,
        C=float(C),
        solver=config.solver,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.seed,
    )
    started = time.monotonic()
    # Do not call threadpoolctl here.  The Windows environment can load Intel
    # and LLVM OpenMP together; threadpoolctl's runtime introspection then emits
    # a warning and can itself be unsafe.  L-BFGS semantics are unchanged.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(
            aggregate.matrix,
            aggregate.target,
            sample_weight=aggregate.sample_weight,
        )
    elapsed = time.monotonic() - started
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    messages = convergence_messages
    iterations = int(estimator.n_iter_[0])
    coefficients = _coefficient_table(estimator, design, config)
    fit_report = pd.DataFrame(
        [
            {
                "C": float(C),
                "solver": config.solver,
                "penalty": config.penalty,
                "tol": config.tol,
                "iterations": iterations,
                "max_iter": config.max_iter,
                "converged": not convergence_messages and iterations < config.max_iter,
                "fit_seconds": elapsed,
                "aggregate_rows": len(aggregate.rows),
                "features": aggregate.matrix.shape[1],
                "represented_positions": int(
                    aggregate.rows["population_count"].sum()
                ),
                "represented_positives": int(
                    aggregate.rows.loc[
                        aggregate.rows["target"] == 1, "population_count"
                    ].sum()
                ),
                "reference_gc_bin": GC_BIN_LABELS[config.reference_bin],
                "intercept": float(estimator.intercept_[0]),
                "warnings": " | ".join(messages),
            }
        ]
    )
    return GCBinnedLogisticFit(
        estimator=estimator,
        design=design,
        aggregate=aggregate,
        coefficient_table=coefficients,
        fit_report=fit_report,
    )


def fit_binned_gc_logistic(
    counts: KmerGCCounts,
    *,
    C: float,
    config: GCBinnedLogisticConfig = GCBinnedLogisticConfig(),
    contigs: Iterable[str] | None = None,
    design: HierarchicalKmerDesign | None = None,
) -> GCBinnedLogisticFit:
    """Fit one explicit EXP-007 candidate; calling this trains a model."""

    if counts.split_name != "train":
        raise ValueError("fit_binned_gc_logistic requires train counts")
    if C not in config.C_grid:
        raise ValueError("C must belong to the preregistered grid")
    design = design or hierarchical_kmer_design(config)
    aggregate = aggregate_binned_gc_binary_likelihood(
        counts,
        design,
        config=config,
        contigs=contigs,
    )
    return _fit_from_aggregate(aggregate, design, C=C, config=config)


def _evaluate_segment(
    counts: KmerGCCounts,
    fit: GCBinnedLogisticFit,
    config: GCBinnedLogisticConfig,
    *,
    contigs: Iterable[str] | None,
    variant: str,
    include_curve: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = _normalise_contigs(contigs)
    summary = _joint_summary(counts, contigs=selected)
    matrix = _matrix_for_summary(summary, fit.design, config)
    scores = fit.estimator.decision_function(matrix).astype(np.float64)
    total = summary["total_positions"].to_numpy(dtype=np.int64)
    positive = summary["positive_positions"].to_numpy(dtype=np.int64)
    ap, curve = _average_precision_from_rows(
        total, positive, scores, include_curve=include_curve
    )

    keys = ["kmer_6_code", "gc_count", "valid_count"]
    score_table = summary[keys].copy()
    score_table["score"] = scores
    positive_rows = counts.positives
    if selected is not None:
        positive_rows = positive_rows.loc[
            positive_rows["contig"].astype(str).isin(selected)
        ]
    predictions = positive_rows.merge(
        score_table,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if predictions["score"].isna().any():
        raise AssertionError("positive row has no candidate score")
    spearman = _dense_rank_correlation(
        predictions["count"].to_numpy(dtype=np.int64),
        predictions["score"].to_numpy(dtype=np.float64),
    )
    positive_loss = np.logaddexp(0.0, -scores)
    negative_loss = np.logaddexp(0.0, scores)
    population = int(total.sum())
    n_positive = int(positive.sum())
    n_negative = population - n_positive
    population_loss = float(
        (positive @ positive_loss + (total - positive) @ negative_loss) / population
    )
    balanced_loss = float(
        0.5
        * (
            (positive @ positive_loss) / n_positive
            + ((total - positive) @ negative_loss) / n_negative
        )
    )
    segment = "overall" if selected is None else ",".join(selected)
    metric = pd.DataFrame(
        [
            {
                "segment": segment,
                "variant": variant,
                "average_precision": ap,
                "epic_spearman": spearman,
                "population_log_loss": population_loss,
                "balanced_log_loss": balanced_loss,
                "positions": population,
                "positives": n_positive,
                "prevalence": n_positive / population,
            }
        ]
    )
    if not curve.empty:
        curve["segment"] = segment
        curve["variant"] = variant
    predictions = predictions.assign(segment=segment, variant=variant)
    return metric, curve, predictions


def evaluate_binned_gc_logistic(
    counts: KmerGCCounts,
    fit: GCBinnedLogisticFit,
    *,
    config: GCBinnedLogisticConfig = GCBinnedLogisticConfig(),
    contigs: Iterable[str] | None = None,
    variant: str = "EXP-007 hierarchical LR + binned GC201",
    include_curve: bool = True,
    include_per_contig: bool = True,
) -> GCWindowEvaluation:
    """Evaluate exact full-population AP and positive-only EPIC Spearman."""

    overall, curve, predictions = _evaluate_segment(
        counts,
        fit,
        config,
        contigs=contigs,
        variant=variant,
        include_curve=include_curve,
    )
    if not include_per_contig:
        per_contig = pd.DataFrame()
    else:
        selected = _normalise_contigs(contigs)
        if selected is None:
            selected = tuple(sorted(counts.totals["contig"].astype(str).unique()))
        frames = [
            _evaluate_segment(
                counts,
                fit,
                config,
                contigs=[contig],
                variant=variant,
                include_curve=False,
            )[0]
            for contig in selected
        ]
        per_contig = pd.concat(frames, ignore_index=True).rename(
            columns={"segment": "contig"}
        )
    return GCWindowEvaluation(overall, per_contig, curve, predictions)


def cross_validate_binned_gc_logistic(
    train_counts: KmerGCCounts,
    fold_assignment: pd.DataFrame,
    *,
    config: GCBinnedLogisticConfig = GCBinnedLogisticConfig(),
    verbose: bool = False,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> pd.DataFrame:
    """Run the train-contig grid and optionally report every completed fit."""

    if train_counts.split_name != "train":
        raise ValueError("cross-validation requires train counts")
    if not {"contig", "fold"}.issubset(fold_assignment.columns):
        raise ValueError("fold_assignment must contain contig and fold")
    expected = set(train_counts.totals["contig"].astype(str))
    actual = set(fold_assignment["contig"].astype(str))
    if expected != actual or fold_assignment["contig"].astype(str).duplicated().any():
        raise ValueError("fold_assignment must contain every train contig exactly once")

    design = hierarchical_kmer_design(config)
    records: list[dict[str, object]] = []
    folds = sorted(fold_assignment["fold"].astype(int).unique())
    total_runs = len(folds) * len(config.C_grid)
    run_index = 0
    bin_table = gc_bin_feature_table(config)
    for fold in folds:
        validation_contigs = tuple(
            fold_assignment.loc[
                fold_assignment["fold"].astype(int) == fold, "contig"
            ].astype(str)
        )
        fitting_contigs = tuple(sorted(expected - set(validation_contigs)))
        aggregate = aggregate_binned_gc_binary_likelihood(
            train_counts,
            design,
            config=config,
            contigs=fitting_contigs,
        )
        for C in config.C_grid:
            run_index += 1
            fitted = _fit_from_aggregate(
                aggregate, design, C=float(C), config=config
            )
            evaluated = evaluate_binned_gc_logistic(
                train_counts,
                fitted,
                config=config,
                contigs=validation_contigs,
                include_curve=False,
                include_per_contig=False,
            )
            report = fitted.fit_report.iloc[0]
            record: dict[str, object] = {
                "C": float(C),
                "fold": int(fold),
                "validation_contigs": ",".join(validation_contigs),
                **evaluated.metric_summary.iloc[0].to_dict(),
                **report[
                    [
                        "iterations",
                        "converged",
                        "fit_seconds",
                        "aggregate_rows",
                        "warnings",
                    ]
                ].to_dict(),
            }
            gc_coefficients = fitted.coefficient_table.loc[
                fitted.coefficient_table["feature_group"] == "GC201 bin",
                ["category_code", "coefficient"],
            ]
            coefficient_by_code = {
                int(row.category_code): float(row.coefficient)
                for row in gc_coefficients.itertuples(index=False)
            }
            for code in range(len(GC_BIN_LABELS)):
                record[f"gc_bin_coefficient_{code}"] = coefficient_by_code.get(
                    code, 0.0
                )
            records.append(record)
            event = {
                "completed": run_index,
                "total": total_runs,
                "fold": int(fold),
                "C": float(C),
                "fit_seconds": float(report["fit_seconds"]),
                "iterations": int(report["iterations"]),
                "converged": bool(report["converged"]),
                "average_precision": float(
                    evaluated.metric_summary.iloc[0]["average_precision"]
                ),
            }
            if verbose:
                print(
                    f"[{run_index:02d}/{total_runs}] fold={fold} C={float(C):g} "
                    f"AP={event['average_precision']:.9f} "
                    f"iter={event['iterations']} "
                    f"time={event['fit_seconds']:.2f}s "
                    f"converged={event['converged']}",
                    flush=True,
                )
            if progress is not None:
                progress(event)
    return pd.DataFrame(records).sort_values(["C", "fold"]).reset_index(drop=True)


def selected_cv_bin_coefficients(
    cv_results: pd.DataFrame,
    *,
    C: float,
) -> pd.DataFrame:
    """Return a tidy fold-by-bin coefficient table for the selected C."""

    selected = cv_results.loc[np.isclose(cv_results["C"], float(C))]
    records: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        for code, label in enumerate(GC_BIN_LABELS):
            coefficient = float(getattr(row, f"gc_bin_coefficient_{code}"))
            records.append(
                {
                    "fold": int(row.fold),
                    "bin_code": code,
                    "gc_bin": label,
                    "coefficient": coefficient,
                    "odds_multiplier_vs_reference": float(np.exp(coefficient)),
                }
            )
    return pd.DataFrame(records)


def gc_bin_enrichment_table(
    counts: KmerGCCounts,
    *,
    contigs: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute exact target enrichment in the preregistered GC201 bins."""

    summary = _joint_summary(counts, contigs=contigs)
    summary["gc_bin_code"] = gc_bin_codes(
        summary["gc_count"], summary["valid_count"]
    )
    grouped = summary.groupby("gc_bin_code", as_index=False).agg(
        positions=("total_positions", "sum"),
        positives=("positive_positions", "sum"),
    )
    full = pd.DataFrame({"gc_bin_code": np.arange(len(GC_BIN_LABELS))})
    grouped = full.merge(grouped, on="gc_bin_code", how="left").fillna(0)
    grouped[["positions", "positives"]] = grouped[
        ["positions", "positives"]
    ].astype(np.int64)
    grouped["gc_bin"] = np.asarray(GC_BIN_LABELS, dtype=object)[
        grouped["gc_bin_code"].to_numpy(dtype=np.int8)
    ]
    grouped["positive_rate"] = np.divide(
        grouped["positives"],
        grouped["positions"],
        out=np.full(len(grouped), np.nan, dtype=np.float64),
        where=grouped["positions"].to_numpy() > 0,
    )
    prevalence = summary["positive_positions"].sum() / summary[
        "total_positions"
    ].sum()
    grouped["enrichment"] = grouped["positive_rate"] / prevalence
    return grouped[
        [
            "gc_bin_code",
            "gc_bin",
            "positions",
            "positives",
            "positive_rate",
            "enrichment",
        ]
    ]


__all__ = [
    "GC_BIN_EDGES",
    "GC_BIN_LABELS",
    "GC_MISSING_BIN",
    "GC_REFERENCE_BIN",
    "AggregateGCBinnedLikelihood",
    "GCBinnedLogisticConfig",
    "GCBinnedLogisticFit",
    "aggregate_binned_gc_binary_likelihood",
    "cross_validate_binned_gc_logistic",
    "evaluate_binned_gc_logistic",
    "fit_binned_gc_logistic",
    "gc_bin_codes",
    "gc_bin_enrichment_table",
    "gc_bin_feature_table",
    "selected_cv_bin_coefficients",
]
