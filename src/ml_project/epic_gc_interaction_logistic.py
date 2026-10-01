"""Exact dinucleotide-by-GC-bin interaction model for EXP-008.

The candidate keeps every EXP-007 main effect and adds only reference-coded
``k2[-1,0] × GC201-bin`` interactions.  Importing this module never trains a
model or reads project data.
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
from .epic_gc_binned_logistic import (
    GC_BIN_LABELS,
    GCBinnedLogisticConfig,
    KmerGCCounts,
    _matrix_for_summary as _binned_matrix_for_summary,
    aggregate_binned_gc_binary_likelihood,
    gc_bin_codes,
    gc_bin_feature_table,
)
from .epic_gc_logistic import (
    GCWindowEvaluation,
    _average_precision_from_rows,
    _joint_summary,
    _normalise_contigs,
)
from .epic_kmer import kmer_category_labels
from .epic_kmer_logistic import HierarchicalKmerDesign, hierarchical_kmer_design


INTERACTION_C_GRID: tuple[float, ...] = (1e-4, 2e-4, 3e-4, 5e-4, 1e-3)
K2_REFERENCE_LABEL = "TT"
K2_LABELS = kmer_category_labels(2)
K2_REFERENCE_CODE = K2_LABELS.index(K2_REFERENCE_LABEL)


@dataclass(frozen=True)
class GCInteractionLogisticConfig(GCBinnedLogisticConfig):
    """Every setting capable of changing the EXP-008 candidate."""

    C_grid: tuple[float, ...] = INTERACTION_C_GRID
    k2_reference_code: int = K2_REFERENCE_CODE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.C_grid != INTERACTION_C_GRID:
            raise ValueError("EXP-008 uses five preregistered C values (15 CV fits)")
        if self.k2_reference_code != K2_REFERENCE_CODE:
            raise ValueError("EXP-008 interaction reference dinucleotide is TT")

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result["C_grid"] = list(self.C_grid)
        result["k2_reference_code"] = self.k2_reference_code
        result["k2_reference_label"] = K2_REFERENCE_LABEL
        result["interaction_columns"] = 128
        result["feature_encoding"] = (
            "EXP-007 main effects + 128 reference-coded "
            "k2[-1,0] x GC201-bin interactions"
        )
        result["thread_control"] = (
            "notebook sets MKL_THREADING_LAYER=SEQUENTIAL before numeric imports; "
            "threadpoolctl is not called during fit"
        )
        return result


@dataclass(frozen=True)
class AggregateGCInteractionLikelihood:
    matrix: sparse.csr_matrix
    target: np.ndarray
    sample_weight: np.ndarray
    rows: pd.DataFrame


@dataclass(frozen=True)
class GCInteractionLogisticFit:
    estimator: LogisticRegression
    design: HierarchicalKmerDesign
    aggregate: AggregateGCInteractionLikelihood
    coefficient_table: pd.DataFrame
    fit_report: pd.DataFrame


def interaction_feature_table(
    config: GCInteractionLogisticConfig = GCInteractionLogisticConfig(),
) -> pd.DataFrame:
    """Return the fixed 16-by-8 reference-coded interaction vocabulary."""

    gc_table = gc_bin_feature_table(config)
    records: list[dict[str, object]] = []
    feature_index = 0
    for k2_code, k2_label in enumerate(K2_LABELS):
        if k2_code == config.k2_reference_code:
            continue
        for gc_row in gc_table.itertuples(index=False):
            records.append(
                {
                    "interaction_index": feature_index,
                    "k2_code": k2_code,
                    "k2": k2_label,
                    "gc_bin_code": int(gc_row.bin_code),
                    "gc_bin": str(gc_row.bin_label),
                    "feature_name": f"k2={k2_label} x gc201_bin={gc_row.bin_label}",
                    "feature_group": "k2 x GC201 bin",
                }
            )
            feature_index += 1
    return pd.DataFrame(records)


def _k2_codes(
    kmer_6_codes: np.ndarray | pd.Series,
    design: HierarchicalKmerDesign,
) -> np.ndarray:
    mapping = (
        design.category_table.sort_values("max_category_code")["kmer_2_code"]
        .to_numpy(dtype=np.int16)
    )
    codes = np.asarray(kmer_6_codes, dtype=np.int32)
    return mapping[codes]


def _interaction_matrix(
    k2_codes: np.ndarray,
    gc_codes: np.ndarray,
    config: GCInteractionLogisticConfig,
) -> sparse.csr_matrix:
    table = interaction_feature_table(config)
    lookup = np.full((len(K2_LABELS), len(GC_BIN_LABELS)), -1, dtype=np.int16)
    lookup[
        table["k2_code"].to_numpy(dtype=np.int16),
        table["gc_bin_code"].to_numpy(dtype=np.int16),
    ] = table["interaction_index"].to_numpy(dtype=np.int16)
    columns = lookup[
        np.asarray(k2_codes, dtype=np.int16),
        np.asarray(gc_codes, dtype=np.int16),
    ]
    active = columns >= 0
    return sparse.csr_matrix(
        (
            np.ones(int(active.sum()), dtype=np.float64),
            (np.flatnonzero(active), columns[active]),
        ),
        shape=(len(columns), len(table)),
    )


def aggregate_gc_interaction_likelihood(
    counts: KmerGCCounts,
    design: HierarchicalKmerDesign,
    *,
    config: GCInteractionLogisticConfig = GCInteractionLogisticConfig(),
    contigs: Iterable[str] | None = None,
) -> AggregateGCInteractionLikelihood:
    """Append exact interaction columns to the EXP-007 likelihood."""

    base = aggregate_binned_gc_binary_likelihood(
        counts,
        design,
        config=config,
        contigs=contigs,
    )
    rows = base.rows.copy()
    rows["kmer_2_code"] = _k2_codes(rows["kmer_6_code"], design)
    interaction = _interaction_matrix(
        rows["kmer_2_code"].to_numpy(),
        rows["gc_bin_code"].to_numpy(),
        config,
    )
    matrix = sparse.hstack([base.matrix, interaction], format="csr")
    return AggregateGCInteractionLikelihood(
        matrix=matrix,
        target=base.target,
        sample_weight=base.sample_weight,
        rows=rows,
    )


def _matrix_for_summary(
    summary: pd.DataFrame,
    design: HierarchicalKmerDesign,
    config: GCInteractionLogisticConfig,
) -> sparse.csr_matrix:
    main = _binned_matrix_for_summary(summary, design, config)
    gc_codes = gc_bin_codes(summary["gc_count"], summary["valid_count"])
    k2_codes = _k2_codes(summary["kmer_6_code"], design)
    return sparse.hstack(
        [main, _interaction_matrix(k2_codes, gc_codes, config)],
        format="csr",
    )


def _coefficient_table(
    estimator: LogisticRegression,
    design: HierarchicalKmerDesign,
    config: GCInteractionLogisticConfig,
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
    interactions = interaction_feature_table(config).copy()
    interactions["feature_index"] = (
        len(kmer) + len(bins) + interactions["interaction_index"]
    )
    interactions["level_k"] = 2
    interactions["category_code"] = interactions["interaction_index"]
    interactions["category"] = (
        interactions["k2"].astype(str) + " x " + interactions["gc_bin"].astype(str)
    )
    interactions = interactions[
        [
            "feature_index",
            "level_k",
            "category_code",
            "category",
            "feature_name",
            "feature_group",
        ]
    ]
    table = pd.concat([kmer, bins, interactions], ignore_index=True)
    table["coefficient"] = estimator.coef_[0]
    table["absolute_coefficient"] = np.abs(estimator.coef_[0])
    return table


def _fit_from_aggregate(
    aggregate: AggregateGCInteractionLikelihood,
    design: HierarchicalKmerDesign,
    *,
    C: float,
    config: GCInteractionLogisticConfig,
) -> GCInteractionLogisticFit:
    estimator = LogisticRegression(
        penalty=config.penalty,
        C=float(C),
        solver=config.solver,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.seed,
    )
    started = time.monotonic()
    # Intentionally no threadpool_limits context.  In this Windows environment
    # its OpenMP introspection sees libiomp and libomp together and emits the
    # warning reported after EXP-007.  Removing introspection does not change
    # the estimator, objective, solver, or fitted coefficients.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(
            aggregate.matrix,
            aggregate.target,
            sample_weight=aggregate.sample_weight,
        )
    elapsed = time.monotonic() - started
    messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
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
                "converged": not messages and iterations < config.max_iter,
                "fit_seconds": elapsed,
                "aggregate_rows": len(aggregate.rows),
                "features": aggregate.matrix.shape[1],
                "interaction_features": len(interaction_feature_table(config)),
                "represented_positions": int(aggregate.rows["population_count"].sum()),
                "represented_positives": int(
                    aggregate.rows.loc[
                        aggregate.rows["target"] == 1, "population_count"
                    ].sum()
                ),
                "reference_k2": K2_REFERENCE_LABEL,
                "reference_gc_bin": GC_BIN_LABELS[config.reference_bin],
                "intercept": float(estimator.intercept_[0]),
                "warnings": " | ".join(messages),
            }
        ]
    )
    return GCInteractionLogisticFit(
        estimator=estimator,
        design=design,
        aggregate=aggregate,
        coefficient_table=coefficients,
        fit_report=fit_report,
    )


def fit_gc_interaction_logistic(
    counts: KmerGCCounts,
    *,
    C: float,
    config: GCInteractionLogisticConfig = GCInteractionLogisticConfig(),
    contigs: Iterable[str] | None = None,
    design: HierarchicalKmerDesign | None = None,
) -> GCInteractionLogisticFit:
    if counts.split_name != "train":
        raise ValueError("fit_gc_interaction_logistic requires train counts")
    if C not in config.C_grid:
        raise ValueError("C must belong to the preregistered five-value grid")
    design = design or hierarchical_kmer_design(config)
    aggregate = aggregate_gc_interaction_likelihood(
        counts,
        design,
        config=config,
        contigs=contigs,
    )
    return _fit_from_aggregate(aggregate, design, C=C, config=config)


def _evaluate_segment(
    counts: KmerGCCounts,
    fit: GCInteractionLogisticFit,
    config: GCInteractionLogisticConfig,
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
        score_table, on=keys, how="left", validate="many_to_one"
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


def evaluate_gc_interaction_logistic(
    counts: KmerGCCounts,
    fit: GCInteractionLogisticFit,
    *,
    config: GCInteractionLogisticConfig = GCInteractionLogisticConfig(),
    contigs: Iterable[str] | None = None,
    variant: str = "EXP-008 LR + k2 x GC-bin",
    include_curve: bool = True,
    include_per_contig: bool = True,
) -> GCWindowEvaluation:
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
        per_contig = pd.concat(
            [
                _evaluate_segment(
                    counts,
                    fit,
                    config,
                    contigs=[contig],
                    variant=variant,
                    include_curve=False,
                )[0]
                for contig in selected
            ],
            ignore_index=True,
        ).rename(columns={"segment": "contig"})
    return GCWindowEvaluation(overall, per_contig, curve, predictions)


def cross_validate_gc_interaction_logistic(
    train_counts: KmerGCCounts,
    fold_assignment: pd.DataFrame,
    *,
    config: GCInteractionLogisticConfig = GCInteractionLogisticConfig(),
    verbose: bool = False,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> pd.DataFrame:
    """Run exactly five C values across three train-contig folds."""

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
    if total_runs != 15:
        raise AssertionError("EXP-008 protocol requires exactly 15 CV fits")
    run_index = 0
    interaction_table = interaction_feature_table(config)
    for fold in folds:
        validation_contigs = tuple(
            fold_assignment.loc[
                fold_assignment["fold"].astype(int) == fold, "contig"
            ].astype(str)
        )
        fitting_contigs = tuple(sorted(expected - set(validation_contigs)))
        aggregate = aggregate_gc_interaction_likelihood(
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
            evaluated = evaluate_gc_interaction_logistic(
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
            interaction_coefficients = fitted.coefficient_table.loc[
                fitted.coefficient_table["feature_group"] == "k2 x GC201 bin",
                "coefficient",
            ].to_numpy(dtype=np.float64)
            if len(interaction_coefficients) != len(interaction_table):
                raise AssertionError("interaction coefficient vocabulary mismatch")
            for index, value in enumerate(interaction_coefficients):
                record[f"interaction_coefficient_{index}"] = float(value)
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


def selected_cv_interaction_coefficients(
    cv_results: pd.DataFrame,
    *,
    C: float,
    config: GCInteractionLogisticConfig = GCInteractionLogisticConfig(),
) -> pd.DataFrame:
    """Return fold-by-feature interaction coefficients for the selected C."""

    selected = cv_results.loc[np.isclose(cv_results["C"], float(C))]
    vocabulary = interaction_feature_table(config)
    records: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        for feature in vocabulary.itertuples(index=False):
            coefficient = float(
                getattr(row, f"interaction_coefficient_{feature.interaction_index}")
            )
            records.append(
                {
                    "fold": int(row.fold),
                    "interaction_index": int(feature.interaction_index),
                    "k2_code": int(feature.k2_code),
                    "k2": str(feature.k2),
                    "gc_bin_code": int(feature.gc_bin_code),
                    "gc_bin": str(feature.gc_bin),
                    "coefficient": coefficient,
                    "odds_multiplier": float(np.exp(coefficient)),
                }
            )
    return pd.DataFrame(records)


def interaction_enrichment_table(
    counts: KmerGCCounts,
    design: HierarchicalKmerDesign,
    *,
    contigs: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Exact train-only support and target enrichment for all k2-by-GC cells."""

    summary = _joint_summary(counts, contigs=contigs)
    summary["k2_code"] = _k2_codes(summary["kmer_6_code"], design)
    summary["gc_bin_code"] = gc_bin_codes(
        summary["gc_count"], summary["valid_count"]
    )
    grouped = summary.groupby(["k2_code", "gc_bin_code"], as_index=False).agg(
        positions=("total_positions", "sum"),
        positives=("positive_positions", "sum"),
    )
    full = pd.MultiIndex.from_product(
        [range(len(K2_LABELS)), range(len(GC_BIN_LABELS))],
        names=["k2_code", "gc_bin_code"],
    ).to_frame(index=False)
    grouped = full.merge(grouped, on=["k2_code", "gc_bin_code"], how="left").fillna(0)
    grouped[["positions", "positives"]] = grouped[
        ["positions", "positives"]
    ].astype(np.int64)
    grouped["k2"] = np.asarray(K2_LABELS, dtype=object)[
        grouped["k2_code"].to_numpy(dtype=np.int16)
    ]
    grouped["gc_bin"] = np.asarray(GC_BIN_LABELS, dtype=object)[
        grouped["gc_bin_code"].to_numpy(dtype=np.int16)
    ]
    grouped["positive_rate"] = np.divide(
        grouped["positives"],
        grouped["positions"],
        out=np.full(len(grouped), np.nan),
        where=grouped["positions"].to_numpy() > 0,
    )
    prevalence = summary["positive_positions"].sum() / summary[
        "total_positions"
    ].sum()
    grouped["enrichment"] = grouped["positive_rate"] / prevalence
    return grouped[
        [
            "k2_code",
            "k2",
            "gc_bin_code",
            "gc_bin",
            "positions",
            "positives",
            "positive_rate",
            "enrichment",
        ]
    ]


__all__ = [
    "INTERACTION_C_GRID",
    "K2_LABELS",
    "K2_REFERENCE_CODE",
    "K2_REFERENCE_LABEL",
    "AggregateGCInteractionLikelihood",
    "GCInteractionLogisticConfig",
    "GCInteractionLogisticFit",
    "aggregate_gc_interaction_likelihood",
    "cross_validate_gc_interaction_logistic",
    "evaluate_gc_interaction_logistic",
    "fit_gc_interaction_logistic",
    "interaction_enrichment_table",
    "interaction_feature_table",
    "selected_cv_interaction_coefficients",
]
