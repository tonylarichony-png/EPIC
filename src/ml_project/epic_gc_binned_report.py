"""Synchronize completed EXP-007 artifacts into its Markdown card."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .epic_gc_report import (
    _image,
    _link,
    _load_json,
    _markdown,
    _percent,
    _update_block,
    _update_frontmatter,
)


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def build_exp007_report(project_root: Path, *, run_name: str = "run_001") -> str:
    root = Path(project_root).resolve()
    run_dir = root / "artifacts/experiments/EXP-007" / run_name
    required = [
        "metadata.json",
        "config.json",
        "criteria.csv",
        "metric_summary.csv",
        "per_contig_metrics.csv",
        "cv_summary.csv",
        "cv_results.csv",
        "paired_cv_gate.csv",
        "selected_cv_bin_coefficients.csv",
        "fit_report.csv",
        "coefficient_table.csv",
        "final_bin_coefficients.csv",
        "train_bin_enrichment.csv",
        "fold_bin_enrichment.csv",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing EXP-007 artifacts: " + ", ".join(missing))

    metadata = _load_json(run_dir / "metadata.json")
    config = _load_json(run_dir / "config.json")
    metrics = metadata["metrics"]
    run = metadata["run"]
    contract = metadata["data_contract"]
    criteria = pd.read_csv(run_dir / "criteria.csv")
    metric_summary = pd.read_csv(run_dir / "metric_summary.csv")
    per_contig = pd.read_csv(run_dir / "per_contig_metrics.csv")
    cv_summary = pd.read_csv(run_dir / "cv_summary.csv")
    cv_results = pd.read_csv(run_dir / "cv_results.csv")
    paired_gate = pd.read_csv(run_dir / "paired_cv_gate.csv")
    cv_bin_coefs = pd.read_csv(run_dir / "selected_cv_bin_coefficients.csv")
    fit = pd.read_csv(run_dir / "fit_report.csv").iloc[0]
    coefficients = pd.read_csv(run_dir / "coefficient_table.csv")
    final_bins = pd.read_csv(run_dir / "final_bin_coefficients.csv")
    train_bins = pd.read_csv(run_dir / "train_bin_enrichment.csv")
    fold_bins = pd.read_csv(run_dir / "fold_bin_enrichment.csv")

    exp003_ap = float(metrics["reference_average_precision"])
    exp005_ap = float(metrics["ablation_reference_average_precision"])
    exp006_ap = float(metrics["linear_gc_average_precision"])
    candidate_ap = float(metrics["candidate_average_precision"])
    exp003_spearman = float(metrics["reference_epic_spearman"])
    exp005_spearman = float(metrics["ablation_reference_epic_spearman"])
    exp006_spearman = float(metrics["linear_gc_epic_spearman"])
    candidate_spearman = float(metrics["candidate_epic_spearman"])
    selected_c = float(config["selected_C"])
    validation = next(
        item for item in contract["summary"] if item["split"] == "validation"
    )

    overview = pd.DataFrame(
        [
            ("Эксперимент", "EXP-007 — Hierarchical 2/4/6-mer LR + fixed GC201 bins"),
            ("Гипотеза", "Немонотонный категориальный GC201 переносится лучше линейного GC"),
            ("Одно изменение", "EXP-005 + fixed reference-coded GC201 bins; без linear GC"),
            ("Решение", str(run["decision"])),
            ("Run", run_name),
            ("Split", str(contract["split_version"])),
            (
                "Validation",
                f"{int(validation['position_strand_count']):,} позиций; "
                f"{int(validation['positive_positions']):,} positive",
            ),
            ("Ablation reference", "EXP-005/run_001"),
            ("Diagnostic reference", "EXP-006/run_001"),
            ("Champion reference", "EXP-003/run_001"),
            ("Выбранный C", f"{selected_c:g}"),
            ("Reference GC-bin", str(config["reference_bin_label"])),
            ("Notebook", _link(root, root / str(run["notebook"]))),
            (
                "Код",
                _link(
                    root,
                    root / "src/ml_project/epic_gc_binned_logistic.py",
                    "epic_gc_binned_logistic.py",
                ),
            ),
        ],
        columns=["Поле", "Значение"],
    )

    criteria_display = criteria.copy()
    criteria_display["passed"] = criteria_display["passed"].map(
        lambda value: "да" if _truth(value) else "нет"
    )
    metrics_display = metric_summary.copy()
    for column in ["average_precision", "epic_spearman"]:
        metrics_display[column] = metrics_display[column].map(lambda value: f"{value:.9f}")

    paired_display = paired_gate.copy()
    for column in [
        "candidate_average_precision",
        "exp005_average_precision",
        "ap_delta",
    ]:
        paired_display[column] = paired_display[column].map(lambda value: f"{value:+.9f}")
    paired_display["relative_ap_delta"] = paired_display["relative_ap_delta"].map(
        _percent
    )
    for column in [
        "candidate_epic_spearman",
        "exp005_epic_spearman",
        "spearman_delta",
    ]:
        paired_display[column] = paired_display[column].map(lambda value: f"{value:+.6f}")

    cv_display = cv_summary.copy()
    cv_display["selected"] = cv_display["C"].map(
        lambda value: "да" if math.isclose(float(value), selected_c) else ""
    )
    cv_display["all_folds_converged"] = cv_display["all_folds_converged"].map(
        lambda value: "да" if _truth(value) else "нет"
    )
    selected_folds = cv_results.loc[
        cv_results["C"].sub(selected_c).abs() <= 1e-15,
        [
            "fold",
            "validation_contigs",
            "average_precision",
            "epic_spearman",
            "balanced_log_loss",
            "iterations",
            "converged",
        ],
    ]
    strongest = coefficients.loc[
        coefficients["feature_group"] != "GC201 bin"
    ].nlargest(15, "absolute_coefficient")[
        ["feature_name", "feature_group", "coefficient", "absolute_coefficient"]
    ]

    per_contig_display = per_contig.copy()
    per_contig_display["AP Δ vs EXP-005"] = (
        per_contig_display["candidate_average_precision"]
        - per_contig_display["exp005_average_precision"]
    ).map(lambda value: f"{value:+.9f}")
    per_contig_display["AP Δ vs EXP-003"] = (
        per_contig_display["candidate_average_precision"]
        - per_contig_display["exp003_average_precision"]
    ).map(lambda value: f"{value:+.9f}")

    plot_dir = run_dir / "plots"
    plots = [
        "01_train_bin_enrichment.png",
        "02_train_cv_diagnostics.png",
        "03_paired_cv_gate.png",
        "04_cv_bin_coefficients.png",
        "05_validation_metrics.png",
        "06_per_contig_metrics.png",
        "07_precision_recall.png",
        "08_final_coefficients.png",
    ]
    missing_plots = [name for name in plots if not (plot_dir / name).is_file()]
    if missing_plots:
        raise FileNotFoundError("Missing EXP-007 plots: " + ", ".join(missing_plots))

    artifact_names = [
        *required,
        "metrics.csv",
        "precision_recall.csv",
        "positive_predictions.csv.gz",
    ]
    artifact_lines = [f"- {_link(root, run_dir / name, name)}" for name in artifact_names]

    sections = [
        "## Контракт эксперимента\n\n" + _markdown(overview),
        "## Train-only paired gate против EXP-005\n\n"
        + _markdown(paired_display)
        + "\n\n"
        + _image(root, plot_dir / plots[2]),
        "## Проверка validation-критериев\n\n" + _markdown(criteria_display),
        "## Итоговые validation-метрики\n\n"
        + _markdown(metrics_display)
        + "\n\n"
        + _image(root, plot_dir / plots[4]),
        (
            "> [!summary] Главные изменения\n"
            f"> AP против EXP-005: {_percent(candidate_ap / exp005_ap - 1)}; "
            f"против линейного EXP-006: {_percent(candidate_ap / exp006_ap - 1)}; "
            f"против EXP-003: {_percent(candidate_ap / exp003_ap - 1)}. "
            f"Spearman Δ против EXP-005: {candidate_spearman - exp005_spearman:+.9f}; "
            f"против EXP-006: {candidate_spearman - exp006_spearman:+.9f}; "
            f"против EXP-003: {candidate_spearman - exp003_spearman:+.9f}."
        ),
        "### Результаты по validation-contig\n\n"
        + _markdown(per_contig_display)
        + "\n\n"
        + _image(root, plot_dir / plots[5]),
        "### Precision–Recall\n\n" + _image(root, plot_dir / plots[6]),
        "## Train-only GC201 bins\n\n"
        + _markdown(train_bins)
        + "\n\n"
        + _image(root, plot_dir / plots[0]),
        "### Enrichment по train-fold\n\n" + _markdown(fold_bins),
        "## Train-only выбор C\n\n"
        + _markdown(cv_display)
        + "\n\n"
        + _image(root, plot_dir / plots[1]),
        "### Folds выбранного C\n\n" + _markdown(selected_folds),
        "## Устойчивость коэффициентов GC-bins\n\n"
        + _markdown(cv_bin_coefs)
        + "\n\n"
        + _image(root, plot_dir / plots[3]),
        "## Финальные коэффициенты GC-bins\n\n"
        + "> [!info] Интерпретация\n"
        + "> Коэффициенты — условные вклады в log-odds относительно bin "
        + f"`{config['reference_bin_label']}` при фиксированных k-mer.\n\n"
        + _markdown(final_bins)
        + "\n\n"
        + _image(root, plot_dir / plots[7]),
        "### Сильнейшие k-mer коэффициенты candidate\n\n" + _markdown(strongest),
        "## Финальный fit\n\n" + _markdown(pd.DataFrame([fit.to_dict()])),
        "## Артефакты\n\n" + "\n".join(artifact_lines),
        f"> [!summary] Решение\n> `{run['decision']}` — {run['interpretation']}",
    ]
    return "\n\n".join(sections)


def sync_exp007_report(
    project_root: Path,
    *,
    run_name: str = "run_001",
    note_path: Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    run_dir = root / "artifacts/experiments/EXP-007" / run_name
    note = (note_path or root / "experiments/EXP-007.md").resolve()
    metadata = _load_json(run_dir / "metadata.json")
    _update_frontmatter(
        note,
        {
            "status": "completed",
            "decision": str(metadata["run"]["decision"]),
            "run_name": run_name,
            "selected_C": f"{float(metadata['parameters']['selected_C']):g}",
        },
    )
    _update_block(note, build_exp007_report(root, run_name=run_name))
    return note


__all__ = ["build_exp007_report", "sync_exp007_report"]
