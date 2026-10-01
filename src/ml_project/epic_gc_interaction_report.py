"""Synchronize completed EXP-008 artifacts into its Markdown card."""

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


def build_exp008_report(project_root: Path, *, run_name: str = "run_001") -> str:
    """Build the generated report only from immutable saved artifacts."""

    root = Path(project_root).resolve()
    run_dir = root / "artifacts/experiments/EXP-008" / run_name
    required = [
        "metadata.json",
        "config.json",
        "criteria.csv",
        "metric_summary.csv",
        "per_contig_metrics.csv",
        "cv_summary.csv",
        "cv_results.csv",
        "paired_cv_gate.csv",
        "selected_cv_interaction_coefficients.csv",
        "interaction_stability.csv",
        "fit_report.csv",
        "coefficient_table.csv",
        "final_interactions.csv",
        "train_interaction_enrichment.csv",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing EXP-008 artifacts: " + ", ".join(missing))

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
    stability = pd.read_csv(run_dir / "interaction_stability.csv")
    final_interactions = pd.read_csv(run_dir / "final_interactions.csv")
    enrichment = pd.read_csv(run_dir / "train_interaction_enrichment.csv")
    fit = pd.read_csv(run_dir / "fit_report.csv").iloc[0]

    selected_c = float(config["selected_C"])
    exp007_ap = float(metrics["reference_average_precision"])
    candidate_ap = float(metrics["candidate_average_precision"])
    exp007_spearman = float(metrics["reference_epic_spearman"])
    candidate_spearman = float(metrics["candidate_epic_spearman"])
    validation = next(
        item for item in contract["summary"] if item["split"] == "validation"
    )

    overview = pd.DataFrame(
        [
            ("Эксперимент", "EXP-008 — EXP-007 + interaction(k2, GC201-bin)"),
            (
                "Гипотеза",
                "Эффект динуклеотида [-1,0] зависит от GC-класса окружения",
            ),
            (
                "Одно изменение",
                "EXP-007 + 128 reference-coded k2 × GC201-bin interactions",
            ),
            ("Решение", str(run["decision"])),
            ("Run", run_name),
            ("Split", str(contract["split_version"])),
            (
                "Validation",
                f"{int(validation['position_strand_count']):,} позиций; "
                f"{int(validation['positive_positions']):,} positive",
            ),
            ("Champion reference", "EXP-007/run_001"),
            ("Historical reference", "EXP-003/run_001"),
            ("Protocol status", str(config.get("protocol_status", "confirmatory"))),
            (
                "Train-CV gate",
                "passed" if _truth(config.get("cv_gate_passed", True)) else "failed",
            ),
            ("Выбранный C", f"{selected_c:g}"),
            ("CV budget", "5 C × 3 folds = 15 fits"),
            ("Reference k2", str(config["k2_reference_label"])),
            ("Reference GC-bin", str(config["reference_bin_label"])),
            ("Notebook", _link(root, root / str(run["notebook"]))),
            (
                "Код",
                _link(
                    root,
                    root / "src/ml_project/epic_gc_interaction_logistic.py",
                    "epic_gc_interaction_logistic.py",
                ),
            ),
        ],
        columns=["Поле", "Значение"],
    )

    criteria_display = criteria.copy()
    criteria_display["passed"] = criteria_display["passed"].map(
        lambda value: "да" if _truth(value) else "нет"
    )
    metric_display = metric_summary.copy()
    for column in ["average_precision", "epic_spearman"]:
        metric_display[column] = metric_display[column].map(
            lambda value: f"{float(value):.9f}"
        )

    paired_display = paired_gate.copy()
    for column in [
        "candidate_average_precision",
        "exp007_average_precision",
        "ap_delta",
    ]:
        paired_display[column] = paired_display[column].map(
            lambda value: f"{float(value):+.9f}"
        )
    paired_display["relative_ap_delta"] = paired_display[
        "relative_ap_delta"
    ].map(_percent)
    for column in [
        "candidate_epic_spearman",
        "exp007_epic_spearman",
        "spearman_delta",
    ]:
        paired_display[column] = paired_display[column].map(
            lambda value: f"{float(value):+.6f}"
        )

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

    per_contig_display = per_contig.copy()
    per_contig_display["AP Δ vs EXP-007"] = (
        per_contig_display["candidate_average_precision"]
        - per_contig_display["exp007_average_precision"]
    ).map(lambda value: f"{value:+.9f}")

    strongest_stable = stability.nlargest(30, "mean_absolute_coefficient")
    strongest_final = final_interactions.nlargest(30, "absolute_coefficient")
    strongest_enrichment = enrichment.loc[
        enrichment["positions"] > 0
    ].nlargest(30, "enrichment")

    plot_dir = run_dir / "plots"
    plots = [
        "01_train_interaction_enrichment.png",
        "02_train_cv_diagnostics.png",
        "03_paired_cv_gate.png",
        "04_cv_interaction_coefficients.png",
        "05_validation_metrics.png",
        "06_per_contig_metrics.png",
        "07_precision_recall.png",
        "08_final_interactions.png",
    ]
    missing_plots = [name for name in plots if not (plot_dir / name).is_file()]
    if missing_plots:
        raise FileNotFoundError("Missing EXP-008 plots: " + ", ".join(missing_plots))

    artifact_names = [
        *required,
        "metrics.csv",
        "precision_recall.csv",
        "positive_predictions.csv.gz",
    ]
    artifact_lines = [f"- {_link(root, run_dir / name, name)}" for name in artifact_names]

    sections = [
        "## Контракт эксперимента\n\n" + _markdown(overview),
        "## Train-only paired gate против EXP-007\n\n"
        + _markdown(paired_display)
        + "\n\n"
        + _image(root, plot_dir / plots[2]),
        "## Проверка validation-критериев\n\n" + _markdown(criteria_display),
        "## Итоговые validation-метрики\n\n"
        + _markdown(metric_display)
        + "\n\n"
        + _image(root, plot_dir / plots[4]),
        (
            "> [!summary] Главное изменение\n"
            f"> AP против EXP-007: {_percent(candidate_ap / exp007_ap - 1)}; "
            f"Spearman Δ: {candidate_spearman - exp007_spearman:+.9f}."
        ),
        "### Результаты по validation-contig\n\n"
        + _markdown(per_contig_display)
        + "\n\n"
        + _image(root, plot_dir / plots[5]),
        "### Precision–Recall\n\n" + _image(root, plot_dir / plots[6]),
        "## Train-only support и enrichment взаимодействий\n\n"
        + _markdown(strongest_enrichment)
        + "\n\n"
        + _image(root, plot_dir / plots[0]),
        "## Train-only выбор C\n\n"
        + _markdown(cv_display)
        + "\n\n"
        + _image(root, plot_dir / plots[1]),
        "### Folds выбранного C\n\n" + _markdown(selected_folds),
        "## Устойчивость interaction-коэффициентов между folds\n\n"
        + _markdown(strongest_stable)
        + "\n\n"
        + _image(root, plot_dir / plots[3]),
        "## Финальные interaction-коэффициенты\n\n"
        + "> [!info] Интерпретация\n"
        + "> Это добавочный вклад в log-odds сверх отдельных k2 и GC-bin. "
        + f"Reference: k2 `{config['k2_reference_label']}`, "
        + f"GC-bin `{config['reference_bin_label']}`.\n\n"
        + _markdown(strongest_final)
        + "\n\n"
        + _image(root, plot_dir / plots[7]),
        "## Финальный fit\n\n" + _markdown(pd.DataFrame([fit.to_dict()])),
        "## Артефакты\n\n" + "\n".join(artifact_lines),
        f"> [!summary] Решение\n> `{run['decision']}` — {run['interpretation']}",
    ]
    return "\n\n".join(sections)


def sync_exp008_report(
    project_root: Path,
    *,
    run_name: str = "run_001",
    note_path: Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    run_dir = root / "artifacts/experiments/EXP-008" / run_name
    note = (note_path or root / "experiments/EXP-008.md").resolve()
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
    _update_block(note, build_exp008_report(root, run_name=run_name))
    return note


__all__ = ["build_exp008_report", "sync_exp008_report"]
