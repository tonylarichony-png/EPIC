"""Synchronize the completed EXP-006 artifacts into its Markdown card."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        value = f"{value:.6f}"
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "> Нет строк для отображения."
    columns = [_clean(column) for column in frame.columns]
    values = [[_clean(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(3, len(column), *(len(row[index]) for row in values))
        for index, column in enumerate(columns)
    ]
    right = {
        _clean(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    }

    def render(row: list[str]) -> str:
        cells = [
            value.rjust(widths[index]) if columns[index] in right else value.ljust(widths[index])
            for index, value in enumerate(row)
        ]
        return "| " + " | ".join(cells) + " |"

    separators = [
        "-" * (width - 1) + ":" if column in right else "-" * width
        for column, width in zip(columns, widths)
    ]
    return "\n".join(
        [render(columns), "| " + " | ".join(separators) + " |", *(render(row) for row in values)]
    )


def _update_frontmatter(path: Path, fields: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Experiment note has no YAML frontmatter: {path}")
    closing = text.find("\n---\n", 4)
    if closing < 0:
        raise ValueError(f"Experiment note has invalid YAML frontmatter: {path}")
    frontmatter = text[4:closing]
    for key, value in fields.items():
        rendered = f"{key}: {value}"
        pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
        if pattern.search(frontmatter):
            frontmatter = pattern.sub(rendered, frontmatter, count=1)
        else:
            frontmatter = frontmatter.rstrip() + "\n" + rendered
    path.write_text("---\n" + frontmatter.rstrip() + text[closing:], encoding="utf-8")


def _update_block(path: Path, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = "<!-- auto:experiment-report:start -->"
    end = "<!-- auto:experiment-report:end -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected one experiment-report marker pair in {path}")
    left = text.index(start) + len(start)
    right = text.index(end)
    path.write_text(
        text[:left] + "\n\n" + content.strip() + "\n\n" + text[right:],
        encoding="utf-8",
    )


def _link(root: Path, path: Path, label: str | None = None) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return f"[[{relative}|{label or path.name}]]"


def _image(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return f"![[{relative}]]"


def _percent(value: float) -> str:
    return f"{value:+.3%}"


def build_exp006_report(project_root: Path, *, run_name: str = "run_001") -> str:
    root = Path(project_root).resolve()
    run_dir = root / "artifacts/experiments/EXP-006" / run_name
    required = [
        "metadata.json",
        "config.json",
        "criteria.csv",
        "metric_summary.csv",
        "per_contig_metrics.csv",
        "cv_summary.csv",
        "cv_results.csv",
        "fit_report.csv",
        "coefficient_table.csv",
        "train_gc_enrichment.csv",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing EXP-006 artifacts: " + ", ".join(missing))

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
    fit = pd.read_csv(run_dir / "fit_report.csv").iloc[0]
    coefficients = pd.read_csv(run_dir / "coefficient_table.csv")
    gc_enrichment = pd.read_csv(run_dir / "train_gc_enrichment.csv")

    exp003_ap = float(metrics["reference_average_precision"])
    candidate_ap = float(metrics["candidate_average_precision"])
    exp003_spearman = float(metrics["reference_epic_spearman"])
    candidate_spearman = float(metrics["candidate_epic_spearman"])
    exp005_ap = float(metrics["ablation_reference_average_precision"])
    exp005_spearman = float(metrics["ablation_reference_epic_spearman"])
    delta_exp003_ap = candidate_ap / exp003_ap - 1
    delta_exp005_ap = candidate_ap / exp005_ap - 1

    validation = next(item for item in contract["summary"] if item["split"] == "validation")
    overview = pd.DataFrame(
        [
            ("Эксперимент", "EXP-006 — Hierarchical 2/4/6-mer LR + GC201"),
            ("Гипотеза", "GC201 добавляет региональный промоторный сигнал сверх локального 2/4/6-mer"),
            ("Одно изменение", "EXP-005 features + один standardized GC fraction в окне [-100,+100]"),
            ("Решение", str(run["decision"])),
            ("Run", run_name),
            ("Split", str(contract["split_version"])),
            ("Validation", f"{int(validation['position_strand_count']):,} позиций; {int(validation['positive_positions']):,} positive"),
            ("Ablation reference", "EXP-005/run_001"),
            ("Champion reference", "EXP-003/run_001"),
            ("Выбранный C", f"{float(config['selected_C']):g}"),
            ("Notebook", _link(root, root / str(run["notebook"]))),
            ("Код", _link(root, root / "src/ml_project/epic_gc_logistic.py", "epic_gc_logistic.py")),
        ],
        columns=["Поле", "Значение"],
    )

    criteria_display = criteria.copy()
    criteria_display["passed"] = criteria_display["passed"].map(
        lambda value: "да"
        if value is True or str(value).strip().lower() == "true"
        else "нет"
    )

    metrics_display = metric_summary.copy()
    metrics_display["average_precision"] = metrics_display["average_precision"].map(
        lambda value: f"{value:.9f}"
    )
    metrics_display["epic_spearman"] = metrics_display["epic_spearman"].map(
        lambda value: f"{value:.9f}"
    )

    per_contig_display = per_contig.copy()
    per_contig_display["AP Δ vs EXP-005"] = (
        per_contig_display["candidate_average_precision"]
        - per_contig_display["exp005_average_precision"]
    ).map(lambda value: f"{value:+.9f}")
    per_contig_display["AP Δ vs EXP-003"] = (
        per_contig_display["candidate_average_precision"]
        - per_contig_display["exp003_average_precision"]
    ).map(lambda value: f"{value:+.9f}")
    per_contig_display = per_contig_display[
        [
            "contig",
            "exp003_average_precision",
            "exp005_average_precision",
            "candidate_average_precision",
            "AP Δ vs EXP-005",
            "AP Δ vs EXP-003",
            "candidate_epic_spearman",
        ]
    ]
    for column in [
        "exp003_average_precision",
        "exp005_average_precision",
        "candidate_average_precision",
        "candidate_epic_spearman",
    ]:
        per_contig_display[column] = per_contig_display[column].map(lambda value: f"{value:.9f}")

    cv_display = cv_summary.copy()
    cv_display["selected"] = cv_display["C"].map(
        lambda value: "да" if math.isclose(float(value), float(config["selected_C"])) else ""
    )
    cv_display["all_folds_converged"] = cv_display["all_folds_converged"].map(
        lambda value: "да" if bool(value) else "нет"
    )
    for column in ["mean_average_precision", "std_average_precision"]:
        cv_display[column] = cv_display[column].map(lambda value: f"{value:.9f}")
    for column in ["mean_epic_spearman", "std_epic_spearman", "mean_balanced_log_loss"]:
        cv_display[column] = cv_display[column].map(lambda value: f"{value:.6f}")

    selected_folds = cv_results.loc[
        cv_results["C"].sub(float(config["selected_C"])).abs() <= 1e-15,
        [
            "fold",
            "validation_contigs",
            "average_precision",
            "epic_spearman",
            "balanced_log_loss",
            "iterations",
            "converged",
            "gc_coefficient",
        ],
    ].copy()

    gc_beta = float(fit["gc_coefficient"])
    gc_std = float(fit["gc_std"])
    gc_effect = pd.DataFrame(
        [
            ("Train mean GC201", f"{float(fit['gc_mean']):.6f}"),
            ("Train std GC201", f"{gc_std:.6f}"),
            ("Coefficient per 1 SD", f"{gc_beta:+.6f}"),
            ("Odds multiplier per 1 SD", f"{math.exp(gc_beta):.4f}x"),
            ("Odds multiplier per +10 pp GC", f"{math.exp(gc_beta * 0.10 / gc_std):.4f}x"),
        ],
        columns=["Показатель", "Значение"],
    )
    strongest = coefficients.loc[coefficients["feature_name"] != "gc201_z"].nlargest(
        15, "absolute_coefficient"
    )[["feature_name", "feature_group", "coefficient", "absolute_coefficient"]]

    plot_dir = run_dir / "plots"
    plots = [
        "01_train_gc_enrichment.png",
        "02_train_cv_diagnostics.png",
        "03_validation_metrics.png",
        "04_per_contig_metrics.png",
        "05_precision_recall.png",
        "06_coefficients_gc.png",
    ]
    missing_plots = [name for name in plots if not (plot_dir / name).is_file()]
    if missing_plots:
        raise FileNotFoundError("Missing EXP-006 plots: " + ", ".join(missing_plots))

    artifact_names = [*required, "metrics.csv", "precision_recall.csv", "positive_predictions.csv.gz"]
    artifact_lines = [f"- {_link(root, run_dir / name, name)}" for name in artifact_names]

    sections = [
        "## Контракт эксперимента\n\n" + _markdown(overview),
        "## Проверка критериев\n\n" + _markdown(criteria_display),
        "## Итоговые validation-метрики\n\n"
        + _markdown(metrics_display)
        + "\n\n"
        + _image(root, plot_dir / plots[2]),
        (
            "> [!summary] Главные изменения\n"
            f"> AP против EXP-005: {_percent(delta_exp005_ap)}. "
            f"AP против EXP-003: {_percent(delta_exp003_ap)}. "
            f"Spearman против EXP-005: {candidate_spearman - exp005_spearman:+.9f}; "
            f"против EXP-003: {candidate_spearman - exp003_spearman:+.9f}."
        ),
        "### Результаты по validation-contig\n\n"
        + _markdown(per_contig_display)
        + "\n\n"
        + _image(root, plot_dir / plots[3]),
        "### Precision–Recall\n\n" + _image(root, plot_dir / plots[4]),
        "## Train-only GC201\n\n"
        + _markdown(gc_enrichment)
        + "\n\n"
        + _image(root, plot_dir / plots[0]),
        "## Train-only выбор C\n\n"
        + _markdown(cv_display)
        + "\n\n"
        + _image(root, plot_dir / plots[1]),
        "### Folds выбранного C\n\n" + _markdown(selected_folds),
        "## Вклад GC201\n\n"
        + "> [!info] Интерпретация\n"
        + "> GC coefficient — аддитивный вклад в log-odds после train-only standardization. "
        + "Это не причинный эффект и не откалиброванная вероятность.\n\n"
        + _markdown(gc_effect)
        + "\n\n"
        + _image(root, plot_dir / plots[5]),
        "### Сильнейшие k-mer коэффициенты candidate\n\n" + _markdown(strongest),
        "## Финальный fit\n\n" + _markdown(pd.DataFrame([fit.to_dict()])),
        "## Артефакты\n\n" + "\n".join(artifact_lines),
        f"> [!summary] Решение\n> `{run['decision']}` — {run['interpretation']}",
    ]
    return "\n\n".join(sections)


def sync_exp006_report(
    project_root: Path,
    *,
    run_name: str = "run_001",
    note_path: Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    run_dir = root / "artifacts/experiments/EXP-006" / run_name
    note = (note_path or root / "experiments/EXP-006.md").resolve()
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
    _update_block(note, build_exp006_report(root, run_name=run_name))
    return note


__all__ = ["build_exp006_report", "sync_exp006_report"]
