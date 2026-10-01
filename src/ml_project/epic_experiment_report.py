"""Build an interpretable Markdown report for the completed EPIC EXP-005 run."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_files(directory: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing EXP-005 artifacts in {directory}: {', '.join(missing)}"
        )


def _update_frontmatter(path: Path, fields: dict[str, str]) -> None:
    """Update scalar fields while preserving the rest of the experiment card."""

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
    path.write_text(
        "---\n" + frontmatter.rstrip() + text[closing:], encoding="utf-8"
    )


def _vault_link(root: Path, path: Path, label: str | None = None) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return f"[[{relative}|{label or path.name}]]"


def _image_link(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return f"![[{relative}]]"


def _signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def _percent(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}%}"


def _yes_no(value: object) -> str:
    return "да" if bool(value) else "нет"


def _markdown(frame: pd.DataFrame, *, right: set[str] | None = None) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""

    if frame.empty:
        return "> Нет строк для отображения."

    def clean(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            value = f"{value:.6f}"
        return (
            str(value)
            .replace("|", r"\|")
            .replace("\r", " ")
            .replace("\n", " ")
        )

    columns = [clean(column) for column in frame.columns]
    values = [
        [clean(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(3, len(column), *(len(row[index]) for row in values))
        for index, column in enumerate(columns)
    ]
    inferred_right = {
        clean(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    }
    right_columns = inferred_right | {clean(column) for column in (right or set())}

    def row_text(row: list[str]) -> str:
        cells = [
            value.rjust(widths[index])
            if columns[index] in right_columns
            else value.ljust(widths[index])
            for index, value in enumerate(row)
        ]
        return "| " + " | ".join(cells) + " |"

    separators = [
        ("-" * (width - 1) + ":")
        if column in right_columns
        else ("-" * width)
        for column, width in zip(columns, widths)
    ]
    return "\n".join(
        [
            row_text(columns),
            "| " + " | ".join(separators) + " |",
            *(row_text(row) for row in values),
        ]
    )


def _update_generated_block(path: Path, block_id: str, content: str) -> None:
    """Replace exactly one generated block without touching manual prose."""

    text = path.read_text(encoding="utf-8")
    start = f"<!-- auto:{block_id}:start -->"
    end = f"<!-- auto:{block_id}:end -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected one {block_id!r} marker pair in {path}")
    start_index = text.index(start) + len(start)
    end_index = text.index(end)
    if start_index > end_index:
        raise ValueError(f"Markers are reversed for {block_id!r} in {path}")
    replacement = "\n\n" + content.strip() + "\n\n"
    path.write_text(
        text[:start_index] + replacement + text[end_index:], encoding="utf-8"
    )


def build_exp005_report(project_root: Path, *, run_name: str = "run_001") -> str:
    """Build the generated report block entirely from saved run artifacts."""

    root = Path(project_root).resolve()
    run_dir = root / "artifacts" / "experiments" / "EXP-005" / run_name
    required = [
        "metadata.json",
        "config.json",
        "metrics.csv",
        "per_contig_metrics.csv",
        "cv_results.csv",
        "cv_summary.csv",
        "fit_report.csv",
        "coefficient_table.csv",
    ]
    _require_files(run_dir, required)

    metadata = _load_json(run_dir / "metadata.json")
    config = _load_json(run_dir / "config.json")
    metrics = metadata["metrics"]
    run = metadata["run"]
    data_contract = metadata["data_contract"]

    per_contig = pd.read_csv(run_dir / "per_contig_metrics.csv")
    cv_results = pd.read_csv(run_dir / "cv_results.csv")
    cv_summary = pd.read_csv(run_dir / "cv_summary.csv")
    fit = pd.read_csv(run_dir / "fit_report.csv").iloc[0]
    coefficients = pd.read_csv(run_dir / "coefficient_table.csv")

    reference_ap = float(metrics["reference_average_precision"])
    candidate_ap = float(metrics["candidate_average_precision"])
    delta_ap = float(metrics["delta_average_precision"])
    relative_ap = float(metrics["relative_average_precision_gain"])
    reference_spearman = float(metrics["reference_epic_spearman"])
    candidate_spearman = float(metrics["candidate_epic_spearman"])
    delta_spearman = float(metrics["delta_epic_spearman"])
    relative_spearman = delta_spearman / reference_spearman

    per_contig["delta_ap"] = (
        per_contig["candidate_average_precision"]
        - per_contig["reference_average_precision"]
    )
    per_contig["relative_ap_gain"] = (
        per_contig["delta_ap"] / per_contig["reference_average_precision"]
    )
    per_contig["delta_spearman"] = (
        per_contig["candidate_epic_spearman"]
        - per_contig["reference_epic_spearman"]
    )
    contig_wins = int((per_contig["delta_ap"] > 0).sum())

    checks = [
        relative_ap >= 0.01,
        candidate_spearman >= reference_spearman,
        contig_wins >= 2,
    ]
    formal_outcome = "passed" if all(checks) else "failed"
    validation_summary = next(
        item
        for item in data_contract["summary"]
        if item["split"] == "validation"
    )
    train_summary = next(
        item for item in data_contract["summary"] if item["split"] == "train"
    )
    manifest = str(data_contract["manifest_sha256"])

    contract = pd.DataFrame(
        [
            ("Эксперимент", "EXP-005 — Hierarchical 2/4/6-mer Logistic Regression"),
            (
                "Гипотеза",
                "L2 logistic regression по hierarchical 2/4/6-mer one-hot "
                "улучшит ранжирование относительно EXP-003 lookup",
            ),
            (
                "Одно изменение",
                "fixed posterior lookup заменён на L2 logistic regression; "
                "окно последовательности и уровни 2/4/6-mer не менялись",
            ),
            (
                "Критерий успеха",
                "relative validation AP gain >= 1%; EPIC Spearman не ниже "
                "EXP-003; AP win минимум на 2 из 3 validation-contig",
            ),
            ("Формальные критерии", formal_outcome),
            ("Решение", str(run["decision"])),
            ("Run", run_name),
            ("Версия split", str(data_contract["split_version"])),
            ("Manifest SHA-256", f"`{manifest[:12]}…`"),
            (
                "Train",
                f"{int(train_summary['position_strand_count']):,} позиций; "
                f"{int(train_summary['positive_positions']):,} positive; "
                f"{int(train_summary['contigs'])} contig",
            ),
            (
                "Validation",
                f"{int(validation_summary['position_strand_count']):,} позиций; "
                f"{int(validation_summary['positive_positions']):,} positive; "
                f"{int(validation_summary['contigs'])} contig",
            ),
            ("Reference", "EXP-003/run_001 — 6-mer lookup"),
            ("Candidate", "L2 logistic regression, hierarchical 2/4/6-mer one-hot"),
            ("Выбранный C", f"{float(config['selected_C']):g}"),
            ("Основная метрика", "Average Precision"),
            ("Дополнительная метрика", "EPIC Spearman"),
            ("Notebook", _vault_link(root, root / str(run["notebook"]))),
            (
                "Код модели",
                _vault_link(
                    root,
                    root / "src/ml_project/epic_kmer_logistic.py",
                    "epic_kmer_logistic.py",
                ),
            ),
        ],
        columns=["Поле", "Значение"],
    )

    criteria = pd.DataFrame(
        [
            ("primary", "Relative AP gain", _percent(relative_ap), ">= +1.000%", checks[0]),
            (
                "guardrail",
                "EPIC Spearman delta",
                _signed(delta_spearman, 9),
                ">= 0",
                checks[1],
            ),
            (
                "guardrail",
                "Validation contig с AP win",
                f"{contig_wins}/3",
                ">= 2/3",
                checks[2],
            ),
        ],
        columns=["Роль", "Метрика", "Наблюдение", "Порог", "Пройден"],
    )

    all_metrics = pd.DataFrame(
        [
            (
                "Average Precision",
                f"{reference_ap:.9f}",
                f"{candidate_ap:.9f}",
                _signed(delta_ap, 9),
                _percent(relative_ap),
            ),
            (
                "EPIC Spearman",
                f"{reference_spearman:.9f}",
                f"{candidate_spearman:.9f}",
                _signed(delta_spearman, 9),
                _percent(relative_spearman),
            ),
        ],
        columns=["Метрика", "EXP-003", "EXP-005", "Абсолютный Δ", "Относительный Δ"],
    )

    contig_table = pd.DataFrame(
        {
            "Contig": per_contig["contig"],
            "AP EXP-003": per_contig["reference_average_precision"].map(
                lambda value: f"{value:.9f}"
            ),
            "AP EXP-005": per_contig["candidate_average_precision"].map(
                lambda value: f"{value:.9f}"
            ),
            "AP Δ": per_contig["delta_ap"].map(lambda value: _signed(value, 9)),
            "AP relative Δ": per_contig["relative_ap_gain"].map(_percent),
            "Spearman Δ": per_contig["delta_spearman"].map(
                lambda value: _signed(value, 9)
            ),
        }
    )

    selected_c = float(config["selected_C"])
    cv_selected = cv_results.loc[
        cv_results["C"].sub(selected_c).abs() <= 1e-15
    ].copy()
    fold_table = pd.DataFrame(
        {
            "Fold": cv_selected["fold"].astype(int),
            "Validation contig": cv_selected["validation_contigs"],
            "AP": cv_selected["average_precision"].map(lambda value: f"{value:.9f}"),
            "Spearman": cv_selected["epic_spearman"].map(lambda value: f"{value:.6f}"),
            "Balanced log-loss": cv_selected["balanced_log_loss"].map(
                lambda value: f"{value:.6f}"
            ),
            "Positions": cv_selected["positions"].map(lambda value: f"{int(value):,}"),
            "Positive": cv_selected["positives"].map(lambda value: f"{int(value):,}"),
            "Iterations": cv_selected["iterations"].astype(int),
            "Converged": cv_selected["converged"].map(_yes_no),
        }
    )

    cv_table = pd.DataFrame(
        {
            "C": cv_summary["C"].map(lambda value: f"{value:g}"),
            "Mean AP": cv_summary["mean_average_precision"].map(
                lambda value: f"{value:.9f}"
            ),
            "Std AP": cv_summary["std_average_precision"].map(
                lambda value: f"{value:.9f}"
            ),
            "Mean Spearman": cv_summary["mean_epic_spearman"].map(
                lambda value: f"{value:.6f}"
            ),
            "Balanced log-loss": cv_summary["mean_balanced_log_loss"].map(
                lambda value: f"{value:.6f}"
            ),
            "Max iterations": cv_summary["max_iterations"].astype(int),
            "Все folds сошлись": cv_summary["all_folds_converged"].map(_yes_no),
            "Выбран": cv_summary["C"].map(
                lambda value: "да" if abs(float(value) - selected_c) <= 1e-15 else ""
            ),
        }
    )
    not_converged = cv_summary.loc[
        ~cv_summary["all_folds_converged"].astype(bool), "C"
    ].map(lambda value: f"{value:g}").tolist()

    fit_table = pd.DataFrame(
        [
            ("Solver / penalty", f"{fit['solver']} / {fit['penalty']}"),
            ("C", f"{float(fit['C']):g}"),
            ("Tolerance", f"{float(fit['tol']):.0e}"),
            ("Iterations", f"{int(fit['iterations'])} / {int(fit['max_iter'])}"),
            ("Converged", _yes_no(fit["converged"])),
            ("Fit seconds", f"{float(fit['fit_seconds']):.3f}"),
            ("Aggregated rows", f"{int(fit['aggregate_rows']):,}"),
            ("Features", f"{int(fit['features']):,}"),
            ("Represented positions", f"{int(fit['represented_positions']):,}"),
            ("Represented positives", f"{int(fit['represented_positives']):,}"),
            ("Intercept", f"{float(fit['intercept']):+.6f}"),
        ],
        columns=["Параметр", "Значение"],
    )

    level_rows: list[tuple[object, ...]] = []
    for level, group in coefficients.groupby("level_k", sort=True):
        absolute = group["absolute_coefficient"]
        level_rows.append(
            (
                f"{int(level)}-mer",
                len(group),
                f"{absolute.mean():.6f}",
                f"{absolute.median():.6f}",
                f"{absolute.max():.6f}",
                1,
            )
        )
    level_table = pd.DataFrame(
        level_rows,
        columns=[
            "Уровень",
            "Признаков",
            "Mean abs(coef)",
            "Median abs(coef)",
            "Max abs(coef)",
            "Активных на позицию",
        ],
    )

    def coefficient_table(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Признак": frame["feature_name"],
                "Коэффициент": frame["coefficient"].map(
                    lambda value: f"{value:+.6f}"
                ),
                "Множитель odds": frame["coefficient"].map(
                    lambda value: f"{math.exp(float(value)):.3f}x"
                ),
            }
        )

    top_positive = coefficient_table(coefficients.nlargest(10, "coefficient"))
    top_negative = coefficient_table(coefficients.nsmallest(10, "coefficient"))

    plot_dir = run_dir / "plots"
    plot_paths = {
        "cv": plot_dir / "01_train_cv_diagnostics.png",
        "metrics": plot_dir / "02_validation_metrics.png",
        "contigs": plot_dir / "03_per_contig_metrics.png",
        "pr": plot_dir / "04_precision_recall.png",
        "coefficients": plot_dir / "05_coefficients.png",
    }
    _require_files(plot_dir, [path.name for path in plot_paths.values()])

    artifact_names = [
        "config.json",
        "metadata.json",
        "metrics.csv",
        "metric_summary.csv",
        "per_contig_metrics.csv",
        "cv_fold_assignment.csv",
        "cv_results.csv",
        "cv_summary.csv",
        "fit_report.csv",
        "feature_table.csv",
        "coefficient_table.csv",
        "precision_recall.csv",
        "positive_predictions.csv.gz",
    ]
    artifact_lines = [
        f"- {_vault_link(root, run_dir / name, name)}" for name in artifact_names
    ]

    sections = [
        "## Контракт эксперимента\n\n" + _markdown(contract),
        (
            "> [!note] Как читать Δ\n"
            "> Положительный Δ означает улучшение EXP-005 относительно EXP-003. "
            "Для AP отдельно показан относительный прирост, потому что абсолютные "
            "значения малы из-за сильного дисбаланса классов."
        ),
        "## Проверка pre-registered criteria\n\n" + _markdown(criteria),
        (
            "> [!failure] Формальный итог: критерий не пройден\n"
            f"> AP вырос на {_percent(relative_ap)}, но требовалось не меньше +1.000%. "
            f"Оба guardrail пройдены: Spearman вырос, AP выше reference на {contig_wins}/3 contig."
        ),
        "## Сравнение итоговых validation-метрик\n\n"
        + _markdown(all_metrics)
        + "\n\n"
        + _image_link(root, plot_paths["metrics"]),
        "### Результаты по validation-contig\n\n"
        + _markdown(contig_table)
        + "\n\n"
        + _image_link(root, plot_paths["contigs"]),
        (
            "### Precision–Recall на полной validation\n\n"
            + _image_link(root, plot_paths["pr"])
            + "\n\n"
            + f"Validation prevalence равна {float(validation_summary['positive_percent']) / 100:.9f}. "
            + "AP кандидата выше случайного baseline примерно в "
            + f"{candidate_ap / (float(validation_summary['positive_percent']) / 100):.3f} раза."
        ),
        "## Выбор регуляризации только на train\n\n"
        + f"Выбран `C={selected_c:g}`. Он дал лучший mean CV AP среди сошедшихся "
        + "вариантов и прошёл Spearman-guardrail правила выбора. "
        + (
            f"Не сошлись при `max_iter={int(config['max_iter'])}`: "
            + ", ".join(f"`C={value}`" for value in not_converged)
            + ". Они исключены из выбора."
            if not_converged
            else "Все варианты C сошлись."
        )
        + "\n\n"
        + _image_link(root, plot_paths["cv"]),
        "### Полная CV-сетка\n\n" + _markdown(cv_table),
        "### Folds для выбранного C\n\n" + _markdown(fold_table),
        "## Финальное обучение на всём train\n\n"
        + _markdown(fit_table)
        + "\n\n"
        + "> [!info] Почему fit занял доли секунды\n"
        + "> Время относится только к L-BFGS на 8 194 агрегированных строках. "
        + "Они точно представляют все 228 064 160 train-позиций через `sample_weight`; "
        + "это не sampling.",
        "## Вклад признаков\n\n"
        + "> [!info] Граница интерпретации\n"
        + "> Коэффициент показывает вклад активного one-hot признака в log-odds модели. "
        + "Это не причинный эффект и не permutation importance. Уровни вложены: каждый "
        + "6-mer однозначно задаёт свои suffix 4-mer и 2-mer, поэтому L2 распределяет "
        + "сигнал между коррелированными уровнями. Из-за class balancing sigmoid(score) "
        + "нельзя трактовать как откалиброванную вероятность события.\n\n"
        + "Для каждой позиции модель считает `score = intercept + coef(2-mer) + "
        + "coef(4-mer) + coef(6-mer)`. В этом запуске intercept равен "
        + f"`{float(fit['intercept']):+.6f}`.\n\n"
        + "### Сила коэффициентов по уровням\n\n"
        + _markdown(level_table)
        + "\n\nНа каждой канонической позиции активен ровно один признак каждого уровня. "
        + "Поэтому `Mean abs(coef)` удобнее для сравнения типичной силы уровня, чем сумма "
        + "модулей: у 6-mer просто намного больше категорий.\n\n"
        + _image_link(root, plot_paths["coefficients"]),
        "### Наиболее сильные положительные коэффициенты\n\n"
        + _markdown(top_positive)
        + "\n\nПоложительный коэффициент повышает score. Например, `k6=CGCCAT` "
        + "добавляет `+1.211` к log-odds, то есть умножает odds примерно на `3.358x` "
        + "при фиксированных остальных активных признаках.",
        "### Наиболее сильные отрицательные коэффициенты\n\n"
        + _markdown(top_negative)
        + "\n\nОтрицательный коэффициент понижает score. Например, `k6=GGGGGG` "
        + "добавляет `-1.294` к log-odds, то есть оставляет примерно `0.274x` исходных odds.",
        "## Артефакты\n\n" + "\n".join(artifact_lines),
        (
            "> [!summary] Вывод\n"
            f"> EXP-005 дал небольшое согласованное улучшение: AP {_percent(relative_ap)}, "
            f"Spearman {_percent(relative_spearman)}, AP win на {contig_wins}/3 contig. "
            "Но основной заранее заданный порог AP +1% не достигнут, поэтому решение "
            f"остается `{run['decision']}`, а EXP-003 — текущим champion."
        ),
    ]
    return "\n\n".join(sections)


def sync_exp005_report(
    project_root: Path,
    *,
    run_name: str = "run_001",
    note_path: Path | None = None,
) -> Path:
    """Refresh only the generated block and status fields of EXP-005.md."""

    root = Path(project_root).resolve()
    note = (note_path or root / "experiments" / "EXP-005.md").resolve()
    run_dir = root / "artifacts" / "experiments" / "EXP-005" / run_name
    metadata = _load_json(run_dir / "metadata.json")
    decision = str(metadata["run"]["decision"])
    _update_frontmatter(
        note,
        {
            "status": "completed",
            "decision": decision,
            "run_name": run_name,
            "selected_C": f"{float(metadata['parameters']['selected_C']):g}",
        },
    )
    report = build_exp005_report(root, run_name=run_name)
    _update_generated_block(note, "experiment-report", report)
    return note


__all__ = ["build_exp005_report", "sync_exp005_report"]
