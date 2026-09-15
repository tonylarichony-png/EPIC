"""Pure Markdown block builders for modeling reports."""

from __future__ import annotations

import platform
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd

from ..docsync import dataframe_to_markdown
from ._utils import _display_value
from .contracts import (
    ModelingSettings,
    FeaturePlan,
    PreparedData,
    SavedBaselineRun,
    ScoringPlan,
)


def _vault_relative(project_root: Path, path: Path) -> str:
    root = Path(project_root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Path resolves outside the project root: {path}") from error


def build_secondary_metrics_block(scoring: ScoringPlan) -> str:
    """Render secondary metric implementations resolved from the config."""

    rows = [
        {
            "Метрика": scoring.labels[key],
            "Scorer / implementation": _display_value(scoring.scorers[key]),
            "Направление": scoring.directions[key],
        }
        for key in scoring.scorers
        if key != "primary"
    ]
    if not rows:
        return (
            "> Вторичные метрики не настроены. Добавьте их в "
            "`secondary_scorers` объекта `BASELINE` в "
            "`src/ml_project/baseline_config.py`."
        )
    return dataframe_to_markdown(pd.DataFrame(rows))


def build_validation_protocol_block(
    settings: ModelingSettings,
    scoring: ScoringPlan,
    *,
    cv_description: str,
) -> str:
    """Render the executable validation contract from ModelingSettings."""

    primary_direction = scoring.directions["primary"]
    report = pd.DataFrame(
        [
            ("Тип задачи", settings.task_type or "не настроен"),
            ("Протокол", cv_description),
            ("CV strategy из конфига", settings.cv_strategy),
            ("Число folds", settings.n_splits),
            ("Shuffle", settings.shuffle),
            ("Seed", _display_value(settings.random_state)),
            ("Group column", _display_value(settings.group_column)),
            ("Time column", _display_value(settings.time_column)),
            ("Основная метрика", scoring.contract_metric),
            ("Основной scorer", _display_value(scoring.scorers["primary"])),
            ("Направление", primary_direction),
            ("N jobs", _display_value(settings.n_jobs)),
            ("Error score", _display_value(settings.error_score)),
            ("Train score сохраняется", settings.return_train_score),
            (
                "Граница preprocessing",
                "fit только внутри train fold sklearn Pipeline",
            ),
        ],
        columns=["Параметр", "Исполняемое значение"],
    )
    return dataframe_to_markdown(report)


def _installed_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "не установлен"


def build_reproducibility_block(
    project_root: Path,
    settings: ModelingSettings,
    *,
    dataset_version: str,
    cv_description: str,
    saved_run: SavedBaselineRun | None = None,
) -> str:
    """Render paths, versions and seeds needed to reproduce a saved run."""

    root = Path(project_root).resolve()
    expected_run_dir = (root / settings.artifact_dir / settings.run_name).resolve()
    run_dir = saved_run.run_dir.resolve() if saved_run is not None else expected_run_dir
    run_exists = run_dir.exists()
    environment = "; ".join(
        [
            f"Python {platform.python_version()}",
            f"numpy {_installed_version('numpy')}",
            f"pandas {_installed_version('pandas')}",
            f"scikit-learn {_installed_version('scikit-learn')}",
        ]
    )
    report = pd.DataFrame(
        [
            ("Dataset version", dataset_version),
            ("Baseline config", "`src/ml_project/baseline_config.py`"),
            ("Dataset config", "`src/ml_project/config.py`"),
            ("Split / evaluation code", "`src/ml_project/modeling/validation.py`"),
            ("Feature code", "`src/ml_project/modeling/features.py`"),
            ("Run", settings.run_name),
            ("Validation", cv_description),
            ("Seed policy", f"RANDOM_STATE={_display_value(settings.random_state)}"),
            ("Environment", environment),
            (
                "Run artifacts",
                (
                    f"`{_vault_relative(root, run_dir)}/`"
                    if run_exists
                    else f"не сохранены (`{_vault_relative(root, run_dir)}/`)"
                ),
            ),
            (
                "Final model",
                (
                    f"`{_vault_relative(root, saved_run.model_path)}`"
                    if saved_run is not None and saved_run.model_path is not None
                    else "не сохранялась"
                ),
            ),
        ],
        columns=["Поле", "Значение"],
    )
    return dataframe_to_markdown(report)


def build_model_ready_block(
    settings: ModelingSettings,
    plan: FeaturePlan,
    *,
    dataset_version: str,
    data: PreparedData | None = None,
) -> str:
    """Render the factual model-ready dataset contract."""

    report = pd.DataFrame(
        [
            ("Версия данных", dataset_version),
            ("Включённые группы", ", ".join(settings.model_feature_groups) or "—"),
            ("Числовые признаки", ", ".join(plan.numeric) or "—"),
            ("Категориальные признаки", ", ".join(plan.categorical) or "—"),
            ("Исключённые признаки", ", ".join(plan.excluded) or "—"),
            ("Признаков в модели", len(plan.model_features)),
            ("Строк в model-ready train", len(data.X) if data is not None else "—"),
            (
                "Target",
                str(data.y.name) if data is not None and data.y.name is not None else "—",
            ),
            (
                "Inference schema обязательна",
                settings.require_inference_features,
            ),
        ],
        columns=["Поле", "Исполняемое значение"],
    )
    return dataframe_to_markdown(report)


def build_feature_registry_block(plan: FeaturePlan) -> str:
    """Render used and excluded features from the resolved feature plan."""

    report = plan.to_frame().rename(
        columns={
            "feature": "Признак",
            "config_group": "Группа",
            "model_role": "Роль в модели",
            "status": "Статус",
            "reason": "Причина",
        }
    )
    return dataframe_to_markdown(report)


def build_preprocessing_block(
    settings: ModelingSettings,
    plan: FeaturePlan,
) -> str:
    """Render preprocessing that is actually built by the baseline pipeline."""

    report = pd.DataFrame(
        [
            ("Numeric", "Признаки", ", ".join(plan.numeric) or "—"),
            ("Numeric", "Imputer", settings.numeric_imputer),
            (
                "Numeric",
                "Missing indicator",
                settings.add_numeric_missing_indicator,
            ),
            ("Numeric", "Scaler", settings.numeric_scaler),
            ("Categorical", "Признаки", ", ".join(plan.categorical) or "—"),
            ("Categorical", "Imputer", settings.categorical_imputer),
            (
                "Categorical",
                "Fill value",
                _display_value(settings.categorical_fill_value),
            ),
            (
                "Categorical",
                "Unknown categories",
                settings.onehot_handle_unknown,
            ),
            (
                "Categorical",
                "Min frequency",
                _display_value(settings.onehot_min_frequency),
            ),
            (
                "Categorical",
                "Max categories",
                _display_value(settings.onehot_max_categories),
            ),
            (
                "Categorical",
                "Sparse output",
                settings.onehot_sparse_output,
            ),
        ],
        columns=["Группа", "Шаг", "Исполняемое значение"],
    )
    return dataframe_to_markdown(report)


def build_key_results_block(
    *,
    baseline_metric: str,
    baseline_score: float,
    baseline_note: Path | None = None,
    experiments: pd.DataFrame | None = None,
) -> str:
    """Render the README leaderboard with links to experiment cards."""

    baseline_label = baseline_note.stem if baseline_note is not None else "Baseline"
    baseline_link = (
        f"[[{baseline_note.as_posix()}|{baseline_label}]]"
        if baseline_note is not None
        else baseline_label
    )
    rows: list[dict[str, str]] = [
        {
            "Версия / эксперимент": baseline_link,
            "Метрика": baseline_metric,
            "Значение": f"{baseline_score:.4f}",
            "Δ к baseline": "—",
            "Решение": "reference",
        }
    ]

    if experiments is not None and not experiments.empty:
        required = {
            "experiment_id",
            "title",
            "note",
            "primary_metric",
            "candidate_score",
            "improvement",
            "decision",
        }
        missing = sorted(required - set(experiments.columns))
        if missing:
            raise ValueError(
                "Experiment results registry is missing columns: "
                + ", ".join(missing)
            )
        for record in experiments.to_dict(orient="records"):
            experiment_id = str(record["experiment_id"])
            title = str(record["title"])
            label = f"{experiment_id} — {title}" if title else experiment_id
            note = Path(str(record["note"])).as_posix()
            rows.append(
                {
                    "Версия / эксперимент": f"[[{note}|{label}]]",
                    "Метрика": str(record["primary_metric"]),
                    "Значение": f"{float(record['candidate_score']):.4f}",
                    "Δ к baseline": f"{float(record['improvement']):+.4f}",
                    "Решение": str(record["decision"]),
                }
            )
    return dataframe_to_markdown(
        pd.DataFrame(rows),
        right_align={"Значение", "Δ к baseline"},
    )


def build_best_result_block(
    *,
    baseline_metric: str,
    baseline_score: float,
    baseline_std: float,
    baseline_direction: str,
    baseline_note: Path | None = None,
    experiments: pd.DataFrame | None = None,
) -> str:
    """Render the best measured comparable result without promoting baseline."""

    baseline_label = baseline_note.stem if baseline_note is not None else "Baseline"
    baseline_value: dict[str, Any] = {
        "label": (
            f"[[{baseline_note.as_posix()}|{baseline_label}]]"
            if baseline_note is not None
            else baseline_label
        ),
        "hypothesis": "reference",
        "score": float(baseline_score),
        "std": float(baseline_std),
        "delta": 0.0,
        "decision": "reference",
    }
    candidates = [baseline_value]
    if experiments is not None and not experiments.empty:
        required = {
            "title",
            "note",
            "hypothesis",
            "primary_metric",
            "direction",
            "candidate_score",
            "improvement",
            "decision",
        }
        if required.issubset(experiments.columns):
            comparable = experiments[
                experiments["primary_metric"].astype(str).eq(baseline_metric)
                & experiments["direction"].astype(str).eq(baseline_direction)
            ]
            for record in comparable.to_dict(orient="records"):
                note = Path(str(record["note"])).as_posix()
                title = str(record["title"])
                candidate_std = record.get("candidate_std")
                candidates.append(
                    {
                        "label": f"[[{note}|{title}]]",
                        "hypothesis": str(record["hypothesis"]),
                        "score": float(record["candidate_score"]),
                        "std": (
                            float(candidate_std)
                            if candidate_std is not None
                            and not pd.isna(candidate_std)
                            else None
                        ),
                        "delta": float(record["improvement"]),
                        "decision": str(record["decision"]),
                    }
                )
    reverse = baseline_direction == "maximize"
    best = sorted(candidates, key=lambda item: item["score"], reverse=reverse)[0]
    stability = (
        f"{best['score']:.4f} ± {best['std']:.4f}"
        if best["std"] is not None
        else "см. карточку эксперимента"
    )
    report = pd.DataFrame(
        [
            ("Эксперимент", best["label"]),
            ("Гипотеза", best["hypothesis"]),
            ("Метрика", baseline_metric),
            ("Значение", f"{best['score']:.4f}"),
            ("Δ к baseline", f"{best['delta']:+.4f}"),
            ("Проверка стабильности", stability),
            ("Решение", best["decision"]),
        ],
        columns=["Поле", "Значение"],
    )
    return (
        "> [!note] Это лучший **измеренный** сопоставимый результат. "
        "Он не становится новым baseline без ручного решения.\n\n"
        + dataframe_to_markdown(report)
    )


def build_experiment_registry_block(
    baseline_note: Path | None,
    experiments: pd.DataFrame | None,
    *,
    baseline_run: str = "baseline",
    baseline_metric: str | None = None,
    baseline_score: float | None = None,
) -> str:
    """Render a GitHub-compatible list of baseline and controlled experiments."""

    rows: list[dict[str, str]] = []
    if baseline_note is not None:
        rows.append(
            {
                "ID / карточка": f"[[{baseline_note.as_posix()}|{baseline_note.stem}]]",
                "Run": baseline_run,
                "Метрика": baseline_metric or "—",
                "Результат": (
                    f"{baseline_score:.4f}" if baseline_score is not None else "—"
                ),
                "Решение": "reference",
            }
        )
    if experiments is not None and not experiments.empty:
        for record in experiments.to_dict(orient="records"):
            rows.append(
                {
                    "ID / карточка": (
                        f"[[{Path(str(record['note'])).as_posix()}|"
                        f"{record['experiment_id']} — {record['title']}]]"
                    ),
                    "Run": str(record["run_name"]),
                    "Метрика": str(record["primary_metric"]),
                    "Результат": f"{float(record['candidate_score']):.4f}",
                    "Решение": str(record["decision"]),
                }
            )
    if not rows:
        return "> Сохранённых экспериментов пока нет."
    return dataframe_to_markdown(pd.DataFrame(rows), right_align={"Результат"})
