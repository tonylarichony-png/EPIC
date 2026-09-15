"""Generate a local notebook-first workbench for one experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path


EXPERIMENT_ID_PATTERN = re.compile(r"EXP-\d{3,}")
SLUG_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


def _markdown(cell_id: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code(cell_id: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def workbench_path(
    project_root: Path,
    experiment_id: str,
    slug: str,
) -> Path:
    """Return the local ignored notebook path for one experiment."""

    normalized_id = experiment_id.upper()
    normalized_slug = slug.lower().replace("-", "_")
    if not EXPERIMENT_ID_PATTERN.fullmatch(normalized_id):
        raise ValueError("experiment_id must look like EXP-003")
    if not SLUG_PATTERN.fullmatch(normalized_slug):
        raise ValueError("slug must contain lowercase letters, digits, underscores")
    return (
        Path(project_root).resolve()
        / "notebooks/workbench"
        / f"{normalized_id}_{normalized_slug}.ipynb"
    )


def build_workbench_notebook(
    experiment_id: str,
    title: str,
    slug: str,
    module_name: str,
    parent_experiment_module: str | None = None,
) -> dict[str, object]:
    """Build a source-only notebook that never writes official experiment results."""

    cells = [
        _markdown(
            "workbench-top",
            f"""# {experiment_id} — Workbench: {title}

Это локальная лаборатория для notebook-first разработки эксперимента
`{module_name}`.

> [!warning]
> Workbench не является официальным запуском: он не обновляет карточку,
> `experiments/results.csv`, README и Git-отслеживаемые графики. После
> проверки перенесите реализацию в Python-модуль и выполните
> `notebooks/04_experiment.ipynb`.

Рабочий цикл: **draft → проверки → перенос в модуль → строгий runner**.

Reference: `{parent_experiment_module or "baseline_config.BASELINE"}`.
""",
        ),
        _markdown(
            "workbench-setup-md",
            """## 1. Безопасно загрузить данные

Эта стадия намеренно не импортирует и не валидирует незавершённый
experiment-модуль. Поэтому синтаксически корректный черновик модели не нужен,
чтобы начать исследование данных.

Fold-safe преобразования родителя здесь не материализуются: например, `Age`
может оставаться с пропусками в DataFrame и будет заполнен внутри pipeline
отдельно на каждом `fit`. Проверка этого поведения находится в smoke-fit.
""",
        ),
        _code(
            "workbench-setup",
            f"""from dataclasses import replace
from pathlib import Path
import copy
import importlib
import sys
from typing import Any, Mapping

from IPython.display import display
import numpy as np
import pandas as pd

CURRENT_DIR = Path.cwd().resolve()
PROJECT_ROOT = next(
    candidate for candidate in (CURRENT_DIR, *CURRENT_DIR.parents)
    if (candidate / "README.md").exists() and (candidate / "src").exists()
)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_project import DataCatalog
import ml_project.config as project_config
import ml_project.baseline_config as baseline_config
import ml_project.modeling as modeling_tools
import ml_project.experiment as experiment_tools
from ml_project.modeling import ExperimentData, ModelingSettings

EXPERIMENT_MODULE = {module_name!r}
PARENT_EXPERIMENT_MODULE = {parent_experiment_module!r}
FEATURE_GROUPS = project_config.FEATURE_GROUPS
KEY = project_config.KEY
TARGET = project_config.TARGET

catalog = DataCatalog(
    PROJECT_ROOT,
    project_config.RAW_DIR,
    project_config.DATASETS,
)
catalog.validate()
train = catalog.load(project_config.TRAIN_DATASET)
train_snapshot = train.copy(deep=True)

# initial_settings — исходная точка цепочки: ModelingSettings из EXP-001.
# n_jobs=1 влияет только на локальный workbench и делает dry-run стабильнее.
initial_settings = replace(baseline_config.BASELINE, n_jobs=1)
parent_data = experiment_tools.prepare_reference_experiment_data(
    PARENT_EXPERIMENT_MODULE,
    train,
    FEATURE_GROUPS,
    initial_settings,
)
# reference_settings — эффективные настройки сравниваемой модели после
# применения всех принятых parent-экспериментов.
reference_settings = parent_data.settings
parent_snapshot = parent_data.frame.copy(deep=True)

print("Project:", PROJECT_ROOT)
print("Module:", EXPERIMENT_MODULE)
print("Parent:", PARENT_EXPERIMENT_MODULE or "baseline")
print("Train shape:", train.shape)
display(parent_data.frame.head())
""",
        ),
        _markdown(
            "workbench-draft-md",
            """## 2. Подготовить candidate data

Редактируйте функцию прямо в этой ячейке. Её сигнатура, локальные переменные и
возвращаемый `ExperimentData` совпадают с `prepare_candidate_data(...)` в
Python-модуле эксперимента. После проверки функцию можно перенести целиком без
переименования `result`/`frame` и `draft_feature_groups`/`groups`.

Функция не должна изменять входной `train`. Детерминированные признаки можно
создавать здесь; статистики, которые обучаются на данных, позднее должны стать
sklearn-transformer внутри candidate Pipeline.
""",
        ),
        _code(
            "workbench-draft",
            """def prepare_candidate_data(
    train: pd.DataFrame,
    feature_groups: Mapping[str, Any],
    reference_settings: ModelingSettings,
) -> ExperimentData:
    \"\"\"Подготовить candidate поверх настроек baseline/чемпиона.\"\"\"

    frame = train.copy(deep=True)
    groups = copy.deepcopy(feature_groups)
    candidate_settings = reference_settings

    # EDIT HERE — создайте или измените экспериментальные признаки.
    # Пример:
    # frame["FamilySize"] = frame["SibSp"] + frame["Parch"] + 1

    # EDIT HERE — зарегистрируйте каждый новый model-ready признак ровно
    # в одной группе:
    # groups["count"] = [*groups.get("count", []), "FamilySize"]

    # EDIT HERE — если меняются настройки относительно reference:
    # candidate_settings = replace(
    #     reference_settings,
    #     exclude_features=(*reference_settings.exclude_features, "OldFeature"),
    # )

    return ExperimentData(
        frame=frame,
        feature_groups=groups,
        settings=candidate_settings,
        diagnostics={},
    )


candidate_data = prepare_candidate_data(
    parent_data.frame,
    parent_data.feature_groups,
    reference_settings,
)
draft_frame = candidate_data.frame
groups = candidate_data.feature_groups
candidate_settings = candidate_data.settings
display(draft_frame.head())
""",
        ),
        _markdown(
            "workbench-contract-md",
            """## 3. Проверить data contract

Проверки защищают исходный train, target, индекс и показывают все реально
изменённые столбцы. Пропуски отображаются как диагностика: решение о том,
допустимы ли они, остаётся частью гипотезы.
""",
        ),
        _code(
            "workbench-contract",
            """if not isinstance(candidate_data, ExperimentData):
    raise TypeError("prepare_candidate_data must return ExperimentData")
if not isinstance(draft_frame, pd.DataFrame):
    raise TypeError("ExperimentData.frame must be a pandas DataFrame")

pd.testing.assert_frame_equal(train, train_snapshot)
pd.testing.assert_frame_equal(parent_data.frame, parent_snapshot)
if not draft_frame.index.equals(parent_data.frame.index):
    raise ValueError("Draft changed row index or row order")
if TARGET not in draft_frame:
    raise KeyError(f"Draft removed target column {TARGET!r}")
pd.testing.assert_series_equal(
    draft_frame[TARGET],
    parent_data.frame[TARGET],
    check_names=True,
)

new_columns = [
    column for column in draft_frame
    if column not in parent_data.frame.columns
]
removed_columns = [
    column for column in parent_data.frame
    if column not in draft_frame.columns
]
changed_columns = [
    column
    for column in parent_data.frame.columns.intersection(draft_frame.columns)
    if not parent_data.frame[column].equals(draft_frame[column])
]
if removed_columns:
    raise ValueError(f"Draft removed source columns: {removed_columns}")

diagnostics = pd.DataFrame(
    [
        {
            "column": column,
            "dtype": str(draft_frame[column].dtype),
            "missing": int(draft_frame[column].isna().sum()),
            "unique": int(draft_frame[column].nunique(dropna=False)),
        }
        for column in new_columns + changed_columns
    ]
)
print("New columns:", new_columns or "none")
print("Changed source columns:", changed_columns or "none")
display(diagnostics)
""",
        ),
        _markdown(
            "workbench-groups-md",
            """## 4. Настроить feature groups

Эта ячейка проверяет `groups`, которые вернула `prepare_candidate_data(...)`.
Добавление признаков в группы и изменение `candidate_settings` выполняются в
самой функции выше, чтобы её можно было перенести в Python-модуль целиком.
""",
        ),
        _code(
            "workbench-groups",
            """memberships = {
    column: [
        group
        for group, columns in groups.items()
        if column in columns
    ]
    for column in new_columns
}
invalid_memberships = {
    column: matched_groups
    for column, matched_groups in memberships.items()
    if len(matched_groups) != 1
}
if invalid_memberships:
    raise ValueError(
        "Every new column must belong to exactly one feature group: "
        f"{invalid_memberships}"
    )

candidate_plan = modeling_tools.resolve_feature_plan(
    draft_frame,
    groups,
    target=TARGET,
    key=KEY,
    settings=candidate_settings,
)
display(candidate_plan.to_frame())
""",
        ),
        _markdown(
            "workbench-model-md",
            """## 5. Собрать и быстро обучить candidate

Default-функция сохраняет точную pipeline принятого родителя, а при отсутствии
родителя использует baseline. Новый candidate preprocessor получает только
draft-изменение. Это smoke-проверка API, а не официальный результат.
""",
        ),
        _code(
            "workbench-model",
            """def draft_build_candidate_models(preprocessor, settings):
    candidate = experiment_tools.build_reference_pipeline(
        PARENT_EXPERIMENT_MODULE,
        preprocessor,
        settings,
    )
    return {"candidate": candidate}


draft_preprocessor = modeling_tools.build_tabular_preprocessor(
    candidate_settings,
    candidate_plan,
)
reference_plan = modeling_tools.resolve_feature_plan(
    parent_data.frame,
    parent_data.feature_groups,
    target=TARGET,
    key=KEY,
    settings=reference_settings,
)
reference_training = modeling_tools.prepare_training_data(
    parent_data.frame,
    target=TARGET,
    plan=reference_plan,
    settings=reference_settings,
)
draft_data = experiment_tools.prepare_experiment_data(
    reference_training,
    draft_frame,
    target=TARGET,
)
reference_preprocessor = modeling_tools.build_tabular_preprocessor(
    reference_settings,
    reference_plan,
)
reference_model = experiment_tools.build_reference_pipeline(
    PARENT_EXPERIMENT_MODULE,
    reference_preprocessor,
    reference_settings,
)
draft_models = draft_build_candidate_models(
    draft_preprocessor,
    candidate_settings,
)
workbench_models = {
    "workbench_reference": reference_model,
    **draft_models,
}
for model_name, model in workbench_models.items():
    model.fit(draft_data.X, draft_data.y)
    print(model_name, "fit: OK")
""",
        ),
        _markdown(
            "workbench-cv-md",
            """## 6. Необязательный короткий dry-run CV

Включите только после smoke-fit. Результат не сохраняется и не считается
официальным: он нужен для раннего обнаружения ошибок pipeline.
""",
        ),
        _code(
            "workbench-cv",
            """RUN_DRY_CV = False

if RUN_DRY_CV:
    dry_settings = replace(
        candidate_settings,
        n_splits=min(3, candidate_settings.n_splits),
        n_jobs=1,
        save_artifacts=False,
        save_metric_figures=False,
        save_final_model=False,
        sync_docs=False,
        sync_experiment_note=False,
    )
    scoring = modeling_tools.resolve_scoring_plan(PROJECT_ROOT, dry_settings)
    dry_cv, _ = modeling_tools.build_cv_splitter(dry_settings, draft_data.y)
    dry_evaluation = modeling_tools.evaluate_models_cv(
        workbench_models,
        draft_data,
        cv=dry_cv,
        scoring=scoring,
        settings=dry_settings,
    )
    display(dry_evaluation.summary.round(4))
else:
    print("Dry-run CV выключен.")
""",
        ),
        _markdown(
            "workbench-promote-md",
            f"""## 7. Перенести проверенную реализацию

Официальный source of truth:
`src/ml_project/experiments/{experiment_id.lower().replace("-", "_")}_{slug}.py`.

1. Замените заготовку `prepare_candidate_data(...)` в модуле одноимённой
   функцией из раздела 2 — её можно перенести целиком без адаптации имён.
2. Перенесите `draft_build_candidate_models(...)` в
   `build_candidate_models(...)`.
3. Удалите `NotImplementedError` и оставшиеся `CHANGE ME`.
4. Запустите следующую smoke-ячейку.

Не импортируйте код из этого notebook: workbench локальный и намеренно
игнорируется Git.
""",
        ),
        _code(
            "workbench-module-smoke",
            """RUN_MODULE_SMOKE = False

if RUN_MODULE_SMOKE:
    import ml_project.experiment as experiment_tools

    experiment_tools = importlib.reload(experiment_tools)
    definition = experiment_tools.load_experiment(EXPERIMENT_MODULE)
    experiment_tools.validate_settings(definition.settings)
    module_parent = experiment_tools.prepare_experiment_parent(
        definition,
        train,
        FEATURE_GROUPS,
        initial_settings,
    )
    module_reference_plan = modeling_tools.resolve_feature_plan(
        module_parent.frame,
        module_parent.feature_groups,
        target=TARGET,
        key=KEY,
        settings=module_parent.settings,
    )
    module_reference_training = modeling_tools.prepare_training_data(
        module_parent.frame,
        target=TARGET,
        plan=module_reference_plan,
        settings=module_parent.settings,
    )
    module_data = experiment_tools.prepare_experiment_candidate(
        definition,
        train,
        FEATURE_GROUPS,
        initial_settings,
    )
    module_plan = modeling_tools.resolve_feature_plan(
        module_data.frame,
        module_data.feature_groups,
        target=TARGET,
        key=KEY,
        settings=module_data.settings,
    )
    module_preprocessor = modeling_tools.build_tabular_preprocessor(
        module_data.settings,
        module_plan,
    )
    module_models = experiment_tools.build_experiment_candidates(
        definition,
        module_preprocessor,
        module_data.settings,
    )
    module_training = experiment_tools.prepare_experiment_data(
        module_reference_training,
        module_data.frame,
        target=TARGET,
    )
    for model_name, model in module_models.items():
        model.fit(module_training.X, module_training.y)
        print(model_name, "module smoke-fit: OK")
else:
    print("Включите RUN_MODULE_SMOKE после переноса кода в модуль.")
""",
        ),
        _markdown(
            "workbench-finish",
            """## 8. Официальный запуск

После успешного module smoke откройте
[универсальный runner](../04_experiment.ipynb), выполните
`Restart Kernel → Run All` и интерпретируйте автоматически созданную карточку.

[Короткая инструкция в README](../../README.md#как-начать-новый-эксперимент)
""",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def create_experiment_workbench(
    project_root: Path,
    experiment_id: str,
    title: str,
    *,
    slug: str,
    module_name: str,
    parent_experiment_module: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a local source-only workbench without touching official reports."""

    path = workbench_path(project_root, experiment_id, slug)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Experiment workbench already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_workbench_notebook(
        experiment_id.upper(),
        title.strip(),
        slug.lower().replace("-", "_"),
        module_name,
        parent_experiment_module,
    )
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "build_workbench_notebook",
    "create_experiment_workbench",
    "workbench_path",
]
