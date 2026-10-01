"""Create an unexecuted, champion-based EPIC experiment workspace."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

try:
    from .experiment_scaffold import slug_from_title
except ImportError:
    from experiment_scaffold import slug_from_title


EXPERIMENT_ID_PATTERN = re.compile(r"EXP-(\d{3,})")


@dataclass(frozen=True)
class Champion:
    experiment_id: str
    title: str
    run_name: str
    notebook: str

    @property
    def run_key(self) -> str:
        return f"{self.experiment_id}/{self.run_name}"


def _markdown(cell_id: str, source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": source.splitlines(keepends=True)}


def _code(cell_id: str, source: str) -> dict[str, object]:
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def _scalar(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*[\"']?([^\n\"']+)[\"']?\s*$", text)
    return match.group(1).strip() if match else None


def find_next_experiment_id(project_root: Path | str) -> str:
    """Find the next free ID across cards, notebooks and candidate modules."""
    root = Path(project_root).resolve()
    numbers = [0]
    for directory, pattern in (
        (root / "experiments", "EXP-*.md"),
        (root / "notebooks/experiments", "EXP-*.ipynb"),
        (root / "src/ml_project/epic_experiments", "exp_*.py"),
    ):
        for path in directory.glob(pattern) if directory.exists() else ():
            match = re.search(r"(?:EXP-|exp_)(\d{3,})", path.name)
            if match:
                numbers.append(int(match.group(1)))
    return f"EXP-{max(numbers) + 1:03d}"


def find_champion(project_root: Path | str, experiment_id: str | None = None) -> Champion:
    """Return the latest adopted experiment that has immutable artifacts."""
    root = Path(project_root).resolve()
    found: list[tuple[int, Champion]] = []
    for card in (root / "experiments").glob("EXP-*.md"):
        text = card.read_text(encoding="utf-8")
        card_id = _scalar(text, "id")
        run_name = _scalar(text, "run_name") or "run_001"
        if not card_id or _scalar(text, "decision") != "adopt":
            continue
        if experiment_id and card_id != experiment_id:
            continue
        match = re.fullmatch(r"EXP-(\d+)", card_id)
        metadata_path = root / "artifacts/experiments" / card_id / run_name / "metadata.json"
        if not match or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run", {}).get("decision") != "adopt":
            continue
        found.append((int(match.group(1)), Champion(
            card_id,
            str(metadata.get("experiment", {}).get("title", card_id)),
            run_name,
            str(metadata.get("run", {}).get("notebook", "")),
        )))
    if not found:
        suffix = f" {experiment_id}" if experiment_id else ""
        raise ValueError(f"No adopted champion{suffix} with saved metadata.json was found")
    return max(found, key=lambda item: item[0])[1]


def feature_module_name(experiment_id: str, slug: str) -> str:
    return f"ml_project.epic_experiments.exp_{experiment_id[4:]}_{slug}"


def build_feature_module(experiment_id: str, title: str, hypothesis: str,
                         changed_variable: str, champion: Champion) -> str:
    """Create the intentionally unfinished, import-safe candidate module."""
    return f'''"""{experiment_id}: {title}.

Only candidate-specific code belongs here. Importing this module must never
count the full dataset or fit a model.
"""
from __future__ import annotations
from typing import Any

EXPERIMENT_ID = {experiment_id!r}
CHAMPION_EXPERIMENT = {champion.experiment_id!r}
CHAMPION_RUN = {champion.run_name!r}
HYPOTHESIS = {hypothesis!r}
CHANGED_VARIABLE = {changed_variable!r}
FEATURE_READY = False

def feature_contract() -> dict[str, Any]:
    return {{
        "champion": f"{{CHAMPION_EXPERIMENT}}/{{CHAMPION_RUN}}",
        "aggregation_key": "TODO",
        "candidate_columns": "TODO",
        "missing_edge_policy": "TODO",
        "leakage_check": "TODO",
    }}

def diagnose_feature(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Implement a cheap train-only pilot together")

def count_candidate(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Implement bounded-memory counting together")

def cross_validate_candidate(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Implement paired train-only CV together")

def fit_candidate(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Implement the final fit together")

def evaluate_candidate(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Implement one-time validation together")
'''


def build_experiment_notebook(experiment_id: str, title: str, hypothesis: str,
                              changed_variable: str, success_criterion: str, *,
                              champion: Champion, implementation: str,
                              module_name: str | None) -> dict[str, object]:
    """Build a source-only notebook with manual STOP gates."""
    slug = slug_from_title(title)
    notebook_filename = f"{experiment_id}_{slug}.ipynb"
    if implementation == "module":
        feature_source = f'''from {module_name} import (
    FEATURE_READY, count_candidate, cross_validate_candidate,
    diagnose_feature, evaluate_candidate, feature_contract, fit_candidate,
)
'''
    else:
        feature_source = '''FEATURE_READY = False
def feature_contract():
    return {"aggregation_key": "TODO", "candidate_columns": "TODO",
            "missing_edge_policy": "TODO", "leakage_check": "TODO"}
def diagnose_feature(*args, **kwargs): raise NotImplementedError
def count_candidate(*args, **kwargs): raise NotImplementedError
def cross_validate_candidate(*args, **kwargs): raise NotImplementedError
def fit_candidate(*args, **kwargs): raise NotImplementedError
def evaluate_candidate(*args, **kwargs): raise NotImplementedError
'''
    cells = [
        _markdown("title", f"""# {experiment_id} — {title}

**Champion:** `{champion.run_key}` — {champion.title}

**Гипотеза:** {hypothesis}

**Единственное основное изменение:** {changed_variable}

**Критерий успеха:** {success_criterion}

Notebook создан пустым и ничего не запускает автоматически.
"""),
        _markdown("roadmap", """## Карта эксперимента

1. Проверить champion и split. 2. Вместе реализовать новый признак.
3. Сделать дешёвый pilot кардинальности и памяти. 4. Полный train count.
5. Paired CV. 6. Full-train fit. 7. Одноразовый validation. 8. Сохранение.
"""),
        _code("environment", '''# Выполнить первой после restart kernel.
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
'''),
        _code("setup", '''from pathlib import Path
import json
import sys
import time
from IPython.display import display
import pandas as pd

CURRENT_DIR = Path.cwd().resolve()
PROJECT_ROOT = next(p for p in (CURRENT_DIR, *CURRENT_DIR.parents)
                    if (p / "README.md").is_file() and (p / "src/ml_project").is_dir())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from ml_project.epic_baseline import load_baseline_sequences
from ml_project.epic_data import ensure_prepared_split
from ml_project.epic_experiment import ExperimentSpec, build_run_record, save_run_record
'''),
        _markdown("config-md", "## 1. Зафиксированный протокол\n"),
        _code("config", f'''RUN_NAME = "run_001"
EXPERIMENT_ID = {experiment_id!r}
CHAMPION_ID = {champion.experiment_id!r}
CHAMPION_RUN = {champion.run_name!r}
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/experiments" / {notebook_filename!r}
ARTIFACT_DIR = PROJECT_ROOT / "artifacts/experiments" / EXPERIMENT_ID / RUN_NAME
CHAMPION_DIR = PROJECT_ROOT / "artifacts/experiments" / CHAMPION_ID / CHAMPION_RUN
SPEC = ExperimentSpec(EXPERIMENT_ID, {title!r}, {hypothesis!r},
                      {changed_variable!r}, {success_criterion!r},
                      primary_metric="average_precision",
                      secondary_metric="epic_spearman", seed=42)
CANDIDATE_PARAMETERS = {{"parent": f"{{CHAMPION_ID}}/{{CHAMPION_RUN}}",
                        "implementation": {implementation!r}}}  # TODO: add feature settings.
assert not (ARTIFACT_DIR / "metadata.json").exists()
display(pd.Series(CANDIDATE_PARAMETERS, name="value").to_frame())
'''),
        _code("champion", '''champion_metadata = json.loads(
    (CHAMPION_DIR / "metadata.json").read_text(encoding="utf-8"))
champion_metrics = pd.read_csv(CHAMPION_DIR / "metrics.csv")
assert champion_metadata["run"]["decision"] == "adopt"
assert champion_metadata["experiment"]["experiment_id"] == CHAMPION_ID
display(champion_metrics)
'''),
        _markdown("data-md", "## 2. Неизменные split и последовательности\n"),
        _code("data", '''split = ensure_prepared_split(PROJECT_ROOT)
display(split.summary)
display(split.contig_assignment.sort_values(["split", "contig"]))
assert split.config.split_version == "contig_holdout_v1" and split.config.seed == SPEC.seed
'''),
        _code("sequences", '''sequences, fasta_alphabet = load_baseline_sequences(split)
display(fasta_alphabet)
print("Loaded contigs:", len(sequences))
'''),
        _markdown("feature-md", f"""## 3. Единственное нововведение

Код размещается в **{implementation}**. До полного подсчёта фиксируем формулу,
strand orientation, missing/edge policy, размерность и ключ агрегации.
"""),
        _code("feature", feature_source + '\ndisplay(pd.Series(feature_contract(), name="value").to_frame())\n'),
        _markdown("pilot-md", """## 4. Дешёвый train-only pilot

Сначала 1–2 млн позиций и один крупный `contig × strand`: число unique rows,
память, время и биологическая sanity-check. Это защита от повторения ISSUE-001.
"""),
        _code("pilot", '''RUN_FEATURE_PILOT = False
if not FEATURE_READY:
    print("Новый признак ещё не реализован — ожидаемая точка совместной работы.")
elif RUN_FEATURE_PILOT:
    pilot = diagnose_feature(split, sequences, parameters=CANDIDATE_PARAMETERS)
    display(pilot)
else:
    print("STOP: сначала проверьте feature contract.")
'''),
        _markdown("stop-count", "# STOP 1 — перед полным подсчётом train\n"),
        _code("count-train", '''RUN_TRAIN_COUNT = False
if RUN_TRAIN_COUNT:
    started = time.monotonic()
    train_candidate_counts = count_candidate(
        split, sequences, "train", parameters=CANDIDATE_PARAMETERS, verbose=True)
    print(f"Train counted in {time.monotonic() - started:.2f} s")
else:
    print("Train count закрыт: RUN_TRAIN_COUNT=False")
'''),
        _markdown("stop-cv", "# STOP 2 — перед paired train-only CV\n"),
        _code("cv", '''RUN_CV = False
if RUN_CV:
    cv_results = cross_validate_candidate(
        train_candidate_counts, champion_dir=CHAMPION_DIR,
        parameters=CANDIDATE_PARAMETERS, verbose=True)
    display(cv_results)
else:
    print("CV закрыт: RUN_CV=False")
'''),
        _markdown("cv-gate", "## 5. Выбор параметров и paired CV gate\n"),
        _code("select-and-gate", '''CV_GATE_PASSED = False  # TODO: вычислять из cv_results.
SELECTED_PARAMETERS = {}       # TODO: зафиксировать train-only выбор.
print("CV gate:", CV_GATE_PASSED)
'''),
        _markdown("stop-fit", "# STOP 3 — перед full-train fit\n"),
        _code("fit", '''RUN_FULL_FIT = False
if RUN_FULL_FIT:
    if not CV_GATE_PASSED:
        raise RuntimeError("Candidate не прошёл paired CV gate")
    final_fit = fit_candidate(train_candidate_counts,
        selected_parameters=SELECTED_PARAMETERS,
        parameters=CANDIDATE_PARAMETERS, verbose=True)
else:
    print("Full fit закрыт: RUN_FULL_FIT=False")
'''),
        _markdown("stop-validation", "# STOP 4 — перед одноразовым validation\n"),
        _code("validation", '''RUN_VALIDATION = False
if RUN_VALIDATION:
    validation_counts = count_candidate(
        split, sequences, "validation", parameters=CANDIDATE_PARAMETERS, verbose=True)
    candidate_evaluation = evaluate_candidate(
        validation_counts, final_fit, parameters=CANDIDATE_PARAMETERS)
else:
    print("Validation закрыт: RUN_VALIDATION=False")
'''),
        _markdown("results-md", "## 6. Метрики, графики и интерпретация\n"),
        _code("results", '''METRICS = {}       # reference/candidate/delta AP и Spearman.
RESULT_TABLES = {}  # "name.csv": DataFrame
PLOT_OBJECTS = {}   # "name.png": Figure
INTERPRETATION = ""
DECISION = "pending"  # adopt/reject/iterate/inconclusive
print("Decision:", DECISION)
'''),
        _markdown("save-md", "## 7. Сохранение проведённого эксперимента\n"),
        _code("save", '''SAVE_RUN = False
if SAVE_RUN:
    if DECISION == "pending" or not INTERPRETATION.strip() or not METRICS:
        raise ValueError("Заполните METRICS, INTERPRETATION и DECISION")
    if (ARTIFACT_DIR / "metadata.json").exists():
        raise FileExistsError(f"Immutable run exists: {ARTIFACT_DIR}")
    record = build_run_record(SPEC, split, run_name=RUN_NAME,
        notebook_path=NOTEBOOK_PATH,
        parameters=CANDIDATE_PARAMETERS | SELECTED_PARAMETERS,
        metrics=METRICS, interpretation=INTERPRETATION, decision=DECISION)
    output = save_run_record(PROJECT_ROOT, record)
    for filename, table in RESULT_TABLES.items():
        table.to_csv(output / filename, index=False)
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    for filename, figure in PLOT_OBJECTS.items():
        figure.savefig(plot_dir / filename, dpi=170, bbox_inches="tight", facecolor="white")
    print("Saved run:", output)
else:
    print("SAVE_RUN=False: nothing was written.")
'''),
        _markdown("report-md", """## 8. Отчёт в Obsidian

После сохранения добавляем report builder, который читает immutable artifacts
и обновляет только автоматический блок карточки. Ручные выводы сохраняются.
"""),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python (epic-eda)", "language": "python", "name": "epic-eda"},
        "language_info": {"name": "python", "version": "3.12"},
        "epic": {"experiment_id": experiment_id, "status": "scaffold_unexecuted",
                 "champion_experiment": champion.experiment_id,
                 "champion_run": champion.run_name, "implementation": implementation,
                 "data_contract": "ml_project.epic_data"}},
        "nbformat": 4, "nbformat_minor": 5}


def build_experiment_card(experiment_id: str, title: str, hypothesis: str,
                          changed_variable: str, success_criterion: str,
                          notebook_relative: str, *, champion: Champion,
                          implementation: str, module_relative: str | None) -> str:
    code = f"[[{module_relative}]]" if module_relative else "ячейка `feature` notebook"
    return f"""---
id: {experiment_id}
type: experiment
experiment_type: hypothesis-test
status: planned
decision: pending
date:
seed: 42
split_version: contig_holdout_v1
run_name: run_001
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: {champion.experiment_id}
reference_run: {champion.run_name}
implementation: {implementation}
---

# {experiment_id} — {title}

← [[experiments/_index.md|Реестр]] · [[experiments/{champion.experiment_id}.md|Champion {champion.experiment_id}]]

## До запуска

- **Гипотеза:** {hypothesis}
- **Единственное изменение:** {changed_variable}
- **Замороженный champion:** `{champion.run_key}` — {champion.title}.
- **Критерий успеха:** {success_criterion}
- **Метрики:** Average Precision и EPIC Spearman.
- **Notebook:** [[{notebook_relative}]]
- **Код нового признака:** {code}

## Автоматический отчёт

<!-- auto:experiment-report:start -->

> Эксперимент создан, но ещё не выполнен.

<!-- auto:experiment-report:end -->

## Ручной анализ результата

- **Train-only диагностика:** ожидает реализации.
- **Paired CV gate:** ожидает запуска.
- **Validation:** закрыт до прохождения gate.
- **Вклад нового признака:** ожидает запуска.

## Решение

`pending` — результат сохраняется при любом содержательном исходе.
"""


def create_experiment(project_root: Path | str, *, experiment_id: str, title: str,
                      hypothesis: str, changed_variable: str,
                      success_criterion: str, champion: Champion,
                      implementation: str = "module") -> tuple[Path, Path, Path | None]:
    """Create card, notebook and optional candidate module without overwriting."""
    root = Path(project_root).resolve()
    experiment_id = experiment_id.upper()
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("experiment_id must look like EXP-010")
    if implementation not in {"module", "notebook"}:
        raise ValueError("implementation must be module or notebook")
    if not all(value.strip() for value in (title, hypothesis, changed_variable, success_criterion)):
        raise ValueError("title, hypothesis, change and success criterion are required")
    slug = slug_from_title(title)
    notebook = root / "notebooks/experiments" / f"{experiment_id}_{slug}.ipynb"
    card = root / "experiments" / f"{experiment_id}.md"
    module_name = feature_module_name(experiment_id, slug) if implementation == "module" else None
    module_path = root / "src" / (module_name.replace(".", "/") + ".py") if module_name else None
    targets = [notebook, card] + ([module_path] if module_path else [])
    for path in targets:
        if path.exists():
            raise FileExistsError(f"File already exists: {path}")
    notebook.parent.mkdir(parents=True, exist_ok=True)
    card.parent.mkdir(parents=True, exist_ok=True)
    if module_path:
        module_path.parent.mkdir(parents=True, exist_ok=True)
        init_path = module_path.parent / "__init__.py"
        if not init_path.exists():
            init_path.write_text('"""Candidate-specific EPIC modules."""\n', encoding="utf-8")
        module_path.write_text(build_feature_module(
            experiment_id, title, hypothesis, changed_variable, champion), encoding="utf-8")
    notebook.write_text(json.dumps(build_experiment_notebook(
        experiment_id, title, hypothesis, changed_variable, success_criterion,
        champion=champion, implementation=implementation, module_name=module_name),
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    notebook_relative = notebook.relative_to(root).as_posix()
    module_relative = module_path.relative_to(root).as_posix() if module_path else None
    card.write_text(build_experiment_card(
        experiment_id, title, hypothesis, changed_variable, success_criterion,
        notebook_relative, champion=champion, implementation=implementation,
        module_relative=module_relative), encoding="utf-8")
    return notebook, card, module_path


def _prompt(label: str, default: str | None = None) -> str:
    value = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return value or (default or "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-id")
    parser.add_argument("--title")
    parser.add_argument("--hypothesis")
    parser.add_argument("--change", dest="changed_variable")
    parser.add_argument("--success", dest="success_criterion")
    parser.add_argument("--parent", help="Adopted champion ID; default: latest")
    parser.add_argument("--implementation", choices=("module", "notebook"))
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    experiment_id = (args.experiment_id or find_next_experiment_id(root)).upper()
    champion = find_champion(root, args.parent)
    prompted = any(value is None for value in (
        args.title, args.hypothesis, args.changed_variable,
        args.success_criterion, args.implementation))
    title = args.title or _prompt("Short title")
    hypothesis = args.hypothesis or _prompt("Hypothesis")
    changed_variable = args.changed_variable or _prompt("Exactly one change")
    success_criterion = args.success_criterion or _prompt("Success criterion and guardrails")
    implementation = args.implementation or _prompt(
        "Candidate code location: module/notebook", "module").casefold()
    if implementation not in {"module", "notebook"}:
        parser.error("Implementation must be module or notebook")
    if prompted and not args.yes:
        print(f"\n{experiment_id} from {champion.run_key}: {title}")
        print(f"One change: {changed_variable}\nImplementation: {implementation}")
        if input("Create these files? [y/N]: ").strip().casefold() not in {"y", "yes", "д", "да"}:
            print("Cancelled; no files were created.")
            return 0
    notebook, card, module_path = create_experiment(
        root, experiment_id=experiment_id, title=title, hypothesis=hypothesis,
        changed_variable=changed_variable, success_criterion=success_criterion,
        champion=champion, implementation=implementation)
    print(f"Created notebook: {notebook}")
    print(f"Created card:     {card}")
    if module_path:
        print(f"Created module:   {module_path}")
    print("Nothing was executed. Work through the notebook manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
