"""Create a clean CNN experiment notebook and an Obsidian preregistration card."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
import unicodedata


CNN_ID_PATTERN = re.compile(r"CNN-EXP-(\d{3})$")


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return slug or "experiment"


def _next_experiment_id(root: Path) -> str:
    numbers: list[int] = []
    for path in (root / "cnn_exp").glob("CNN-EXP-*.md"):
        match = CNN_ID_PATTERN.fullmatch(path.stem)
        if match:
            numbers.append(int(match.group(1)))
    return f"CNN-EXP-{max(numbers, default=0) + 1:03d}"


def _load_champion(root: Path) -> dict[str, str]:
    path = root / "cnn_exp/champion.json"
    if not path.is_file():
        raise FileNotFoundError(f"CNN champion registry not found: {path}")
    champion = json.loads(path.read_text(encoding="utf-8"))
    for field in ("experiment_id", "notebook", "metrics", "checkpoint"):
        if not champion.get(field):
            raise ValueError(f"CNN champion registry misses {field!r}")
    return champion


def _source_cell(cell_id: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _finalization_cell(
    experiment_id: str,
    title: str,
    hypothesis: str,
    notebook_relative: str,
    reference_id: str,
    reference_metrics: str,
    artifact_relative: str,
) -> dict[str, object]:
    source = f'''# Запускай эту ячейку только после обучения и полной validation.
SAVE_EXPERIMENT = False
FINAL_DECISION = "review"  # review / adopt / reject / iterate
FINAL_NOTES = ""

if SAVE_EXPERIMENT:
    from ml_project.cnn_experiment_report import finalize_cnn_experiment

    finalization = finalize_cnn_experiment(
        project_root=PROJECT_ROOT,
        experiment_id={experiment_id!r},
        title={title!r},
        hypothesis={hypothesis!r},
        notebook_path={notebook_relative!r},
        artifact_dir=PROJECT_ROOT / {artifact_relative!r},
        training_history=history_path,
        candidate_metrics=VALIDATION_METRICS_PATH,
        reference_experiment={reference_id!r},
        reference_metrics={reference_metrics!r},
        decision=FINAL_DECISION,
        notes=FINAL_NOTES,
    )
    print("Obsidian:", finalization.card_path)
    print("Summary:", finalization.summary_path)
    print("Plots:", finalization.plot_dir)
else:
    print("Сохранение выключено. Установи SAVE_EXPERIMENT = True после validation.")'''
    return _source_cell("finalize-experiment", source)


def _card(
    experiment_id: str,
    title: str,
    hypothesis: str,
    notebook_relative: str,
    reference_id: str,
) -> str:
    return f'''---
id: {experiment_id}
type: experiment
experiment_type: cnn-candidate
status: planned
decision: pending
date: {date.today().isoformat()}
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: {reference_id}
tags:
  - ml/experiment
  - ml/cnn
---

# {experiment_id} — {title}

← [[cnn_exp/_index.md|Реестр CNN]] · [[cnn_exp/{reference_id}.md|Текущий champion]]

## Pre-registration

- **Гипотеза:** {hypothesis}
- **Единственное изменение:** TODO — заполнить до запуска.
- **Что заморожено:** split, generator, targets, loss, optimizer, validation protocol.
- **Reference:** [[cnn_exp/{reference_id}.md|{reference_id}]].
- **Notebook:** [[{notebook_relative}]].
- **Критерий:** AP выше reference, EPIC Spearman не ниже reference; проверить 3/3 contigs.

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

Эксперимент ещё не сохранён финальной ячейкой notebook.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

-
'''


def create_cnn_experiment(
    root: Path,
    *,
    title: str,
    hypothesis: str,
) -> tuple[Path, Path, Path]:
    experiment_id = _next_experiment_id(root)
    champion = _load_champion(root)
    reference_id = champion["experiment_id"]
    reference_notebook = root / champion["notebook"]
    if not reference_notebook.is_file():
        raise FileNotFoundError(f"Champion notebook not found: {reference_notebook}")

    number = experiment_id.rsplit("-", 1)[1]
    token = f"CNN_EXP{number}"
    slug = _slug(title)
    notebook = (
        root
        / "notebooks/experiments/CNN_experiments"
        / f"{token}_{slug}.ipynb"
    )
    card = root / "cnn_exp" / f"{experiment_id}.md"
    artifact_dir = root / "artifacts/experiments" / token / "run_001"
    for target in (notebook, card):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")

    template = json.loads(reference_notebook.read_text(encoding="utf-8"))
    reference_number = reference_id.rsplit("-", 1)[1]
    reference_token = f"CNN_EXP{reference_number}"
    replacements = {
        reference_id: experiment_id,
        reference_token: token,
        f"EXP{reference_number}": f"EXP{number}",
        "presence_intensity_v1": "run_001",
    }
    for cell in template["cells"]:
        source = "".join(cell.get("source", []))
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook_relative = notebook.relative_to(root).as_posix()
    artifact_relative = artifact_dir.relative_to(root).as_posix()
    if template["cells"] and template["cells"][0].get("cell_type") == "markdown":
        first_source = "".join(template["cells"][0].get("source", []))
        first_lines = first_source.splitlines()
        if first_lines:
            first_lines[0] = f"# {experiment_id} — {title}"
            template["cells"][0]["source"] = [
                line + "\n" for line in first_lines[:-1]
            ] + ([first_lines[-1]] if first_lines else [])
    template["cells"].append(
        _finalization_cell(
            experiment_id,
            title,
            hypothesis,
            notebook_relative,
            reference_id,
            champion["metrics"],
            artifact_relative,
        )
    )
    template.setdefault("metadata", {})["cnn_experiment"] = {
        "experiment_id": experiment_id,
        "reference_experiment": reference_id,
        "status": "scaffold_unexecuted",
    }

    notebook.parent.mkdir(parents=True, exist_ok=True)
    card.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    notebook.write_text(
        json.dumps(template, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    card.write_text(
        _card(
            experiment_id,
            title,
            hypothesis,
            notebook_relative,
            reference_id,
        ),
        encoding="utf-8",
    )
    return notebook, card, artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--title")
    parser.add_argument("--hypothesis")
    args = parser.parse_args()
    title = args.title or input("CNN experiment title: ").strip()
    hypothesis = args.hypothesis or input("Hypothesis: ").strip()
    if not title or not hypothesis:
        parser.error("title and hypothesis must not be empty")
    notebook, card, artifact_dir = create_cnn_experiment(
        Path(args.project_root).resolve(),
        title=title,
        hypothesis=hypothesis,
    )
    print(f"Created notebook: {notebook}")
    print(f"Created Obsidian card: {card}")
    print(f"Created artifact directory: {artifact_dir}")
    print("Nothing was executed. Edit the architecture/config cells first.")


if __name__ == "__main__":
    main()
