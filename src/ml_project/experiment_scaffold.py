"""Create and select a versioned experiment module."""

from __future__ import annotations

import argparse
import ast
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Mapping

try:
    from .experiment_workbench import (
        create_experiment_workbench,
        workbench_path,
    )
except ImportError:  # Direct execution from src/ml_project.
    from experiment_workbench import (
        create_experiment_workbench,
        workbench_path,
    )


EXPERIMENT_ID_PATTERN = re.compile(r"EXP-\d{3,}")
SLUG_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
SELECTOR_PATTERN = re.compile(
    r'(?m)^EXPERIMENT_MODULE\s*=\s*["\']([^"\']+)["\']\s*$'
)
CARD_ID_PATTERN = re.compile(r'(?m)^id:\s*["\']?(EXP-\d+)["\']?\s*$')
CARD_DECISION_PATTERN = re.compile(
    r'(?m)^decision:\s*["\']?(pending|adopt|reject|iterate|inconclusive)["\']?\s*$'
)
CYRILLIC_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def find_next_experiment_id(project_root: Path) -> str:
    """Return the next free EXP ID declared by versioned experiment modules."""

    package_dir = Path(project_root).resolve() / "src/ml_project/experiments"
    numbers = [1]
    if package_dir.exists():
        for module_path in package_dir.glob("exp_*.py"):
            source = module_path.read_text(encoding="utf-8")
            match = re.search(
                r'''(?m)^\s*experiment_id\s*=\s*["']EXP-(\d+)["']''',
                source,
            )
            if match:
                numbers.append(int(match.group(1)))
    return f"EXP-{max(numbers) + 1:03d}"


def slug_from_title(title: str) -> str:
    """Build a readable Python slug from Latin or Russian text."""

    transliterated = "".join(
        CYRILLIC_TO_LATIN.get(character, character)
        for character in title.casefold()
    )
    slug = re.sub(r"[^a-z0-9]+", "_", transliterated).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"experiment_{slug}".rstrip("_")
    return slug


def parse_guardrails(value: str) -> dict[str, float]:
    """Parse ``Metric=min_delta`` pairs separated by commas."""

    if not value.strip():
        return {}
    result: dict[str, float] = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError(
                "Guardrails must look like Recall=-0.005, F1=-0.002"
            )
        metric, raw_threshold = item.rsplit("=", 1)
        metric = metric.strip()
        if not metric:
            raise ValueError("Guardrail metric cannot be empty")
        try:
            threshold = float(raw_threshold.strip())
        except ValueError as error:
            raise ValueError(
                f"Guardrail threshold for {metric!r} must be a number"
            ) from error
        if not math.isfinite(threshold):
            raise ValueError(
                f"Guardrail threshold for {metric!r} must be finite"
            )
        if metric in result:
            raise ValueError(f"Duplicate guardrail metric: {metric!r}")
        result[metric] = threshold
    return result


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def _confirmed() -> bool:
    answer = input("Создать эксперимент? [y/N]: ").strip().casefold()
    return answer in {"y", "yes", "д", "да"}


def _experiment_literals(
    module_path: Path,
    names: set[str],
) -> dict[str, object]:
    tree = ast.parse(
        module_path.read_text(encoding="utf-8"),
        filename=str(module_path),
    )
    experiment_call = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "EXPERIMENT"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
        ):
            experiment_call = node.value
            break
    if experiment_call is None:
        raise ValueError(f"Cannot find EXPERIMENT assignment in {module_path}")
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in experiment_call.keywords
        if keyword.arg in names
    }


def _module_path(project_root: Path, module_name: str) -> Path:
    return (
        Path(project_root).resolve()
        / "src"
        / (module_name.replace(".", "/") + ".py")
    )


def _card_decisions(project_root: Path) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for path in sorted((Path(project_root) / "experiments").glob("EXP-*.md")):
        text = path.read_text(encoding="utf-8")
        id_match = CARD_ID_PATTERN.search(text)
        decision_match = CARD_DECISION_PATTERN.search(text)
        if id_match and decision_match:
            decisions[id_match.group(1)] = decision_match.group(1)
    return decisions


def find_adopted_champion_module(project_root: Path) -> str | None:
    """Return the latest module adopted in its Markdown card."""

    root = Path(project_root).resolve()
    package = root / "src/ml_project/experiments"
    adopted: list[tuple[int, str]] = []
    decisions = _card_decisions(root)
    if not package.exists():
        return None
    for module_path in package.glob("exp_*.py"):
        values = _experiment_literals(
            module_path,
            {"experiment_id"},
        )
        experiment_id = str(values.get("experiment_id", ""))
        match = re.fullmatch(r"EXP-(\d+)", experiment_id)
        if match and decisions.get(experiment_id) == "adopt":
            adopted.append(
                (
                    int(match.group(1)),
                    f"ml_project.experiments.{module_path.stem}",
                )
            )
    return max(adopted)[1] if adopted else None


def validate_parent_module(project_root: Path, module_name: str) -> None:
    """Require an existing adopted module before using it as champion."""

    module_path = _module_path(project_root, module_name)
    if not module_path.exists():
        raise FileNotFoundError(f"Parent experiment module not found: {module_path}")
    values = _experiment_literals(
        module_path,
        {"experiment_id"},
    )
    experiment_id = str(values.get("experiment_id", module_name))
    if _card_decisions(project_root).get(experiment_id) != "adopt":
        raise ValueError(
            f"Parent {experiment_id} must have decision: adopt in its "
            "Markdown experiment card"
        )


def selected_experiment_spec(
    project_root: Path,
) -> tuple[str, str, str, str, str | None]:
    """Read selected metadata without importing unfinished experiment code."""

    root = Path(project_root).resolve()
    selector_path = root / "src/ml_project/experiment_config.py"
    selector = selector_path.read_text(encoding="utf-8")
    match = SELECTOR_PATTERN.search(selector)
    if not match:
        raise ValueError(
            f"Cannot find EXPERIMENT_MODULE assignment in {selector_path}"
        )
    module_name = match.group(1)
    module_path = _module_path(root, module_name)
    if not module_path.exists():
        raise FileNotFoundError(f"Selected experiment module not found: {module_path}")

    values = _experiment_literals(
        module_path,
        {
            "experiment_id",
            "experiment_title",
            "parent_experiment_module",
        },
    )
    experiment_id = str(values.get("experiment_id", ""))
    title = str(values.get("experiment_title", ""))
    stem_match = re.fullmatch(r"exp_\d+_(.+)", module_path.stem)
    if not experiment_id or not title or not stem_match:
        raise ValueError(
            f"Cannot resolve experiment ID, title or slug from {module_path}"
        )
    parent = values.get("parent_experiment_module")
    return (
        experiment_id,
        title,
        stem_match.group(1),
        module_name,
        str(parent) if parent else None,
    )


def _module_source(
    experiment_id: str,
    title: str,
    slug: str,
    primary_improvement_min: float,
    metric_guardrails: Mapping[str, float],
    parent_experiment_module: str | None,
) -> str:
    module_run = experiment_id.lower().replace("-", "_")
    note_title = slug.replace("_", " ").title()
    reference_model = (
        "champion_reference"
        if parent_experiment_module is not None
        else "baseline_reference"
    )
    return textwrap.dedent(
        f'''\
        """{experiment_id}: {title}."""

        from __future__ import annotations

        import copy
        from dataclasses import replace
        from pathlib import Path
        from typing import Any, Mapping

        import pandas as pd

        from ml_project.experiment import build_reference_pipeline
        from ml_project.modeling import (
            ModelingSettings,
            ExperimentData,
            ExperimentSettings,
        )


        EXPERIMENT = ExperimentSettings(
            experiment_id={experiment_id!r},
            experiment_title={title!r},
            experiment_note=Path(
                "experiments/{experiment_id} {note_title}.md"
            ),
            hypothesis="CHANGE ME — if ..., then ..., because ...",
            change_description="CHANGE ME — exactly one controlled change",
            success_criterion=(
                "Primary improvement >= {primary_improvement_min:+.4f}; "
                "add explicit metric guardrails below."
            ),
            primary_improvement_min={primary_improvement_min!r},
            metric_guardrails={dict(metric_guardrails)!r},
            reference_model={reference_model!r},
            primary_candidate="candidate",
            experiment_parameters={{
                # "important_parameter": "...",
            }},
            run_name="{module_run}_v1",
            artifact_dir=Path("artifacts/experiments"),
            results_registry=Path("experiments/results.csv"),
            save_artifacts=True,
            save_metric_figures=True,
            metric_figure_dpi=160,
            save_final_model=False,
            sync_experiment_note=True,
            sync_docs=True,
            allow_overwrite=True,
            parent_experiment_module={parent_experiment_module!r},
        )


        def prepare_candidate_data(
            train: pd.DataFrame,
            feature_groups: Mapping[str, Any],
            reference_settings: ModelingSettings,
        ) -> ExperimentData:
            """Подготовить candidate поверх настроек baseline/чемпиона."""

            frame = train.copy(deep=True)
            groups = copy.deepcopy(feature_groups)

            # reference_settings — настройки сравниваемой модели: исходного
            # baseline либо принятого parent-чемпиона. Не изменяйте объект
            # напрямую: ModelingSettings immutable.
            #
            # Если гипотеза меняет признаки, preprocessing или estimator,
            # создайте candidate_settings = replace(reference_settings, ...).
            # Validation contract наследуется: его смена требует нового протокола.
            candidate_settings = reference_settings

            # CHANGE ME — создавайте здесь только детерминированные raw-признаки.
            # Обучаемая статистика должна жить в sklearn transformer внутри
            # build_candidate_models, чтобы fit выполнялся отдельно на fold.

            return ExperimentData(
                frame=frame,
                feature_groups=groups,
                settings=candidate_settings,
                diagnostics={{}},
            )


        def build_candidate_models(
            preprocessor: Any,
            candidate_settings: ModelingSettings,
            experiment_settings: ExperimentSettings,
        ) -> dict[str, Any]:
            """Собрать candidate с подготовленными candidate_settings."""

            # Для feature-only изменения поверх принятого чемпиона:
            # candidate = build_reference_pipeline(
            #     experiment_settings.parent_experiment_module,
            #     preprocessor,
            #     candidate_settings,
            # )
            # return {{experiment_settings.primary_candidate: candidate}}
            raise NotImplementedError(
                "CHANGE ME — build one sklearn-compatible candidate Pipeline"
            )


        __all__ = [
            "EXPERIMENT",
            "build_candidate_models",
            "prepare_candidate_data",
        ]
        '''
    )


def scaffold_experiment(
    project_root: Path,
    experiment_id: str,
    title: str,
    *,
    slug: str,
    primary_improvement_min: float = 0.0,
    metric_guardrails: Mapping[str, float] | None = None,
    parent_experiment_module: str | None = None,
    select: bool = True,
) -> tuple[Path, str]:
    """Create one experiment module and optionally select it for the notebook."""

    root = Path(project_root).resolve()
    normalized_id = experiment_id.upper()
    normalized_slug = slug.lower().replace("-", "_")
    if not EXPERIMENT_ID_PATTERN.fullmatch(normalized_id):
        raise ValueError("experiment_id must look like EXP-003")
    if not SLUG_PATTERN.fullmatch(normalized_slug):
        raise ValueError("slug must contain lowercase letters, digits, underscores")
    if not title.strip():
        raise ValueError("title cannot be empty")
    if not math.isfinite(primary_improvement_min):
        raise ValueError("primary_improvement_min must be finite")
    guardrails = dict(metric_guardrails or {})
    if parent_experiment_module is not None:
        validate_parent_module(root, parent_experiment_module)
    for metric, threshold in guardrails.items():
        if not str(metric).strip():
            raise ValueError("metric_guardrails contains an empty metric")
        if not math.isfinite(float(threshold)):
            raise ValueError(
                f"metric_guardrails[{metric!r}] must be finite"
            )

    package_dir = (root / "src/ml_project/experiments").resolve()
    package_dir.relative_to(root)
    package_dir.mkdir(parents=True, exist_ok=True)
    module_stem = f"{normalized_id.lower().replace('-', '_')}_{normalized_slug}"
    module_path = package_dir / f"{module_stem}.py"
    if module_path.exists():
        raise FileExistsError(f"Experiment module already exists: {module_path}")

    for existing in package_dir.glob("exp_*.py"):
        if re.search(
            rf'''(?m)^\s*experiment_id\s*=\s*["']{re.escape(normalized_id)}["']''',
            existing.read_text(encoding="utf-8"),
        ):
            raise FileExistsError(
                f"{normalized_id} is already declared in {existing.name}"
            )

    selector_path = root / "src/ml_project/experiment_config.py"
    selector = ""
    if select:
        selector = selector_path.read_text(encoding="utf-8")
        if not SELECTOR_PATTERN.search(selector):
            raise ValueError(
                f"Cannot find EXPERIMENT_MODULE assignment in {selector_path}"
            )

    module_path.write_text(
        _module_source(
            normalized_id,
            title.strip(),
            normalized_slug,
            float(primary_improvement_min),
            guardrails,
            parent_experiment_module,
        ),
        encoding="utf-8",
    )
    module_name = f"ml_project.experiments.{module_stem}"

    if select:
        replacement = f'EXPERIMENT_MODULE = "{module_name}"'
        selector_path.write_text(
            SELECTOR_PATTERN.sub(replacement, selector, count=1),
            encoding="utf-8",
        )
    return module_path, module_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a versioned experiment module and select it."
    )
    parser.add_argument(
        "experiment_id",
        nargs="?",
        help="Stable ID, for example EXP-003; omitted in interactive mode",
    )
    parser.add_argument("--title", help="Human-readable title")
    parser.add_argument("--slug", help="Python-safe short slug")
    parser.add_argument(
        "--min-improvement",
        type=float,
        help="Minimum positive-direction primary delta",
    )
    parser.add_argument(
        "--guardrail",
        action="append",
        default=[],
        help="Metric=min_delta; repeat the option or separate pairs by commas",
    )
    parser.add_argument(
        "--no-select",
        action="store_true",
        help="Create the module without changing experiment_config.py",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current directory)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation when prompts are used",
    )
    parser.add_argument(
        "--no-workbench",
        action="store_true",
        help="Create only the versioned module",
    )
    parser.add_argument(
        "--workbench-only",
        action="store_true",
        help="Create a workbench for the currently selected experiment",
    )
    parser.add_argument(
        "--overwrite-workbench",
        action="store_true",
        help="Replace an existing local workbench",
    )
    parent_group = parser.add_mutually_exclusive_group()
    parent_group.add_argument(
        "--parent-module",
        help="Explicit adopted experiment module used as champion reference",
    )
    parent_group.add_argument(
        "--from-baseline",
        action="store_true",
        help="Ignore adopted champions and compare directly with baseline",
    )
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if args.workbench_only:
        (
            experiment_id,
            title,
            slug,
            module_name,
            parent_experiment_module,
        ) = selected_experiment_spec(root)
        planned_workbench = workbench_path(root, experiment_id, slug)
        if planned_workbench.exists() and not args.overwrite_workbench:
            print(f"Workbench already exists: {planned_workbench}")
            print("Existing notebook was not changed.")
            return 0
        path = create_experiment_workbench(
            root,
            experiment_id,
            title,
            slug=slug,
            module_name=module_name,
            parent_experiment_module=parent_experiment_module,
            overwrite=args.overwrite_workbench,
        )
        print(f"Workbench: {path}")
        print("Official reports were not changed.")
        return 0

    interactive = (
        args.experiment_id is None
        or args.title is None
        or args.slug is None
        or args.min_improvement is None
    )
    if interactive and not sys.stdin.isatty():
        parser.error(
            "missing arguments in a non-interactive terminal; provide "
            "experiment_id, --title, --slug and --min-improvement"
        )

    experiment_id = args.experiment_id or find_next_experiment_id(root)
    title = args.title or _prompt("Название эксперимента")
    if not title:
        parser.error("title cannot be empty")
    slug = args.slug or _prompt(
        "Техническое имя",
        slug_from_title(title),
    )
    raw_minimum = (
        str(args.min_improvement)
        if args.min_improvement is not None
        else _prompt("Минимальное улучшение основной метрики", "0.0")
    )
    try:
        primary_improvement_min = float(raw_minimum)
    except ValueError:
        parser.error("minimum improvement must be a number")

    guardrail_values = list(args.guardrail)
    if interactive and not guardrail_values:
        prompted = _prompt(
            "Guardrails (например Recall=-0.005, пусто = нет)",
            "",
        )
        if prompted:
            guardrail_values.append(prompted)
    try:
        guardrails = parse_guardrails(",".join(guardrail_values))
    except ValueError as error:
        parser.error(str(error))
    parent_experiment_module = (
        None
        if args.from_baseline
        else (
            args.parent_module
            if args.parent_module
            else find_adopted_champion_module(root)
        )
    )
    if parent_experiment_module is not None:
        try:
            validate_parent_module(root, parent_experiment_module)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))

    if interactive:
        module_stem = (
            f"{experiment_id.lower().replace('-', '_')}_"
            f"{slug.lower().replace('-', '_')}"
        )
        print("\nБудет создан эксперимент:")
        print(f"  ID: {experiment_id}")
        print(f"  Название: {title}")
        print(f"  Модуль: src/ml_project/experiments/{module_stem}.py")
        print(f"  Минимальное улучшение: {primary_improvement_min:+.4f}")
        print(f"  Guardrails: {guardrails or 'нет'}")
        print(
            "  Reference: "
            + (parent_experiment_module or "baseline_config.BASELINE")
        )
        print(f"  Выбрать для notebook: {not args.no_select}")
        print(f"  Создать локальный workbench: {not args.no_workbench}")
        if not args.yes and not _confirmed():
            print("Отменено: файлы не изменены.")
            return 0

    if not args.no_workbench:
        planned_workbench = workbench_path(root, experiment_id, slug)
        if planned_workbench.exists() and not args.overwrite_workbench:
            parser.error(
                "local workbench already exists; use --overwrite-workbench "
                f"to replace it: {planned_workbench}"
            )

    path, module_name = scaffold_experiment(
        root,
        experiment_id,
        title,
        slug=slug,
        primary_improvement_min=primary_improvement_min,
        metric_guardrails=guardrails,
        parent_experiment_module=parent_experiment_module,
        select=not args.no_select,
    )
    workbench = None
    if not args.no_workbench:
        workbench = create_experiment_workbench(
            root,
            experiment_id,
            title,
            slug=slug,
            module_name=module_name,
            parent_experiment_module=parent_experiment_module,
            overwrite=args.overwrite_workbench,
        )
    print(f"Created: {path}")
    print(f"Module: {module_name}")
    print(f"Reference: {parent_experiment_module or 'baseline_config.BASELINE'}")
    if workbench is not None:
        print(f"Workbench: {workbench}")
        print("Next: develop in the workbench, then promote code to the module.")
        print("Official run: notebooks/04_experiment.ipynb")
    if not args.no_select:
        print("Selected in: src/ml_project/experiment_config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
