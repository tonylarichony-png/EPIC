"""Synchronize mutable experiment decisions from Markdown cards."""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from . import modeling as modeling_tools
from .docsync import MarkdownDocument, dataframe_to_markdown
from .experiment_relations import sync_experiment_eda_relations
from .modeling import ExperimentSettings


DECISIONS = {"pending", "adopt", "reject", "iterate", "inconclusive"}
_SCALAR_FIELD = re.compile(
    r"(?m)^(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>[^\r\n]*)$"
)
_REPORT_DECISION_ROW = re.compile(r"(?m)^\|\s*Решение\s*\|[^\n]*$")
_DOC_REGISTRY_COLUMNS = {
    "experiment_id",
    "title",
    "note",
    "hypothesis",
    "change",
    "run_name",
    "primary_metric",
    "direction",
    "reference_score",
    "reference_std",
    "candidate_score",
    "improvement",
    "criteria_passed",
    "decision",
    "parent_experiment_id",
}


def _frontmatter_scalars(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Experiment card has no YAML frontmatter: {path}")
    closing = text.find("\n---\n", 4)
    if closing < 0:
        raise ValueError(f"Experiment card has invalid YAML frontmatter: {path}")
    values: dict[str, str] = {}
    for match in _SCALAR_FIELD.finditer(text[4:closing]):
        value = match.group("value").strip().strip("'\"")
        values[match.group("key")] = value
    return values


def experiment_card_decisions(project_root: str | Path) -> dict[str, str]:
    """Return validated experiment-level decisions keyed by experiment ID."""

    root = Path(project_root).resolve()
    decisions: dict[str, str] = {}
    for path in sorted((root / "experiments").glob("EXP-*.md")):
        values = _frontmatter_scalars(path)
        experiment_id = values.get("id", "").upper()
        decision = values.get("decision", "")
        if not experiment_id or not decision:
            continue
        if decision not in DECISIONS:
            raise ValueError(
                f"Invalid decision {decision!r} in {path}; expected one of: "
                + ", ".join(sorted(DECISIONS))
            )
        if experiment_id in decisions:
            raise ValueError(f"Duplicate experiment card ID: {experiment_id}")
        decisions[experiment_id] = decision
    return decisions


def decision_from_card(
    project_root: str | Path,
    note: Path,
    *,
    default: str = "pending",
) -> str:
    """Read one decision from its card, falling back before the first run."""

    root = Path(project_root).resolve()
    path = (root / note).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Experiment note resolves outside project root: {note}") from error
    if not path.exists():
        return default
    decision = _frontmatter_scalars(path).get("decision", default)
    if decision not in DECISIONS:
        raise ValueError(
            f"Invalid decision {decision!r} in {path}; expected one of: "
            + ", ".join(sorted(DECISIONS))
        )
    return decision


def settings_with_card_decision(
    project_root: str | Path,
    settings: ExperimentSettings,
) -> ExperimentSettings:
    """Overlay mutable card state without changing the versioned module."""

    decision = decision_from_card(
        project_root,
        settings.experiment_note,
        default=settings.decision,
    )
    return replace(settings, decision=decision)


def _update_report_decision(path: Path, decision: str) -> bool:
    text = path.read_text(encoding="utf-8")
    start_marker = "<!-- auto:experiment-report:start -->"
    end_marker = "<!-- auto:experiment-report:end -->"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        return False
    block = text[start:end]
    matches = list(_REPORT_DECISION_ROW.finditer(block))
    if not matches:
        return False
    if len(matches) > 1:
        raise ValueError(f"Multiple generated decision rows in {path}")
    updated_block = _REPORT_DECISION_ROW.sub(
        f"| Решение | {decision} |",
        block,
        count=1,
    )
    if updated_block == block:
        return False
    path.write_text(text[:start] + updated_block + text[end:], encoding="utf-8")
    return True


def _update_registries(
    root: Path,
    decisions: Mapping[str, str],
) -> tuple[int, list[pd.DataFrame]]:
    changed_rows = 0
    documentation_frames: list[pd.DataFrame] = []
    for path in sorted((root / "experiments").glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
        if not {"experiment_id", "decision"}.issubset(frame.columns):
            continue
        mapped = frame["experiment_id"].astype(str).map(decisions)
        mask = mapped.notna() & frame["decision"].astype(str).ne(mapped.astype(str))
        changed_rows += int(mask.sum())
        if mask.any():
            frame.loc[mask, "decision"] = mapped[mask]
            frame.to_csv(path, index=False)
        if _DOC_REGISTRY_COLUMNS.issubset(frame.columns):
            documentation_frames.append(frame.copy())
    return changed_rows, documentation_frames


def _combined_registry(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    registry = pd.concat(frames, ignore_index=True)
    registry["_experiment_number"] = pd.to_numeric(
        registry["experiment_id"].astype(str).str.extract(r"EXP-(\d+)")[0],
        errors="coerce",
    )
    registry["_source_order"] = range(len(registry))
    registry = registry.sort_values(
        ["_experiment_number", "run_name", "_source_order"],
        kind="stable",
    )
    registry = registry.drop_duplicates(
        subset=["experiment_id", "run_name"],
        keep="last",
    )
    return registry.drop(columns=["_source_order"]).reset_index(drop=True)


def _refresh_overview_docs(root: Path, registry: pd.DataFrame) -> list[str]:
    if registry.empty:
        return []
    latest_row = registry.sort_values(
        ["_experiment_number", "run_name"], kind="stable"
    ).iloc[-1]
    baseline_rows = registry[
        registry["parent_experiment_id"].astype(str).eq("EXP-001")
    ]
    baseline_record = (
        baseline_rows.sort_values("_experiment_number").iloc[0]
        if not baseline_rows.empty
        else registry.sort_values("_experiment_number").iloc[0]
    )
    baseline_note = Path("experiments/EXP-001 Baseline.md")
    baseline_score = float(baseline_record["reference_score"])
    baseline_std = float(baseline_record["reference_std"])
    public_registry = registry.drop(columns=["_experiment_number"])

    latest = pd.DataFrame(
        [
            (
                "Эксперимент",
                f"[[{latest_row['note']}|{latest_row['experiment_id']} — "
                f"{latest_row['title']}]]",
            ),
            ("Родитель", latest_row["parent_experiment_id"]),
            ("Гипотеза", latest_row["hypothesis"]),
            ("Изменение", latest_row["change"]),
            ("Метрика", latest_row["primary_metric"]),
            ("Reference", f"{float(latest_row['reference_score']):.4f}"),
            ("Кандидат", f"{float(latest_row['candidate_score']):.4f}"),
            ("Δ к reference", f"{float(latest_row['improvement']):+.4f}"),
            (
                "Формальные критерии",
                "passed" if bool(latest_row["criteria_passed"]) else "failed",
            ),
            ("Решение", latest_row["decision"]),
        ],
        columns=["Поле", "Значение"],
    )
    leaderboard = public_registry.copy()
    leaderboard["Experiment"] = leaderboard.apply(
        lambda item: f"[[{item['note']}|{item['experiment_id']}]]", axis=1
    )
    leaderboard = leaderboard[
        [
            "Experiment",
            "parent_experiment_id",
            "hypothesis",
            "change",
            "primary_metric",
            "reference_score",
            "candidate_score",
            "improvement",
            "criteria_passed",
            "decision",
        ]
    ].rename(
        columns={
            "hypothesis": "Hypothesis",
            "change": "Change",
            "parent_experiment_id": "Parent",
            "primary_metric": "Metric",
            "reference_score": "Reference",
            "candidate_score": "Result",
            "improvement": "Δ",
            "criteria_passed": "Criteria",
            "decision": "Decision",
        }
    )

    updated: list[str] = []
    docs_path = root / "docs/05_experiments.md"
    if docs_path.exists():
        updated.extend(
            MarkdownDocument(docs_path).update_blocks(
                {
                    "latest-experiment": dataframe_to_markdown(latest, float_digits=4),
                    "experiment-leaderboard": dataframe_to_markdown(
                        leaderboard, float_digits=4
                    ),
                    "best-measured-result": modeling_tools.build_best_result_block(
                        baseline_metric=str(latest_row["primary_metric"]),
                        baseline_score=baseline_score,
                        baseline_std=baseline_std,
                        baseline_direction=str(latest_row["direction"]),
                        baseline_note=baseline_note,
                        experiments=public_registry,
                    ),
                }
            )
        )
    index_path = root / "experiments/_index.md"
    if index_path.exists():
        updated.extend(
            MarkdownDocument(index_path).update_blocks(
                {
                    "experiment-registry": modeling_tools.build_experiment_registry_block(
                        baseline_note,
                        public_registry,
                        baseline_run="baseline",
                        baseline_metric=str(latest_row["primary_metric"]),
                        baseline_score=baseline_score,
                    )
                }
            )
        )
    readme_path = root / "README.md"
    if readme_path.exists():
        updated.extend(
            MarkdownDocument(readme_path).update_blocks(
                {
                    "key-results": modeling_tools.build_key_results_block(
                        baseline_metric=str(latest_row["primary_metric"]),
                        baseline_score=baseline_score,
                        baseline_note=baseline_note,
                        experiments=public_registry,
                    )
                }
            )
        )
    return updated


def sync_experiment_state(project_root: str | Path) -> dict[str, Any]:
    """Propagate card decisions to registries, reports and EDA relations."""

    root = Path(project_root).resolve()
    decisions = experiment_card_decisions(root)
    changed_rows, frames = _update_registries(root, decisions)
    changed_cards = 0
    for path in sorted((root / "experiments").glob("EXP-*.md")):
        values = _frontmatter_scalars(path)
        decision = decisions.get(values.get("id", "").upper())
        if decision and _update_report_decision(path, decision):
            changed_cards += 1
    registry = _combined_registry(frames)
    overview_files = _refresh_overview_docs(root, registry)
    relations = sync_experiment_eda_relations(root)
    return {
        "decisions": len(decisions),
        "registry_rows_updated": changed_rows,
        "experiment_reports_updated": changed_cards,
        "overview_files": overview_files,
        "relations": relations["relations"],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize experiment-card decisions and EDA links."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = sync_experiment_state(args.project_root)
    print(
        "Синхронизировано: "
        f"решений {result['decisions']}, "
        f"строк registry {result['registry_rows_updated']}, "
        f"отчётов {result['experiment_reports_updated']}, "
        f"EDA-связей {result['relations']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DECISIONS",
    "decision_from_card",
    "experiment_card_decisions",
    "settings_with_card_decision",
    "sync_experiment_state",
]
