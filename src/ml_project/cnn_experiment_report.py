"""Save CNN experiment comparisons, plots and an Obsidian report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_START = "<!-- auto:cnn-experiment-report:start -->"
REPORT_END = "<!-- auto:cnn-experiment-report:end -->"
VALID_DECISIONS = {"review", "adopt", "reject", "iterate"}


@dataclass(frozen=True)
class CNNFinalization:
    card_path: Path
    summary_path: Path
    plot_dir: Path


def _read_table(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.read_csv(value)


def _primary_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if "score" in metrics:
        selected = metrics.loc[metrics["score"] == "expected_signal"]
        if selected.empty:
            raise ValueError("metrics contain no expected_signal rows")
        return selected.copy()
    return metrics.copy()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _save_plots(
    history: pd.DataFrame,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    plot_dir: Path,
    reference_id: str,
    experiment_id: str,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if {"step", "ema_presence", "ema_intensity", "ema_total"} <= set(history):
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        for column, label in (
            ("ema_presence", "presence"),
            ("ema_intensity", "intensity"),
            ("ema_total", "total"),
        ):
            ax.plot(history["step"], history[column], label=label)
        ax.set(xlabel="Training step", ylabel="EMA loss", title="Training losses")
        ax.legend()
        fig.savefig(plot_dir / "01_training_losses.png", dpi=170, facecolor="white")
        plt.close(fig)

    ref_overall = reference.loc[reference["segment"] == "overall"].iloc[0]
    cand_overall = candidate.loc[candidate["segment"] == "overall"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, metric, label in (
        (axes[0], "average_precision", "Validation AP"),
        (axes[1], "epic_spearman", "EPIC Spearman"),
    ):
        values = [float(ref_overall[metric]), float(cand_overall[metric])]
        ax.bar([reference_id, experiment_id], values, color=["#9AA0A6", "#1967D2"])
        ax.set_title(label)
        ax.bar_label(ax.containers[0], fmt="%.6f")
    fig.savefig(plot_dir / "02_overall_comparison.png", dpi=170, facecolor="white")
    plt.close(fig)

    merged = reference.merge(candidate, on="segment", suffixes=("_ref", "_candidate"))
    merged = merged.loc[merged["segment"] != "overall"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(len(merged))
    for ax, metric, label in (
        (axes[0], "average_precision", "AP"),
        (axes[1], "epic_spearman", "EPIC Spearman"),
    ):
        ax.plot(x, merged[f"{metric}_ref"], marker="o", label=reference_id)
        ax.plot(x, merged[f"{metric}_candidate"], marker="o", label=experiment_id)
        ax.set_xticks(x, merged["segment"], rotation=20)
        ax.set_title(f"Per-contig {label}")
        ax.legend()
    fig.savefig(plot_dir / "03_per_contig_comparison.png", dpi=170, facecolor="white")
    plt.close(fig)


def _report_markdown(
    root: Path,
    experiment_id: str,
    reference_id: str,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    plot_dir: Path,
    notes: str,
) -> tuple[str, dict[str, float]]:
    ref = reference.loc[reference["segment"] == "overall"].iloc[0]
    cand = candidate.loc[candidate["segment"] == "overall"].iloc[0]
    ref_ap = float(ref["average_precision"])
    cand_ap = float(cand["average_precision"])
    ref_sp = float(ref["epic_spearman"])
    cand_sp = float(cand["epic_spearman"])
    summary = {
        "reference_ap": ref_ap,
        "candidate_ap": cand_ap,
        "delta_ap": cand_ap - ref_ap,
        "relative_ap": cand_ap / ref_ap - 1.0,
        "reference_spearman": ref_sp,
        "candidate_spearman": cand_sp,
        "delta_spearman": cand_sp - ref_sp,
    }
    rel_plot = _relative(root, plot_dir)
    table = candidate[["segment", "average_precision", "epic_spearman"]].copy()
    table["average_precision"] = table["average_precision"].map(lambda x: f"{x:.9f}")
    table["epic_spearman"] = table["epic_spearman"].map(lambda x: f"{x:.9f}")
    rows = "\n".join(
        f"| {row.segment} | {row.average_precision} | {row.epic_spearman} |"
        for row in table.itertuples(index=False)
    )
    note_text = notes.strip() or "—"
    return f'''## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| {reference_id} | {ref_ap:.9f} | {ref_sp:.9f} |
| **{experiment_id}** | **{cand_ap:.9f}** | **{cand_sp:.9f}** |

- Δ AP: `{cand_ap - ref_ap:+.9f}` ({(cand_ap / ref_ap - 1.0):+.2%}).
- Δ EPIC Spearman: `{cand_sp - ref_sp:+.9f}`.

### Candidate по contigs

| Segment | AP | EPIC Spearman |
|---|---:|---:|
{rows}

### Графики

![[{rel_plot}/01_training_losses.png]]

![[{rel_plot}/02_overall_comparison.png]]

![[{rel_plot}/03_per_contig_comparison.png]]

### Финальные заметки

{note_text}
''', summary


def _update_registry(
    root: Path,
    *,
    experiment_id: str,
    title: str,
    decision: str,
    average_precision: float,
    epic_spearman: float,
) -> None:
    index_path = root / "cnn_exp/_index.md"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    escaped_link = f"[[cnn_exp/{experiment_id}.md\\|{experiment_id}]]"
    row = (
        f"| {escaped_link} | {title} | {average_precision:.9f} | "
        f"{epic_spearman:.9f} | `{decision}` |"
    )
    prefix = f"| [[cnn_exp/{experiment_id}.md"
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = row
            break
    else:
        table_rows = [
            index
            for index, line in enumerate(lines)
            if line.startswith("| [[cnn_exp/CNN-EXP-")
        ]
        if not table_rows:
            raise ValueError("CNN registry table was not found")
        lines.insert(table_rows[-1] + 1, row)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_cnn_experiment(
    *,
    project_root: str | Path,
    experiment_id: str,
    title: str,
    hypothesis: str,
    notebook_path: str | Path,
    artifact_dir: str | Path,
    training_history: str | Path | pd.DataFrame,
    candidate_metrics: str | Path | pd.DataFrame,
    reference_experiment: str,
    reference_metrics: str | Path | pd.DataFrame,
    decision: str = "review",
    notes: str = "",
) -> CNNFinalization:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")
    root = Path(project_root).resolve()
    artifacts = Path(artifact_dir)
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    history = _read_table(training_history)
    candidate_all = _read_table(candidate_metrics)
    reference_all = _read_table(
        reference_metrics if isinstance(reference_metrics, pd.DataFrame)
        else root / reference_metrics
    )
    candidate = _primary_rows(candidate_all)
    reference = _primary_rows(reference_all)
    required = {"segment", "average_precision", "epic_spearman"}
    for name, frame in (("candidate", candidate), ("reference", reference)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} metrics miss columns: {sorted(missing)}")

    plot_dir = artifacts / "plots"
    _save_plots(
        history,
        reference,
        candidate,
        plot_dir,
        reference_experiment,
        experiment_id,
    )
    report, summary = _report_markdown(
        root,
        experiment_id,
        reference_experiment,
        reference,
        candidate,
        plot_dir,
        notes,
    )
    summary_payload = {
        "experiment_id": experiment_id,
        "title": title,
        "hypothesis": hypothesis,
        "reference_experiment": reference_experiment,
        "decision": decision,
        "date": date.today().isoformat(),
        "notebook": str(notebook_path).replace("\\", "/"),
        **summary,
    }
    summary_path = artifacts / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    card_path = root / "cnn_exp" / f"{experiment_id}.md"
    text = card_path.read_text(encoding="utf-8")
    if REPORT_START not in text or REPORT_END not in text:
        raise ValueError(f"Auto-report markers missing in {card_path}")
    before, remainder = text.split(REPORT_START, 1)
    _, after = remainder.split(REPORT_END, 1)
    text = before + REPORT_START + "\n\n" + report + "\n" + REPORT_END + after
    text = re.sub(r"(?m)^status: .+$", "status: completed", text, count=1)
    text = re.sub(r"(?m)^decision: .+$", f"decision: {decision}", text, count=1)
    card_path.write_text(text, encoding="utf-8")

    _update_registry(
        root,
        experiment_id=experiment_id,
        title=title,
        decision=decision,
        average_precision=float(summary["candidate_ap"]),
        epic_spearman=float(summary["candidate_spearman"]),
    )

    if decision == "adopt":
        checkpoint_candidates = sorted(artifacts.glob("*.pt"))
        champion = {
            "experiment_id": experiment_id,
            "notebook": str(notebook_path).replace("\\", "/"),
            "metrics": _relative(root, Path(candidate_metrics)),
            "checkpoint": (
                _relative(root, checkpoint_candidates[-1])
                if checkpoint_candidates
                else ""
            ),
        }
        (root / "cnn_exp/champion.json").write_text(
            json.dumps(champion, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return CNNFinalization(card_path, summary_path, plot_dir)


__all__ = ["CNNFinalization", "finalize_cnn_experiment"]
