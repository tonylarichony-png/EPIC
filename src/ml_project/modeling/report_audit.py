"""Integrity checks for generated modeling reports."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


def audit_modeling_report(project_root: Path) -> list[str]:
    """Return deterministic integrity warnings for generated modeling reports."""

    root = Path(project_root).resolve()
    issues: list[str] = []
    marker_pattern = re.compile(r"<!-- auto:([^:]+):(start|end) -->")
    link_pattern = re.compile(r"\[\[([^\]|#]+)")

    report_paths = [
        root / "README.md",
        root / "docs/03_validation.md",
        root / "docs/04_features.md",
        root / "docs/05_experiments.md",
        root / "experiments/_index.md",
        *sorted((root / "experiments").glob("*.md")),
    ]
    for note in dict.fromkeys(path for path in report_paths if path.exists()):
        text = note.read_text(encoding="utf-8")
        markers: dict[str, dict[str, int]] = {}
        for name, boundary in marker_pattern.findall(text):
            markers.setdefault(name, {"start": 0, "end": 0})[boundary] += 1
        for name, counts in markers.items():
            if counts != {"start": 1, "end": 1}:
                issues.append(
                    f"{note.relative_to(root).as_posix()}: auto-block {name!r} "
                    f"has start={counts['start']}, end={counts['end']}"
                )
        for target in link_pattern.findall(text):
            normalized = target.strip()
            if not normalized.startswith(("assets/", "artifacts/")):
                continue
            if not (root / normalized).exists():
                issues.append(
                    f"{note.relative_to(root).as_posix()}: missing {normalized}"
                )

    registry_path = root / "experiments/results.csv"
    if registry_path.exists() and registry_path.stat().st_size:
        registry = pd.read_csv(registry_path)
        for record in registry.to_dict(orient="records"):
            note_path = root / str(record["note"])
            if not note_path.exists():
                issues.append(
                    f"results.csv: missing experiment note {record['note']}"
                )
                continue
            text = note_path.read_text(encoding="utf-8")
            frontmatter = re.search(
                r"(?m)^decision:\s*[\"']?([^\"'\r\n]+)", text
            )
            if frontmatter and frontmatter.group(1).strip() != str(record["decision"]):
                issues.append(
                    f"{record['experiment_id']}: card decision "
                    f"{frontmatter.group(1).strip()!r} != registry "
                    f"{str(record['decision'])!r}"
                )
            dataset_version = str(record.get("dataset_version", ""))
            if dataset_version and dataset_version != "nan" and len(dataset_version) < 64:
                issues.append(
                    f"{record['experiment_id']}: results.csv stores a shortened "
                    "dataset_version; rerun the generic experiment notebook"
                )
            implementation = str(record.get("implementation_path", ""))
            expected_hash = str(record.get("implementation_sha256", ""))
            if implementation and implementation != "nan":
                source_path = root / implementation
                if not source_path.exists():
                    issues.append(
                        f"{record['experiment_id']}: missing implementation "
                        f"{implementation}"
                    )
                elif expected_hash and expected_hash != "nan":
                    actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
                    if actual_hash != expected_hash:
                        issues.append(
                            f"{record['experiment_id']}: implementation hash "
                            "differs from results.csv"
                        )
    return issues
