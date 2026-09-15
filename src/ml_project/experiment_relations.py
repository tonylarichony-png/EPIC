"""Synchronize post-run EDA links between experiment and finding cards."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .docsync import MarkdownDocument, dataframe_to_markdown


_EXPERIMENT_ID = re.compile(r"^EXP-\d{3,}$")
_EDA_ID = re.compile(r"^EDA-\d{3,}$")
_EXPERIMENT_HEADING = re.compile(r"^# (EXP-\d{3,}) — (.+)$", re.MULTILINE)
_EDA_HEADING = re.compile(r"^# (EDA-\d{3,}) — (.+)$", re.MULTILINE)
_FRONTMATTER_FIELD = re.compile(r"^([A-Za-z0-9_-]+):\s*(.*)$")


def _split_frontmatter(text: str, path: Path) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise ValueError(f"Card has no YAML frontmatter: {path}")
    closing = text.find("\n---\n", 4)
    if closing < 0:
        raise ValueError(f"Card has invalid YAML frontmatter: {path}")
    return text[4:closing], text[closing + 5 :]


def _frontmatter_values(text: str, path: Path) -> dict[str, str]:
    frontmatter, _ = _split_frontmatter(text, path)
    values: dict[str, str] = {}
    lines = frontmatter.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _FRONTMATTER_FIELD.fullmatch(line)
        if match is None:
            index += 1
            continue

        key = match.group(1)
        inline_value = match.group(2).strip()
        if inline_value:
            values[key] = inline_value
            index += 1
            continue

        # Obsidian normally formats YAML lists as an indented block:
        #
        # eda_findings:
        #   - EDA-003
        #
        # Keep that block as the field value so the type-specific parser can
        # validate it.  A following top-level field starts at column zero.
        block: list[str] = []
        index += 1
        while index < len(lines):
            nested_line = lines[index]
            if nested_line and not nested_line[0].isspace():
                break
            if nested_line.strip():
                block.append(nested_line.strip())
            index += 1
        values[key] = "\n".join(block)
    return values


def _scalar(value: str, default: str = "") -> str:
    if not value:
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value.strip().strip("'\"")
    return str(parsed) if parsed is not None else default


def _id_list(value: str, *, path: Path) -> tuple[str, ...]:
    value = value.replace("\u00a0", " ").strip()
    if not value:
        return ()

    if value.startswith("-"):
        parsed_items: list[Any] = []
        for line in value.splitlines():
            item = line.strip()
            if not item:
                continue
            if not item.startswith("-") or not item[1:].strip():
                raise ValueError(
                    f"eda_findings must be a YAML list of EDA IDs: {path}"
                )
            raw_item = item[1:].strip()
            try:
                parsed_item = json.loads(raw_item)
            except json.JSONDecodeError:
                try:
                    parsed_item = ast.literal_eval(raw_item)
                except (SyntaxError, ValueError):
                    parsed_item = raw_item.strip("'\"")
            # Be forgiving of the accidental Obsidian form
            # ``- [\"EDA-003\"]`` and treat it as one list item.
            if isinstance(parsed_item, (list, tuple)):
                parsed_items.extend(parsed_item)
            else:
                parsed_items.append(parsed_item)
        parsed: Any = parsed_items
    else:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"eda_findings must be a list such as [\"EDA-003\"]: {path}"
                ) from error
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"eda_findings must be a list: {path}")
    identifiers = tuple(str(item).strip().upper() for item in parsed)
    invalid = [item for item in identifiers if not _EDA_ID.fullmatch(item)]
    if invalid:
        raise ValueError(
            f"Invalid EDA IDs in {path}: " + ", ".join(invalid)
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate EDA IDs in {path}")
    return identifiers


def _ensure_frontmatter_field(path: Path, field: str, default: str) -> None:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text, path)
    pattern = re.compile(rf"^{re.escape(field)}:.*$", re.MULTILINE)
    if pattern.search(frontmatter):
        return
    updated = frontmatter.rstrip() + f"\n{field}: {default}"
    path.write_text(
        "---\n" + updated + "\n---\n" + body,
        encoding="utf-8",
    )


def _ensure_generated_block(
    path: Path,
    *,
    block_id: str,
    heading: str,
    before_heading: str | None,
) -> None:
    text = path.read_text(encoding="utf-8")
    start = f"<!-- auto:{block_id}:start -->"
    end = f"<!-- auto:{block_id}:end -->"
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count == 1 and end_count == 1:
        return
    if start_count or end_count:
        raise ValueError(f"Broken {block_id!r} marker pair in {path}")
    section = (
        f"## {heading}\n\n"
        f"{start}\n\n"
        "> Связи появятся после синхронизации.\n\n"
        f"{end}\n\n"
    )
    marker = f"## {before_heading}\n" if before_heading else ""
    if marker and marker in text:
        index = text.index(marker)
        updated = text[:index] + section + text[index:]
    else:
        updated = text.rstrip() + "\n\n" + section
    path.write_text(updated, encoding="utf-8")


def _update_generated_block(path: Path, block_id: str, content: str) -> None:
    """Update a relation block, with a Windows fallback for open notes."""

    try:
        MarkdownDocument(path).update_blocks({block_id: content})
    except PermissionError:
        temporary = path.with_suffix(path.suffix + ".tmp")
        if not temporary.exists():
            raise
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink()


def _abstract_conclusion(text: str) -> str:
    match = re.search(
        r"^> \[!abstract\][^\n]*\n(?P<body>(?:>[^\n]*(?:\n|$))+)",
        text,
        re.MULTILINE,
    )
    if match is None:
        return "—"
    lines = [
        line.removeprefix("> ").strip()
        for line in match.group("body").splitlines()
    ]
    return " ".join(line for line in lines if line) or "—"


def _features(value: str) -> str:
    if not value:
        return "—"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    if not isinstance(parsed, (list, tuple)):
        return str(parsed)
    return ", ".join(map(str, parsed)) or "—"


def _finding_catalog(root: Path) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    directory = root / "eda/findings"
    for path in sorted(directory.glob("EDA-*.md")):
        text = path.read_text(encoding="utf-8")
        values = _frontmatter_values(text, path)
        heading = _EDA_HEADING.search(text)
        finding_id = _scalar(values.get("id", ""), path.stem).upper()
        if not _EDA_ID.fullmatch(finding_id):
            continue
        if finding_id in catalog:
            raise ValueError(f"Duplicate EDA finding ID: {finding_id}")
        title = (
            _scalar(values.get("title", ""))
            or (heading.group(2) if heading else finding_id)
        )
        catalog[finding_id] = {
            "id": finding_id,
            "title": title,
            "path": path,
            "relative_path": path.relative_to(root).as_posix(),
            "features": _features(values.get("features", "")),
            "conclusion": _abstract_conclusion(text),
        }
    return catalog


def _experiment_cards(root: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted((root / "experiments").glob("EXP-*.md")):
        text = path.read_text(encoding="utf-8")
        values = _frontmatter_values(text, path)
        heading = _EXPERIMENT_HEADING.search(text)
        experiment_id = _scalar(values.get("id", ""), "").upper()
        if not experiment_id and heading:
            experiment_id = heading.group(1)
        if not _EXPERIMENT_ID.fullmatch(experiment_id) or experiment_id == "EXP-001":
            continue
        if experiment_id in seen:
            raise ValueError(f"Duplicate experiment card ID: {experiment_id}")
        seen.add(experiment_id)
        cards.append(
            {
                "id": experiment_id,
                "title": heading.group(2) if heading else experiment_id,
                "path": path,
                "relative_path": path.relative_to(root).as_posix(),
                "hypothesis": _scalar(values.get("hypothesis", ""), "—"),
                "decision": _scalar(values.get("decision", ""), "pending"),
                "primary_metric": _scalar(
                    values.get("primary_metric", ""),
                    "—",
                ),
                "eda_findings": _id_list(
                    values.get("eda_findings", ""),
                    path=path,
                ),
            }
        )
    return cards


def _experiment_results(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "experiments").glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
        if "experiment_id" not in frame:
            continue
        for row in frame.to_dict(orient="records"):
            experiment_id = str(row.get("experiment_id", "")).upper()
            if _EXPERIMENT_ID.fullmatch(experiment_id):
                records[experiment_id] = row
    return records


def _format_delta(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:+.4f}" if np.isfinite(number) else "—"


def sync_experiment_eda_relations(project_root: str | Path) -> dict[str, Any]:
    """Rebuild both card directions from experiment-card ``eda_findings``."""

    root = Path(project_root).resolve()
    experiments = _experiment_cards(root)
    findings = _finding_catalog(root)
    unknown = sorted(
        {
            finding_id
            for experiment in experiments
            for finding_id in experiment["eda_findings"]
            if finding_id not in findings
        }
    )
    if unknown:
        raise ValueError(
            "Experiment cards refer to missing EDA findings: "
            + ", ".join(unknown)
        )

    results = _experiment_results(root)
    for experiment in experiments:
        path = experiment["path"]
        _ensure_frontmatter_field(path, "eda_findings", "[]")
        _ensure_generated_block(
            path,
            block_id="experiment-eda-links",
            heading="EDA-основания",
            before_heading="Анализ результата — заполнить вручную",
        )
        rows = []
        for finding_id in experiment["eda_findings"]:
            finding = findings[finding_id]
            rows.append(
                {
                    "EDA-наблюдение": (
                        f"[[{finding['relative_path']}|"
                        f"{finding_id} — {finding['title']}]]"
                    ),
                    "Признаки": finding["features"],
                    "Ключевой вывод": finding["conclusion"],
                }
            )
        if rows:
            content = dataframe_to_markdown(pd.DataFrame(rows))
        else:
            content = (
                "> EDA-основания пока не указаны. Добавьте ID в frontmatter: "
                "`eda_findings: [\"EDA-003\"]`, затем запустите "
                "`sync-experiment-links.cmd`."
            )
        _update_generated_block(path, "experiment-eda-links", content)

    linked_by_finding: dict[str, list[dict[str, Any]]] = {
        finding_id: [] for finding_id in findings
    }
    for experiment in experiments:
        result = results.get(experiment["id"], {})
        for finding_id in experiment["eda_findings"]:
            linked_by_finding[finding_id].append(
                {
                    "Эксперимент": (
                        f"[[{experiment['relative_path']}|"
                        f"{experiment['id']} — {experiment['title']}]]"
                    ),
                    "Гипотеза": experiment["hypothesis"],
                    "Решение": experiment["decision"],
                    "Метрика": result.get(
                        "primary_metric",
                        experiment["primary_metric"],
                    ),
                    "Δ": _format_delta(result.get("improvement")),
                }
            )

    for finding_id, finding in findings.items():
        path = finding["path"]
        _ensure_generated_block(
            path,
            block_id="eda-experiment-links",
            heading="Связанные эксперименты",
            before_heading="Источник",
        )
        rows = linked_by_finding[finding_id]
        content = (
            dataframe_to_markdown(pd.DataFrame(rows))
            if rows
            else "> Эксперименты на основании этого наблюдения пока не зарегистрированы."
        )
        _update_generated_block(path, "eda-experiment-links", content)

    relation_count = sum(
        len(experiment["eda_findings"]) for experiment in experiments
    )
    return {
        "experiment_cards": len(experiments),
        "eda_cards": len(findings),
        "relations": relation_count,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize EDA links in experiment and finding cards."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = sync_experiment_eda_relations(args.project_root)
    print(
        "Синхронизировано: "
        f"experiment-карточек {result['experiment_cards']}, "
        f"EDA-карточек {result['eda_cards']}, "
        f"связей {result['relations']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["sync_experiment_eda_relations"]
