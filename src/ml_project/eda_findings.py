"""Save selected EDA findings as Obsidian-native notes and artifacts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from .docsync import MarkdownDocument, dataframe_to_markdown


_FINDING_FILE = re.compile(r"^EDA-(\d{3,})\.md$")
_FINDING_HEADING = re.compile(r"^# (EDA-\d{3,}) — (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class EdaFinding:
    """Paths and identifiers produced by :func:`save_eda_finding`."""

    finding_id: str
    title: str
    note_path: Path
    figure_path: Path | None
    table_paths: tuple[Path, ...] = ()

    @property
    def note_wikilink(self) -> str:
        relative = self.note_path.as_posix()
        return f"[[{relative}|{self.finding_id} — {self.title}]]"

    @property
    def figure_embed(self) -> str:
        if self.figure_path is None:
            return ""
        return f"![[{self.figure_path.as_posix()}]]"


TableFormatter = str | Callable[[Any], Any]


def _clean_single_line(value: str, *, field: str) -> str:
    cleaned = " ".join(str(value).split())
    if not cleaned:
        raise ValueError(f"{field} must not be empty.")
    return cleaned


def _yaml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _next_finding_id(directory: Path) -> str:
    numbers = [
        int(match.group(1))
        for path in directory.glob("EDA-*.md")
        if (match := _FINDING_FILE.fullmatch(path.name))
    ]
    number = max(numbers, default=0) + 1
    return f"EDA-{number:03d}"


def _read_finding_label(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    match = _FINDING_HEADING.search(text)
    if match is None:
        return path.stem, path.stem
    finding_id, title = match.groups()
    return finding_id, title.replace("|", "—")


def _has_default_index(frame: pd.DataFrame) -> bool:
    index = frame.index
    return (
        isinstance(index, pd.RangeIndex)
        and index.name is None
        and index.start == 0
        and index.stop == len(frame)
        and index.step == 1
    )


def _format_table_value(value: Any, formatter: TableFormatter) -> Any:
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if callable(formatter):
        return formatter(value)
    return str(formatter).format(value)


def _prepare_table(
    frame: pd.DataFrame,
    formats: Mapping[str, TableFormatter] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw and presentation copies with a visible meaningful index."""
    raw = frame.copy()
    if not _has_default_index(raw):
        raw = raw.reset_index()

    formatted = raw.copy()
    for column, formatter in (formats or {}).items():
        if column not in formatted.columns:
            raise KeyError(f"Table format refers to absent column: {column!r}")
        formatted[column] = formatted[column].map(
            lambda value: _format_table_value(value, formatter)
        )
    return raw, formatted


def sync_eda_findings(
    project_root: str | Path,
    *,
    docs_path: str | Path = "docs/02_eda.md",
) -> list[Path]:
    """Refresh the managed list of finding links in ``docs/02_eda.md``."""
    root = Path(project_root).resolve()
    findings_directory = root / "eda" / "findings"
    notes = sorted(
        (
            path
            for path in findings_directory.glob("EDA-*.md")
            if _FINDING_FILE.fullmatch(path.name)
        ),
        key=lambda path: int(_FINDING_FILE.fullmatch(path.name).group(1)),
    )

    if notes:
        rows = []
        for path in notes:
            finding_id, title = _read_finding_label(path)
            relative = path.relative_to(root).as_posix()
            rows.append(f"- [[{relative}|{finding_id} — {title}]]")
        content = "\n".join(rows)
    else:
        content = "> Сохранённых EDA-наблюдений пока нет."

    MarkdownDocument(root / docs_path).update_blocks(
        {"eda-findings": content}
    )
    return [path.relative_to(root) for path in notes]


def save_eda_finding(
    *,
    project_root: str | Path,
    title: str,
    question: str,
    method: str,
    conclusion: str,
    figure: Any = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    table_formats: Mapping[
        str,
        Mapping[str, TableFormatter],
    ]
    | None = None,
    table_preview_rows: int = 30,
    features: Iterable[str] = (),
    hypothesis: str = "",
    notebook: str = "notebooks/02_eda_hypotheses.ipynb",
    dpi: int = 160,
) -> EdaFinding:
    """Save figures/tables, create an EDA note, and link it from docs.

    A new immutable ``EDA-NNN`` card is created on every call. Existing cards
    and artifacts are never overwritten. Tables larger than
    ``table_preview_rows`` are previewed in Markdown and saved fully as CSV.
    """
    root = Path(project_root).resolve()
    title = _clean_single_line(title, field="title")
    question = str(question).strip()
    method = str(method).strip()
    conclusion = str(conclusion).strip()
    hypothesis = str(hypothesis).strip()
    feature_names = [
        _clean_single_line(feature, field="feature")
        for feature in features
        if str(feature).strip()
    ]

    if not conclusion:
        raise ValueError("conclusion must not be empty.")
    if table_preview_rows < 1:
        raise ValueError("table_preview_rows must be positive.")

    has_figure = figure is not None
    if has_figure and not callable(getattr(figure, "savefig", None)):
        raise ValueError(
            "figure must be a Matplotlib Figure. "
            "Run the graph cell immediately before saving."
        )

    table_items: list[
        tuple[str, pd.DataFrame, pd.DataFrame]
    ] = []
    cleaned_table_titles: set[str] = set()
    tables = tables or {}
    table_formats = table_formats or {}
    unknown_format_tables = set(table_formats) - set(tables)
    if unknown_format_tables:
        raise KeyError(
            "table_formats contains unknown tables: "
            + ", ".join(sorted(unknown_format_tables))
        )
    for raw_title, frame in tables.items():
        table_title = _clean_single_line(raw_title, field="table title")
        if table_title in cleaned_table_titles:
            raise ValueError(f"Duplicate table title: {table_title!r}")
        cleaned_table_titles.add(table_title)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"Table {raw_title!r} must be a pandas DataFrame, "
                f"got {type(frame).__name__}."
            )
        raw_frame, formatted_frame = _prepare_table(
            frame,
            table_formats.get(raw_title),
        )
        table_items.append((table_title, raw_frame, formatted_frame))

    if not has_figure and not table_items:
        raise ValueError("Provide at least one figure or table to save.")

    findings_directory = root / "eda" / "findings"
    figures_directory = root / "assets" / "eda"
    findings_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)

    finding_id = _next_finding_id(findings_directory)
    note_relative = Path("eda") / "findings" / f"{finding_id}.md"
    figure_relative = (
        Path("assets") / "eda" / f"{finding_id}.png"
        if has_figure
        else None
    )
    note_path = root / note_relative
    figure_path = root / figure_relative if figure_relative else None
    if note_path.exists() or (figure_path is not None and figure_path.exists()):
        raise FileExistsError(f"Finding paths already exist for {finding_id}.")

    table_sections: list[str] = []
    table_paths: list[Path] = []
    csv_writes: list[tuple[Path, pd.DataFrame]] = []
    for table_number, (table_title, raw_frame, formatted_frame) in enumerate(
        table_items,
        start=1,
    ):
        preview = formatted_frame.head(table_preview_rows)
        details: list[str] = []
        if len(raw_frame) > table_preview_rows:
            csv_relative = (
                Path("assets")
                / "eda"
                / f"{finding_id}-table-{table_number:02d}.csv"
            )
            csv_path = root / csv_relative
            if csv_path.exists():
                raise FileExistsError(f"Table artifact already exists: {csv_path}")
            table_paths.append(csv_relative)
            csv_writes.append((csv_path, raw_frame))
            details.append(
                f"> Показаны первые {table_preview_rows} из {len(raw_frame)} "
                f"строк. Полная таблица: [[{csv_relative.as_posix()}]]."
            )
        else:
            details.append(f"> Строк в таблице: {len(raw_frame)}.")

        table_sections.append(
            f"### {table_title}\n\n"
            + "\n".join(details)
            + "\n\n"
            + dataframe_to_markdown(preview)
        )

    created = datetime.now().astimezone().isoformat(timespec="seconds")
    features_yaml = (
        "[" + ", ".join(_yaml_string(value) for value in feature_names) + "]"
    )
    notebook_path = Path(notebook).as_posix()
    source_link = f"[[{notebook_path}]]"
    question_text = question or "—"
    method_text = method or "—"
    hypothesis_text = hypothesis or "Гипотеза пока не сформулирована."

    artifact_sections: list[str] = []
    if figure_relative is not None:
        artifact_sections.append(
            "## График\n\n"
            f"![[{figure_relative.as_posix()}]]"
        )
    if table_sections:
        artifact_sections.append(
            "## Таблицы\n\n" + "\n\n".join(table_sections)
        )
    artifacts_markdown = "\n\n".join(artifact_sections)

    note = (
        "---\n"
        "type: eda-finding\n"
        f"id: {_yaml_string(finding_id)}\n"
        f"title: {_yaml_string(title)}\n"
        "status: observed\n"
        "stage: eda\n"
        f"created: {_yaml_string(created)}\n"
        f"source_notebook: {_yaml_string(source_link)}\n"
        f"features: {features_yaml}\n"
        "tags:\n"
        "  - ml/eda\n"
        "  - ml/finding\n"
        "---\n\n"
        f"# {finding_id} — {title}\n\n"
        "← [[eda/findings/_index.md|EDA-наблюдения]] · "
        "[[docs/02_eda.md|02 — EDA]]\n\n"
        "> [!abstract] Ключевой вывод\n"
        f"> {conclusion}\n\n"
        "## Исследовательский вопрос\n\n"
        f"{question_text}\n\n"
        "## Метод\n\n"
        f"{method_text}\n\n"
        f"{artifacts_markdown}\n\n"
        "## Возможная гипотеза\n\n"
        f"{hypothesis_text}\n\n"
        "## Связанные эксперименты\n\n"
        "<!-- auto:eda-experiment-links:start -->\n\n"
        "> Эксперименты на основании этого наблюдения пока не зарегистрированы.\n\n"
        "<!-- auto:eda-experiment-links:end -->\n\n"
        "## Источник\n\n"
        f"- Notebook: {source_link}\n"
    )

    temporary_figure = (
        figure_path.with_name(f".{figure_path.stem}.tmp{figure_path.suffix}")
        if figure_path is not None
        else None
    )
    temporary_csvs = [
        (
            csv_path.with_name(f".{csv_path.stem}.tmp{csv_path.suffix}"),
            csv_path,
            frame,
        )
        for csv_path, frame in csv_writes
    ]
    temporary_note = note_path.with_suffix(".md.tmp")
    try:
        if temporary_figure is not None:
            figure.savefig(
                temporary_figure,
                dpi=dpi,
                bbox_inches="tight",
                format="png",
            )
        for temporary_csv, _, frame in temporary_csvs:
            frame.to_csv(temporary_csv, index=False, encoding="utf-8")
        temporary_note.write_text(note, encoding="utf-8", newline="\n")
        if temporary_figure is not None and figure_path is not None:
            os.replace(temporary_figure, figure_path)
        for temporary_csv, csv_path, _ in temporary_csvs:
            os.replace(temporary_csv, csv_path)
        os.replace(temporary_note, note_path)
    finally:
        if temporary_figure is not None:
            temporary_figure.unlink(missing_ok=True)
        for temporary_csv, _, _ in temporary_csvs:
            temporary_csv.unlink(missing_ok=True)
        temporary_note.unlink(missing_ok=True)

    sync_eda_findings(root)
    return EdaFinding(
        finding_id=finding_id,
        title=title,
        note_path=note_relative,
        figure_path=figure_relative,
        table_paths=tuple(table_paths),
    )
