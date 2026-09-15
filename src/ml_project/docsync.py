"""Safe synchronization of generated notebook reports into Markdown blocks."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

import pandas as pd

from .data import DataCatalog
from .profiling import DatasetProfiler


class MarkdownDocument:
    """Update only explicitly marked generated blocks in a Markdown file."""

    _BLOCK_ID = re.compile(r"^[a-z0-9-]+$")

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def update_blocks(self, blocks: Mapping[str, str]) -> list[str]:
        with self.path.open("r", encoding="utf-8", newline="") as stream:
            text = stream.read()
        newline = "\r\n" if "\r\n" in text else "\n"

        updated: list[str] = []
        for block_id, content in blocks.items():
            if not self._BLOCK_ID.fullmatch(block_id):
                raise ValueError(f"Invalid generated block id: {block_id!r}")

            start = f"<!-- auto:{block_id}:start -->"
            end = f"<!-- auto:{block_id}:end -->"
            if text.count(start) != 1 or text.count(end) != 1:
                raise ValueError(
                    f"Expected exactly one marker pair for {block_id!r} "
                    f"in {self.path}."
                )

            start_index = text.index(start) + len(start)
            end_index = text.index(end)
            if start_index > end_index:
                raise ValueError(f"Markers are reversed for block {block_id!r}.")

            normalized = content.strip().replace("\r\n", "\n").replace("\r", "\n")
            normalized = normalized.replace("\n", newline)
            # Blank lines isolate generated Markdown from the surrounding HTML
            # comments. Obsidian Live Preview may otherwise treat a table placed
            # directly after a comment as plain source text.
            replacement = (newline * 2) + normalized + (newline * 2)
            text = text[:start_index] + replacement + text[end_index:]
            updated.append(block_id)

        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        try:
            os.replace(temporary, self.path)
        except PermissionError:
            # Obsidian and some Windows editors keep Markdown files open without
            # FILE_SHARE_DELETE. In that state the file is writable, but an
            # atomic replacement is rejected with WinError 5. Fall back to an
            # in-place write; if it also fails, keep the complete temporary file
            # beside the report so the generated content is not lost.
            try:
                with self.path.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(text)
            except OSError as fallback_error:
                raise PermissionError(
                    f"Cannot update {self.path}. Close it in external editors "
                    f"and retry; generated content remains in {temporary}."
                ) from fallback_error
            temporary.unlink(missing_ok=True)
        return updated


def dataframe_to_markdown(
    frame: pd.DataFrame,
    *,
    float_digits: int = 3,
    right_align: set[str] | None = None,
) -> str:
    """Render a padded Markdown table without requiring ``tabulate``.

    Padding keeps the table readable both in Obsidian's source editor and in
    Reading view. Numeric columns are right-aligned by default.
    """
    if frame.empty:
        return "> Нет строк для отображения."

    def format_value(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            value = f"{value:.{float_digits}f}"
        text = str(value)
        return text.replace("|", r"\|").replace("\r", " ").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    values = [
        [format_value(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(3, len(column), *(len(row[index]) for row in values))
        for index, column in enumerate(columns)
    ]

    inferred_right = {
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    }
    right_align = inferred_right | (right_align or set())

    def padded_row(row: list[str]) -> str:
        cells = [
            (
                value.rjust(widths[index])
                if columns[index] in right_align
                else value.ljust(widths[index])
            )
            for index, value in enumerate(row)
        ]
        return "| " + " | ".join(cells) + " |"

    separators = [
        ("-" * (width - 1) + ":") if column in right_align else ("-" * width)
        for column, width in zip(columns, widths)
    ]
    return "\n".join(
        [
            padded_row(columns),
            "| " + " | ".join(separators) + " |",
            *(padded_row(row) for row in values),
        ]
    )


def _translate_values(
    frame: pd.DataFrame,
    translations: Mapping[str, Mapping[object, object]],
) -> pd.DataFrame:
    """Return a copy with presentation-only value translations."""
    result = frame.copy()
    for column, mapping in translations.items():
        if column in result.columns:
            result[column] = result[column].replace(mapping)
    return result


def _schema_to_markdown(schema: pd.DataFrame) -> str:
    """Render one compact schema table per dataset instead of one wide table."""
    sections: list[str] = []
    for dataset, group in schema.groupby("Датасет", sort=False):
        compact = group.drop(columns=["Датасет"]).reset_index(drop=True)
        sections.append(
            f"### `{dataset}`\n\n"
            + dataframe_to_markdown(compact)
        )
    return "\n\n".join(sections)


def _versions_to_markdown(versions: pd.DataFrame) -> str:
    """Show short hashes in the table and keep full hashes in a disclosure."""
    compact = versions.copy()
    compact["SHA-256"] = compact["SHA-256"].map(
        lambda value: f"`{str(value)[:12]}…`"
    )
    full_hashes = "\n".join(
        f"{row['Датасет']}: {row['SHA-256']}"
        for _, row in versions.iterrows()
    )
    return (
        dataframe_to_markdown(compact)
        + "\n\n<details>\n"
        + "<summary>Полные SHA-256 для проверки воспроизводимости</summary>\n\n"
        + "```text\n"
        + full_hashes
        + "\n```\n\n"
        + "</details>"
    )


def build_field_descriptions_template(
    catalog: DataCatalog,
    field_descriptions: Mapping[str, str] | None = None,
) -> str:
    """Create a copy-ready ``FIELD_DESCRIPTIONS`` dictionary.

    Columns are discovered from the configured files in their original order.
    Existing descriptions are preserved, while new columns receive an empty
    string that can be filled manually.
    """
    field_descriptions = field_descriptions or {}
    fields = list(
        dict.fromkeys(
            str(column)
            for dataset in catalog.available_names()
            for column in catalog.load(dataset).columns
        )
    )
    rows = [
        f"    {field!r}: {str(field_descriptions.get(field, ''))!r},"
        for field in fields
    ]
    return "\n".join(["FIELD_DESCRIPTIONS = {", *rows, "}"])


def build_data_blocks(
    catalog: DataCatalog,
    *,
    key: str | None,
    target: str | None,
    inference_dataset: str | None,
    field_descriptions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    file_report = catalog.file_report()
    schema_report = catalog.schema_report(
        key=key,
        target=target,
        inference_dataset=inference_dataset,
        field_descriptions=field_descriptions,
    )
    versions = file_report[["dataset", "file", "sha256"]]

    file_report = file_report.drop(columns=["sha256"]).rename(
        columns={
            "dataset": "Датасет",
            "file": "Файл",
            "role": "Роль",
            "rows": "Строки",
            "columns": "Столбцы",
            "disk_kib": "Диск, KiB",
            "memory_mib": "RAM, MiB",
        }
    )
    schema_report = schema_report.rename(
        columns={
            "dataset": "Датасет",
            "field": "Поле",
            "dtype": "Тип",
            "description": "Смысл",
            "role": "Роль",
            "available_at_inference": "Доступно для прогноза",
        }
    )
    versions = versions.rename(
        columns={
            "dataset": "Датасет",
            "file": "Файл",
            "sha256": "SHA-256",
        }
    )
    file_report = _translate_values(
        file_report,
        {
            "Роль": {
                "train": "обучение",
                "inference": "прогноз",
                "submission_example": "пример submission",
                "unspecified": "не указана",
            }
        },
    )
    schema_report = _translate_values(
        schema_report,
        {
            "Роль": {
                "id": "идентификатор",
                "target": "целевая переменная",
                "feature": "признак",
                "output": "результат",
            },
            "Доступно для прогноза": {
                "yes": "да",
                "no": "нет",
                "not applicable": "не применимо",
            },
        },
    )

    return {
        "data-file-report": dataframe_to_markdown(file_report),
        "data-schema": _schema_to_markdown(schema_report),
        "data-versions": _versions_to_markdown(versions),
    }


def build_eda_blocks(
    catalog: DataCatalog,
    profiles: Mapping[str, DatasetProfiler],
    *,
    train_dataset: str,
) -> dict[str, str]:
    profile_names = set(profiles)
    snapshot = catalog.file_report()
    snapshot = snapshot[snapshot["dataset"].isin(profile_names)]
    snapshot = snapshot[
        ["dataset", "rows", "columns", "memory_mib"]
    ].rename(
        columns={
            "dataset": "Датасет",
            "rows": "Строки",
            "columns": "Столбцы",
            "memory_mib": "RAM, MiB",
        }
    )

    train_profile = profiles[train_dataset]
    try:
        target_report = train_profile.target_report().rename(
            columns={"count": "Количество", "share": "Доля"}
        )
        target_report["Доля"] = target_report["Доля"].map(lambda value: f"{value:.2%}")
        target = dataframe_to_markdown(
            target_report,
            right_align={"Количество", "Доля"},
        )
    except (KeyError, ValueError) as error:
        target = f"> Target report недоступен: {error}"

    quality = pd.concat(
        [profile.duplicate_report() for profile in profiles.values()],
        ignore_index=True,
    ).rename(
        columns={
            "dataset": "Датасет",
            "rows": "Строки",
            "full_row_duplicates": "Полные дубликаты",
            "key": "Ключ",
            "key_missing": "Пропуски ключа",
            "key_duplicates": "Дубликаты ключа",
            "key_unique": "Ключ уникален",
        }
    )
    quality = _translate_values(
        quality,
        {"Ключ уникален": {True: "да", False: "нет"}},
    )
    missing = pd.concat(
        [profile.missing_report() for profile in profiles.values()],
        ignore_index=True,
    ).rename(
        columns={
            "dataset": "Датасет",
            "field": "Поле",
            "missing": "Пропуски",
            "missing_share": "Доля пропусков",
        }
    )
    if not missing.empty:
        missing["Доля пропусков"] = missing["Доля пропусков"].map(
            lambda value: f"{value:.2%}"
        )
    quality_block = (
        "### Дубликаты и ключи\n\n"
        + dataframe_to_markdown(quality)
        + "\n\n### Пропуски\n\n"
        + dataframe_to_markdown(
            missing,
            right_align={"Пропуски", "Доля пропусков"},
        )
    )

    return {
        "eda-snapshot": dataframe_to_markdown(snapshot),
        "eda-target": target,
        "eda-quality": quality_block,
    }
