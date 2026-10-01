"""Единый, воспроизводимый data contract для экспериментов EPIC.

Модуль отвечает только за неизменяемую часть всех экспериментов:

* чтение whitelist/template и проверку BED-координат;
* объединение двух csRNA-повторностей в pooled target;
* фиксированное разделение целых contig на train/validation/test;
* сохранение и проверку компактных split-артефактов;
* потоковую выдачу position-strand без огромной таблицы в памяти.

Признаки, модель, sampling train и метрики относятся к конкретному эксперименту
и намеренно не находятся в этом модуле.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")
KEY_COLUMNS = ("contig", "coordinate_0based", "strand")


@dataclass(frozen=True)
class SplitConfig:
    """Версия фиксированного разбиения, общая для сравнимых экспериментов."""

    assembly: str = "jaNemVect1.1"
    split_version: str = "contig_holdout_v1"
    seed: int = 42
    validation_contig_fraction: float = 0.25


DEFAULT_SPLIT_CONFIG = SplitConfig()


@dataclass(frozen=True)
class EpicPaths:
    """Все пути, которые использует общий data contract."""

    project_root: Path
    assembly: str
    source: Path
    participants: Path
    output: Path
    inputs: Mapping[str, Path]


@dataclass(frozen=True)
class PreparedSplit:
    """Компактное полное разбиение и ненулевые train/validation targets."""

    config: SplitConfig
    paths: EpicPaths
    contig_assignment: pd.DataFrame
    summary: pd.DataFrame
    intervals_by_split: Mapping[str, pd.DataFrame]
    positive_targets_by_split: Mapping[str, pd.DataFrame | None]
    manifest: Mapping[str, object]

    def intervals(self, split: str) -> pd.DataFrame:
        _validate_split_name(split)
        return self.intervals_by_split[split]

    def positive_targets(self, split: str) -> pd.DataFrame | None:
        _validate_split_name(split)
        return self.positive_targets_by_split[split]

    def iter_batches(
        self,
        split: str,
        *,
        batch_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """Выдать все position-strand части небольшими порциями."""

        targets = self.positive_targets(split)
        yield from iter_position_batches(
            self.intervals(split),
            targets,
            batch_size=batch_size,
        )


def find_project_root(start: Path | str | None = None) -> Path:
    """Найти корень проекта из notebook, теста или команды в терминале."""

    current = Path(start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        if (
            (candidate / "README.md").is_file()
            and (candidate / "src/ml_project").is_dir()
            and (candidate / "data/raw").is_dir()
        ):
            return candidate
    raise FileNotFoundError(f"Не найден корень EPIC над {current}")


def _discover_filtered_replicate_pair(
    participants: Path,
) -> tuple[Path, Path]:
    # Найти единственную filtered пару csRNA r1/r2 для assembly.
    r1_candidates = sorted(
        path
        for path in participants.glob("csRNA*-r1-train.bed*")
        if "unfiltered" not in path.name.lower()
    )
    r2_candidates = sorted(
        path
        for path in participants.glob("csRNA*-r2-train.bed*")
        if "unfiltered" not in path.name.lower()
    )
    r2_by_key = {
        path.name.replace("-r2-train", "-rX-train"): path
        for path in r2_candidates
    }
    pairs = []
    for r1_path in r1_candidates:
        key = r1_path.name.replace("-r1-train", "-rX-train")
        r2_path = r2_by_key.get(key)
        if r2_path is not None:
            pairs.append((r1_path, r2_path))

    if len(pairs) != 1:
        details = ", ".join(
            f"{first.name} + {second.name}" for first, second in pairs
        ) or "no matched filtered pairs"
        raise FileNotFoundError(
            "Ожидалась ровно одна filtered csRNA r1/r2 пара в "
            f"{participants}, найдено {len(pairs)}: {details}"
        )
    return pairs[0]


def paths_for(
    project_root: Path | str,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> EpicPaths:
    """Построить пути без чтения данных."""

    root = Path(project_root).resolve()
    source = root / "data/raw/unpacked" / config.assembly
    participants = source / "for_participants"
    output = (
        root
        / "data/processed"
        / config.assembly
        / "splits"
        / config.split_version
    )
    csrna_r1, csrna_r2 = _discover_filtered_replicate_pair(participants)
    inputs = {
        "fai": source / "genome/genome.fa.fai",
        "whitelist_train": participants / "whitelist.train.bed.gz",
        "whitelist_test": participants / "whitelist.test.bed.gz",
        "template_train": participants / "template.train.bed.gz",
        "template_test": participants / "template.test.bed.gz",
        "csrna_r1": csrna_r1,
        "csrna_r2": csrna_r2,
    }
    return EpicPaths(root, config.assembly, source, participants, output, inputs)


def sha256_file(path: Path | str) -> str:
    """Потоковый SHA-256 файла."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_split_name(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split должен быть одним из {SPLITS}, получено {split!r}")


def _require_inputs(paths: EpicPaths) -> None:
    missing = [str(path) for path in paths.inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Не найдены распакованные входные файлы: " + ", ".join(missing)
        )


def _read_contig_lengths(path: Path) -> dict[str, int]:
    fai = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["contig", "length", "offset", "line_bases", "line_bytes"],
    )
    if fai.empty or not fai["contig"].is_unique:
        raise ValueError("FAI пуст или содержит повторяющиеся contig")
    return dict(zip(fai["contig"].astype(str), fai["length"].astype("int64")))


def read_whitelist(
    path: Path | str,
    contig_lengths: Mapping[str, int],
) -> pd.DataFrame:
    """Прочитать BED3 и проверить границы, порядок и непересечение."""

    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["contig", "start", "end"],
        dtype={"contig": str, "start": "int64", "end": "int64"},
    )
    if frame.empty:
        raise ValueError("Whitelist пуст")
    if not set(frame["contig"]).issubset(contig_lengths):
        raise ValueError("В whitelist есть неизвестный contig")
    if (frame["start"] < 0).any() or (frame["end"] <= frame["start"]).any():
        raise ValueError("Неверные границы BED-интервала")
    if (frame["end"] > frame["contig"].map(contig_lengths)).any():
        raise ValueError("Whitelist-интервал выходит за границу contig")
    for contig, group in frame.groupby("contig", sort=False):
        starts = group["start"].to_numpy()
        ends = group["end"].to_numpy()
        if np.any(starts[1:] < ends[:-1]):
            raise ValueError(
                f"Whitelist-интервалы пересекаются или не упорядочены: {contig}"
            )
    frame["length_bp"] = frame["end"] - frame["start"]
    frame["whitelist_offset"] = (
        frame["length_bp"].cumsum() - frame["length_bp"]
    )
    return frame


def read_template(
    path: Path | str,
    whitelist: pd.DataFrame,
    source_split: str,
) -> pd.DataFrame:
    """Прочитать BED6 и доказать соответствие whitelist: сначала +, затем -."""

    if source_split not in {"train", "test"}:
        raise ValueError("source_split должен быть train или test")
    bed = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["contig", "start", "end", "name", "score", "strand"],
        dtype={"contig": str, "start": "int64", "end": "int64", "strand": str},
    )
    expected = pd.concat(
        [whitelist.assign(strand="+"), whitelist.assign(strand="-")],
        ignore_index=True,
    )
    columns = ["contig", "start", "end", "strand"]
    pd.testing.assert_frame_equal(bed[columns], expected[columns])

    result = bed[columns].copy()
    result["length_bp"] = result["end"] - result["start"]
    result["template_stop"] = result["length_bp"].cumsum()
    result["template_start"] = (
        result["template_stop"] - result["length_bp"]
    )
    result["source_split"] = source_split
    return result


def assign_contigs(
    train_names,
    test_names,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Назначить целые contig независимо от target и порядка входных строк."""

    train_names = sorted(set(train_names))
    test_names = sorted(set(test_names))
    if set(train_names) & set(test_names):
        raise ValueError("Официальные train и test имеют общие contig")
    if not 0 < validation_fraction < 1:
        raise ValueError("Доля validation должна быть между 0 и 1")
    n_validation = math.ceil(len(train_names) * validation_fraction)
    if n_validation >= len(train_names) or not test_names:
        raise ValueError("Train, validation и test должны быть непустыми")

    rng = np.random.default_rng(seed)
    validation = set(rng.permutation(train_names)[:n_validation])
    records = [
        {
            "contig": name,
            "split": "validation" if name in validation else "train",
            "source_split": "train",
        }
        for name in train_names
    ]
    records.extend(
        {"contig": name, "split": "test", "source_split": "test"}
        for name in test_names
    )
    return pd.DataFrame(records)


def read_sparse_signal(
    path: Path | str,
    whitelist: pd.DataFrame,
    count_column: str,
) -> pd.DataFrame:
    """Раскрыть ненулевой BED6 до отдельных position-strand."""

    records: list[tuple[str, int, str, int]] = []
    with gzip.open(path, "rt", encoding="ascii") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 6:
                raise ValueError("Ожидалось 6 полей BED")
            contig, start, end, _name, count_text, strand = fields
            start, end = int(start), int(end)
            try:
                count_decimal = Decimal(count_text)
            except InvalidOperation as error:
                raise ValueError(
                    f"Неверный csRNA count: {count_text!r}"
                ) from error
            if (
                not count_decimal.is_finite()
                or count_decimal <= 0
                or count_decimal != count_decimal.to_integral_value()
            ):
                raise ValueError(
                    f"csRNA count должен быть положительным целым: {count_text!r}"
                )
            count = int(count_decimal)
            if end <= start or strand not in {"+", "-"}:
                raise ValueError("Неверный сигнальный BED-интервал")
            records.extend(
                (contig, coordinate, strand, count)
                for coordinate in range(start, end)
            )

    signal = pd.DataFrame(
        records,
        columns=[*KEY_COLUMNS, count_column],
    ).astype({"coordinate_0based": "int64", count_column: "int64"})
    if signal.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Повтор position-strand внутри одной повторности")

    for contig, group in signal.groupby("contig", sort=False):
        allowed = whitelist.loc[whitelist["contig"] == contig]
        if allowed.empty:
            raise ValueError(f"Сигнал на неизвестном train-contig: {contig}")
        coordinates = group["coordinate_0based"].to_numpy()
        starts = allowed["start"].to_numpy()
        ends = allowed["end"].to_numpy()
        indices = np.searchsorted(starts, coordinates, side="right") - 1
        if np.any(indices < 0) or np.any(coordinates >= ends[np.maximum(indices, 0)]):
            raise ValueError("Сигнал находится вне train-whitelist")
    return signal


def pool_targets(
    first: pd.DataFrame,
    second: pd.DataFrame,
    whitelist: pd.DataFrame,
) -> pd.DataFrame:
    """Сложить r1+r2 и сопоставить с исходным train template index."""

    merged = first.merge(
        second,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
    )
    for column in ("r1_count", "r2_count"):
        merged[column] = merged[column].fillna(0).astype("int64")
    merged["count"] = merged["r1_count"] + merged["r2_count"]
    merged["target"] = (merged["count"] > 0).astype("int8")
    merged["template_index"] = np.int64(-1)

    plus_length = int(whitelist["length_bp"].sum())
    for contig, group in merged.groupby("contig", sort=False):
        allowed = whitelist.loc[whitelist["contig"] == contig]
        coordinates = group["coordinate_0based"].to_numpy()
        starts = allowed["start"].to_numpy()
        row = np.searchsorted(starts, coordinates, side="right") - 1
        offsets = allowed["whitelist_offset"].to_numpy()[row]
        indices = offsets + coordinates - starts[row]
        indices += (group["strand"].to_numpy() == "-") * plus_length
        merged.loc[group.index, "template_index"] = indices

    if not merged["template_index"].is_unique:
        raise ValueError("Позиции имеют совпадающий template_index")
    return merged.sort_values("template_index").reset_index(drop=True)


def iter_position_batches(
    interval_table: pd.DataFrame,
    positive_table: pd.DataFrame | None = None,
    *,
    batch_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Развернуть полное множество позиций по частям, восстанавливая нули."""

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size должен быть положительным целым числом")
    if interval_table.empty or interval_table["split"].nunique() != 1:
        raise ValueError("Передайте одну непустую часть разбиения")

    split = str(interval_table["split"].iloc[0])
    _validate_split_name(split)
    source_split = str(interval_table["source_split"].iloc[0])
    if split == "test":
        if positive_table is not None:
            raise ValueError("Test-ответы генератору не передаются")
    elif positive_table is None:
        raise ValueError("Для train/validation нужна таблица положительных ответов")

    lengths = interval_table["length_bp"].to_numpy(dtype=np.int64)
    local_stops = np.cumsum(lengths)
    local_starts = local_stops - lengths
    contigs = interval_table["contig"].to_numpy()
    starts = interval_table["start"].to_numpy(dtype=np.int64)
    strands = interval_table["strand"].to_numpy()
    original_offsets = interval_table["template_start"].to_numpy(dtype=np.int64)

    if split != "test":
        lookup = positive_table.set_index("template_index")["count"]
        if not lookup.index.is_unique:
            raise ValueError("Неуникальный template_index в targets")

    for first in range(0, int(local_stops[-1]), batch_size):
        local_index = np.arange(
            first,
            min(first + batch_size, int(local_stops[-1])),
            dtype=np.int64,
        )
        row = np.searchsorted(local_stops, local_index, side="right")
        within_interval = local_index - local_starts[row]
        original_index = original_offsets[row] + within_interval
        batch = pd.DataFrame(
            {
                "source_split": source_split,
                "template_index": original_index,
                "contig": contigs[row],
                "coordinate_0based": starts[row] + within_interval,
                "strand": strands[row],
                "split": split,
            }
        )
        if split != "test":
            batch["count"] = lookup.reindex(
                original_index,
                fill_value=0,
            ).to_numpy(dtype=np.int64)
            batch["target"] = (batch["count"] > 0).astype("int8")
        yield batch


def _split_summary(
    intervals_by_split: Mapping[str, pd.DataFrame],
    positive_targets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        frame = intervals_by_split[split]
        rows.append(
            {
                "split": split,
                "contigs": frame["contig"].nunique(),
                "intervals_both_strands": len(frame),
                "whitelist_bp": int(frame["length_bp"].sum()) // 2,
                "position_strand_count": int(frame["length_bp"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["percent_of_all_positions"] = (
        100
        * summary["position_strand_count"]
        / summary["position_strand_count"].sum()
    )
    counts = positive_targets.groupby("split").size()
    summary["positive_positions"] = summary["split"].map(counts).astype("Int64")
    summary["negative_positions"] = (
        summary["position_strand_count"] - summary["positive_positions"]
    )
    summary["positive_percent"] = (
        100 * summary["positive_positions"] / summary["position_strand_count"]
    )
    return summary


def _audit_complete_split(
    official_train: pd.DataFrame,
    official_test: pd.DataFrame,
    train_template: pd.DataFrame,
    test_template: pd.DataFrame,
    assignment: pd.DataFrame,
    intervals: Mapping[str, pd.DataFrame],
) -> None:
    sets = {split: set(intervals[split]["contig"]) for split in SPLITS}
    if any(sets[left] & sets[right] for left, right in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    )):
        raise AssertionError("Части имеют общие contig")
    if sets["train"] | sets["validation"] != set(official_train["contig"]):
        raise AssertionError("Локальные train+validation не покрывают официальный train")
    if sets["test"] != set(official_test["contig"]):
        raise AssertionError("Локальный test отличается от официального test")
    if not assignment["contig"].is_unique:
        raise AssertionError("Contig назначен больше одного раза")

    rebuilt = pd.concat([intervals["train"], intervals["validation"]])
    rebuilt = rebuilt.sort_values("template_start").reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt[train_template.columns], train_template)
    pd.testing.assert_frame_equal(
        intervals["test"][test_template.columns].reset_index(drop=True),
        test_template,
    )
    for frame in intervals.values():
        if set(frame["strand"]) != {"+", "-"}:
            raise AssertionError("В части отсутствует одна из цепей")
        plus = frame.loc[frame["strand"] == "+", ["contig", "start", "end"]]
        minus = frame.loc[frame["strand"] == "-", ["contig", "start", "end"]]
        pd.testing.assert_frame_equal(
            plus.reset_index(drop=True),
            minus.reset_index(drop=True),
        )


def build_split(
    project_root: Path | str,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> PreparedSplit:
    """Пересобрать общий split из raw-файлов без записи на диск."""

    paths = paths_for(project_root, config)
    _require_inputs(paths)
    contig_lengths = _read_contig_lengths(paths.inputs["fai"])
    official_train = read_whitelist(
        paths.inputs["whitelist_train"],
        contig_lengths,
    )
    official_test = read_whitelist(
        paths.inputs["whitelist_test"],
        contig_lengths,
    )
    train_template = read_template(
        paths.inputs["template_train"],
        official_train,
        "train",
    )
    test_template = read_template(
        paths.inputs["template_test"],
        official_test,
        "test",
    )

    assignment = assign_contigs(
        official_train["contig"],
        official_test["contig"],
        config.validation_contig_fraction,
        config.seed,
    )
    contig_to_split = assignment.set_index("contig")["split"]
    all_intervals = pd.concat([train_template, test_template], ignore_index=True)
    all_intervals["split"] = all_intervals["contig"].map(contig_to_split)
    if all_intervals["split"].isna().any():
        raise AssertionError("Есть interval без назначения split")
    intervals = {
        split: all_intervals.loc[all_intervals["split"] == split].copy()
        for split in SPLITS
    }
    _audit_complete_split(
        official_train,
        official_test,
        train_template,
        test_template,
        assignment,
        intervals,
    )

    r1 = read_sparse_signal(paths.inputs["csrna_r1"], official_train, "r1_count")
    r2 = read_sparse_signal(paths.inputs["csrna_r2"], official_train, "r2_count")
    pooled = pool_targets(r1, r2, official_train)
    pooled["source_split"] = "train"
    pooled["split"] = pooled["contig"].map(contig_to_split)
    if set(pooled["split"]) != {"train", "validation"}:
        raise AssertionError("Train target попал за пределы train/validation")
    targets = {
        "train": pooled.loc[pooled["split"] == "train"].copy(),
        "validation": pooled.loc[pooled["split"] == "validation"].copy(),
        "test": None,
    }
    summary = _split_summary(intervals, pooled)
    if config.assembly == "jaNemVect1.1":
        if len(official_train) != 114_692 or len(official_test) != 27_423:
            raise AssertionError("Неожиданный размер whitelist jaNemVect1.1")
        if int(train_template["length_bp"].sum()) != 311_871_646:
            raise AssertionError("Неожиданный train template jaNemVect1.1")
        if int(test_template["length_bp"].sum()) != 72_613_390:
            raise AssertionError("Неожиданный test template jaNemVect1.1")
        if len(pooled) != 226_648:
            raise AssertionError("Неожиданное число положительных jaNemVect1.1")

    source_hashes = {
        path.relative_to(paths.project_root).as_posix(): sha256_file(path)
        for path in paths.inputs.values()
    }
    manifest: dict[str, object] = {
        "assembly": config.assembly,
        "split_version": config.split_version,
        "seed": config.seed,
        "validation_fraction_of_official_train_contigs": (
            config.validation_contig_fraction
        ),
        "rule": (
            "sorted official train contigs; numpy.default_rng(seed).permutation; "
            "first ceil(n*fraction) to validation"
        ),
        "group": "whole contig, both strands and both replicates together",
        "test_policy": "official test unchanged; no test labels read by this notebook",
        "validation_policy": (
            "development holdout; official train already explored in prior EDA"
        ),
        "target": "count = csRNA-r1 + csRNA-r2; target = int(count > 0)",
        "negative_policy": (
            "all whitelist positions retained; absent sparse signal means zero "
            "only for train/validation"
        ),
        "template_index": (
            "0-based in ORIGINAL source_split TXT; end offsets exclusive"
        ),
        "assignment": assignment.to_dict(orient="records"),
        "source_sha256": source_hashes,
    }
    return PreparedSplit(
        config=config,
        paths=paths,
        contig_assignment=assignment,
        summary=summary,
        intervals_by_split=intervals,
        positive_targets_by_split=targets,
        manifest=manifest,
    )


def _table_payload(table: pd.DataFrame, *, compressed: bool) -> bytes:
    data = table.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")
    return gzip.compress(data, mtime=0) if compressed else data


def save_split(bundle: PreparedSplit) -> Path:
    """Сохранить split детерминированно, не перезаписывая иной результат."""

    output = bundle.paths.output
    tables = {
        "contig_assignment.csv": bundle.contig_assignment,
        "split_summary.csv": bundle.summary,
        "train_intervals.csv.gz": bundle.intervals("train"),
        "validation_intervals.csv.gz": bundle.intervals("validation"),
        "test_intervals.csv.gz": bundle.intervals("test"),
        "train_positive_targets.csv.gz": bundle.positive_targets("train"),
        "validation_positive_targets.csv.gz": bundle.positive_targets("validation"),
    }
    payloads = {
        filename: _table_payload(table, compressed=filename.endswith(".gz"))
        for filename, table in tables.items()
    }
    manifest = dict(bundle.manifest)
    manifest["artifact_sha256"] = {
        filename: hashlib.sha256(data).hexdigest()
        for filename, data in payloads.items()
    }
    payloads["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")

    for filename, data in payloads.items():
        path = output / filename
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(
                f"Существующий split отличается: {path}. "
                "Для нового правила задайте новую split_version."
            )
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in payloads.items():
        path = output / filename
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
    return output


def load_split(
    project_root: Path | str,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
    *,
    verify_hashes: bool = True,
) -> PreparedSplit:
    """Загрузить сохранённый split и проверить raw/derived SHA-256."""

    paths = paths_for(project_root, config)
    manifest_path = paths.output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split ещё не подготовлен: {manifest_path}. Вызовите prepare_split()."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_config = {
        "assembly": config.assembly,
        "split_version": config.split_version,
        "seed": config.seed,
        "validation_fraction_of_official_train_contigs": (
            config.validation_contig_fraction
        ),
    }
    for key, expected in expected_config.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Manifest {key}={manifest.get(key)!r}, ожидалось {expected!r}"
            )
    if verify_hashes:
        for relative, expected in manifest["source_sha256"].items():
            path = paths.project_root / relative
            if sha256_file(path) != expected:
                raise ValueError(f"Исходный файл изменился: {path}")
        for filename, expected in manifest["artifact_sha256"].items():
            path = paths.output / filename
            if sha256_file(path) != expected:
                raise ValueError(f"Split-артефакт изменился: {path}")

    read = lambda name: pd.read_csv(paths.output / name)
    assignment = read("contig_assignment.csv")
    summary = read("split_summary.csv")
    intervals = {
        split: read(f"{split}_intervals.csv.gz") for split in SPLITS
    }
    targets: dict[str, pd.DataFrame | None] = {
        "train": read("train_positive_targets.csv.gz"),
        "validation": read("validation_positive_targets.csv.gz"),
        "test": None,
    }
    return PreparedSplit(
        config=config,
        paths=paths,
        contig_assignment=assignment,
        summary=summary,
        intervals_by_split=intervals,
        positive_targets_by_split=targets,
        manifest=manifest,
    )


def prepare_split(
    project_root: Path | str,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> PreparedSplit:
    """Собрать, сохранить и повторно проверить единый split."""

    built = build_split(project_root, config)
    save_split(built)
    return load_split(project_root, config, verify_hashes=True)


def ensure_prepared_split(
    project_root: Path | str | None = None,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> PreparedSplit:
    """Загрузить существующий split или один раз подготовить его из raw."""

    root = find_project_root(project_root)
    manifest = paths_for(root, config).output / "manifest.json"
    return (
        load_split(root, config, verify_hashes=True)
        if manifest.is_file()
        else prepare_split(root, config)
    )


__all__ = [
    "DEFAULT_SPLIT_CONFIG",
    "EpicPaths",
    "KEY_COLUMNS",
    "PreparedSplit",
    "SPLITS",
    "SplitConfig",
    "assign_contigs",
    "build_split",
    "ensure_prepared_split",
    "find_project_root",
    "iter_position_batches",
    "load_split",
    "paths_for",
    "pool_targets",
    "prepare_split",
    "read_sparse_signal",
    "read_template",
    "read_whitelist",
    "save_split",
    "sha256_file",
]
