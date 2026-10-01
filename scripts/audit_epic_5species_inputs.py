from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


ASSEMBLIES = [
    "ilPloInte3.2",
    "cgigas_uk_roslin_v1",
    "Ofas_2.0",
    "ASM119413v2",
    "sSquSuc5.hap2",
]


def project_root(start: Path | str | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "README.md").is_file()
            and (candidate / "src/ml_project").is_dir()
            and (candidate / "data/raw/unpacked").is_dir()
        ):
            return candidate
    raise FileNotFoundError(f"EPIC project root not found above {current}")


def read_bed3(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["contig", "start", "end"],
        dtype={"contig": str, "start": "int64", "end": "int64"},
        compression="infer",
    )


def read_bed6(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["contig", "start", "end", "name", "score", "strand"],
        dtype={"contig": str, "start": "int64", "end": "int64", "strand": str},
        compression="infer",
    )


def validate_whitelist(frame: pd.DataFrame, contig_lengths: dict[str, int], label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label}: empty whitelist")
    if not set(frame["contig"]).issubset(contig_lengths):
        unknown = sorted(set(frame["contig"]) - set(contig_lengths))[:10]
        raise ValueError(f"{label}: unknown contigs {unknown}")
    if (frame["start"] < 0).any() or (frame["end"] <= frame["start"]).any():
        raise ValueError(f"{label}: invalid BED bounds")
    max_len = frame["contig"].map(contig_lengths)
    if (frame["end"] > max_len).any():
        raise ValueError(f"{label}: interval outside contig")
    ordered = frame.sort_values(["contig", "start", "end"], kind="stable")
    for contig, group in ordered.groupby("contig", sort=False):
        starts = group["start"].to_numpy(np.int64)
        ends = group["end"].to_numpy(np.int64)
        if len(group) > 1 and np.any(starts[1:] < ends[:-1]):
            raise ValueError(f"{label}: overlapping intervals on {contig}")


def validate_template(template: pd.DataFrame, whitelist: pd.DataFrame, label: str) -> dict:
    if template.empty:
        raise ValueError(f"{label}: empty template")
    if not set(template["strand"]) == {"+", "-"}:
        raise ValueError(f"{label}: expected both strands")
    expected = pd.concat(
        [
            whitelist.assign(strand="+"),
            whitelist.assign(strand="-"),
        ],
        ignore_index=True,
    )
    cols = ["contig", "start", "end", "strand"]
    left = template[cols].reset_index(drop=True)
    right = expected[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    lengths = template["end"].to_numpy(np.int64) - template["start"].to_numpy(np.int64)
    return {
        "template_rows": int(len(template)),
        "position_strand_count": int(lengths.sum()),
    }


def load_fai(path: Path) -> dict[str, int]:
    fai = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["contig", "length", "offset", "line_bases", "line_bytes"],
    )
    if fai.empty or not fai["contig"].is_unique:
        raise ValueError(f"Bad FAI: {path}")
    return dict(zip(fai["contig"].astype(str), fai["length"].astype("int64")))


def choose_single(candidates: list[Path], label: str, required: bool = True) -> Path | None:
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and not required:
        return None
    raise RuntimeError(
        f"{label}: expected exactly one file, found {len(candidates)}: "
        + ", ".join(str(x.name) for x in candidates)
    )


def identify_files(root: Path, assembly: str) -> dict[str, Path | None | list[Path]]:
    source = root / "data/raw/unpacked" / assembly
    participants = source / "for_participants"
    genome = source / "genome"
    if not participants.is_dir():
        raise FileNotFoundError(participants)
    if not genome.is_dir():
        raise FileNotFoundError(genome)

    fasta_candidates = [
        p for p in [genome / "genome.fa.gz", genome / "genome.fa"]
        if p.is_file()
    ]
    fasta = choose_single(fasta_candidates, f"{assembly} FASTA")
    fai = choose_single(
        [p for p in [genome / "genome.fa.fai", genome / "genome.fa.gz.fai"] if p.is_file()],
        f"{assembly} FAI",
    )

    whitelist_train = choose_single(
        list(participants.glob("whitelist.train.bed*")),
        f"{assembly} whitelist.train",
    )
    whitelist_test = choose_single(
        list(participants.glob("whitelist.test.bed*")),
        f"{assembly} whitelist.test",
    )
    template_train = choose_single(
        list(participants.glob("template.train.bed*")),
        f"{assembly} template.train BED",
    )
    template_test = choose_single(
        list(participants.glob("template.test.bed*")),
        f"{assembly} template.test BED",
    )

    # Filtered sparse csRNA replicate BEDs only; explicitly exclude *unfiltered*.
    r1 = sorted(
        p for p in participants.glob("csRNA*-r1-train.bed*")
        if "unfiltered" not in p.name.lower()
    )
    r2 = sorted(
        p for p in participants.glob("csRNA*-r2-train.bed*")
        if "unfiltered" not in p.name.lower()
    )

    # Match r1/r2 by replacing the literal replicate token.
    pairs = []
    r2_by_key = {
        p.name.replace("-r2-train", "-rX-train"): p for p in r2
    }
    for p1 in r1:
        key = p1.name.replace("-r1-train", "-rX-train")
        p2 = r2_by_key.get(key)
        if p2 is not None:
            pairs.append((p1, p2))

    dense_train = sorted(participants.glob("csRNA-train.txt*"))
    dense_train_path = choose_single(
        dense_train,
        f"{assembly} pooled csRNA-train.txt",
        required=False,
    )

    return {
        "source": source,
        "participants": participants,
        "fasta": fasta,
        "fai": fai,
        "whitelist_train": whitelist_train,
        "whitelist_test": whitelist_test,
        "template_train": template_train,
        "template_test": template_test,
        "r1_candidates": r1,
        "r2_candidates": r2,
        "replicate_pairs": pairs,
        "dense_train": dense_train_path,
    }


def interval_contains(
    whitelist_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    contig: str,
    start: int,
    end: int,
) -> bool:
    item = whitelist_lookup.get(contig)
    if item is None:
        return False
    starts, ends = item
    idx = int(np.searchsorted(starts, start, side="right") - 1)
    return idx >= 0 and start >= int(starts[idx]) and end <= int(ends[idx])


def sparse_bed_stats(path: Path, whitelist: pd.DataFrame) -> dict:
    ordered = whitelist.sort_values(["contig", "start"], kind="stable")
    lookup = {
        str(contig): (
            group["start"].to_numpy(np.int64),
            group["end"].to_numpy(np.int64),
        )
        for contig, group in ordered.groupby("contig", sort=False)
    }
    rows = 0
    positive_position_strands = 0
    total_count_mass = 0
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="ascii") as stream:
        for line_no, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"{path.name}:{line_no}: expected BED6")
            contig, start, end, _name, count, strand = fields
            start, end = int(start), int(end)
            count_value = float(count)

            if not count_value.is_integer():
                raise ValueError(
                    f"{path.name}:{line_no}: non-integer csRNA count {count!r}"
                )

            count = int(count_value)

            if end <= start or count <= 0 or strand not in {"+", "-"}:
                raise ValueError(f"{path.name}:{line_no}: invalid signal row")
            if not interval_contains(lookup, contig, start, end):
                raise ValueError(
                    f"{path.name}:{line_no}: signal interval outside train whitelist"
                )
            width = end - start
            rows += 1
            positive_position_strands += width
            total_count_mass += width * count
    return {
        "rows": rows,
        "positive_position_strands": positive_position_strands,
        "total_count_mass": total_count_mass,
    }


def rel(root: Path, path: Path | None) -> str | None:
    return None if path is None else path.relative_to(root).as_posix()


def audit_assembly(root: Path, assembly: str) -> tuple[dict, dict]:
    files = identify_files(root, assembly)
    contig_lengths = load_fai(files["fai"])

    w_train = read_bed3(files["whitelist_train"])
    w_test = read_bed3(files["whitelist_test"])
    validate_whitelist(w_train, contig_lengths, f"{assembly} train whitelist")
    validate_whitelist(w_test, contig_lengths, f"{assembly} test whitelist")

    train_contigs = set(w_train["contig"].astype(str))
    test_contigs = set(w_test["contig"].astype(str))
    overlap = sorted(train_contigs & test_contigs)
    if overlap:
        raise ValueError(f"{assembly}: official train/test contigs overlap: {overlap[:10]}")

    t_train = read_bed6(files["template_train"])
    t_test = read_bed6(files["template_test"])
    train_contract = validate_template(t_train, w_train, f"{assembly} template.train")
    test_contract = validate_template(t_test, w_test, f"{assembly} template.test")

    pairs = files["replicate_pairs"]
    if len(pairs) != 1:
        raise RuntimeError(
            f"{assembly}: expected one filtered csRNA r1/r2 pair, found {len(pairs)}: "
            + ", ".join(f"{a.name} + {b.name}" for a, b in pairs)
        )
    r1, r2 = pairs[0]
    r1_stats = sparse_bed_stats(r1, w_train)
    r2_stats = sparse_bed_stats(r2, w_train)

    expected_current_r1 = files["participants"] / "csRNA-r1-train.bed.gz"
    expected_current_r2 = files["participants"] / "csRNA-r2-train.bed.gz"
    current_epic_data_compatible = expected_current_r1.is_file() and expected_current_r2.is_file()

    row = {
        "assembly": assembly,
        "fasta": rel(root, files["fasta"]),
        "fai": rel(root, files["fai"]),
        "train_contigs": len(train_contigs),
        "test_contigs": len(test_contigs),
        "train_whitelist_intervals": len(w_train),
        "test_whitelist_intervals": len(w_test),
        "train_position_strands": train_contract["position_strand_count"],
        "test_position_strands": test_contract["position_strand_count"],
        "r1_file": rel(root, r1),
        "r2_file": rel(root, r2),
        "r1_sparse_positive_positions": r1_stats["positive_position_strands"],
        "r2_sparse_positive_positions": r2_stats["positive_position_strands"],
        "r1_total_count_mass": r1_stats["total_count_mass"],
        "r2_total_count_mass": r2_stats["total_count_mass"],
        "pooled_dense_train_file": rel(root, files["dense_train"]),
        "current_epic_data_hardcoded_paths_work": current_epic_data_compatible,
        "status": "PASS",
    }

    mapping = {
        "assembly": assembly,
        "source_dir": rel(root, files["source"]),
        "genome": {
            "fasta": rel(root, files["fasta"]),
            "fai": rel(root, files["fai"]),
        },
        "participants": {
            "whitelist_train": rel(root, files["whitelist_train"]),
            "whitelist_test": rel(root, files["whitelist_test"]),
            "template_train": rel(root, files["template_train"]),
            "template_test": rel(root, files["template_test"]),
            "csrna_r1_filtered": rel(root, r1),
            "csrna_r2_filtered": rel(root, r2),
            "csrna_dense_pooled": rel(root, files["dense_train"]),
        },
        "contract": {
            "official_train_test_contigs_disjoint": True,
            "template_train_exactly_whitelist_plus_then_minus": True,
            "template_test_exactly_whitelist_plus_then_minus": True,
            "sparse_replicates_inside_train_whitelist": True,
            "train_position_strands": train_contract["position_strand_count"],
            "test_position_strands": test_contract["position_strand_count"],
        },
    }
    return row, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = project_root(args.project_root)
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / "artifacts" / "audits" / "EPIC_5SPECIES_INPUT_AUDIT"
    )
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    mappings = {}
    failures = []
    for assembly in ASSEMBLIES:
        print(f"\n=== {assembly} ===", flush=True)
        try:
            row, mapping = audit_assembly(root, assembly)
            rows.append(row)
            mappings[assembly] = mapping
            print("PASS", flush=True)
            print(
                f"train={row['train_position_strands']:,} "
                f"test={row['test_position_strands']:,} "
                f"r1={row['r1_file']} r2={row['r2_file']}",
                flush=True,
            )
        except Exception as error:
            failures.append({"assembly": assembly, "error": repr(error)})
            rows.append({"assembly": assembly, "status": "FAIL", "error": repr(error)})
            print(f"FAIL: {error!r}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "assembly_audit.csv", index=False)
    (output / "assembly_file_map.json").write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "audit_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    print(frame.to_string(index=False))
    print(f"\nArtifacts: {output}")
    if failures:
        raise SystemExit(f"Audit failed for {len(failures)} assemblies; see audit_failures.json")


if __name__ == "__main__":
    main()
