"""Reproducible descriptive EPIC audit. No model fitting and no test-label reads.

Run from the repository root: python scripts/eda_nematostella.py
The immutable source archive stays in data/raw; extracted inputs and full JSON
results remain Git-ignored. Only aggregate plots are written to assets/eda.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import tarfile
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.sequence_context import (
    COMPLEMENT_CODE,
    FASTA_LOOKUP,
    UPPERCASE_BASE_CODE,
    extract_oriented_context_from_arrays,
    read_fasta,
)

ASM = "jaNemVect1.1"
MD5 = "733dccb6393a3c07b6e0d7ffaed3686f"
# Compatibility aliases used by the existing EDA and its cached artifacts.
LUT = FASTA_LOOKUP
BASE = UPPERCASE_BASE_CODE
COMP = COMPLEMENT_CODE


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_inputs(archive, root):
    """Extract only genome and participant files; reject links and unsafe paths."""
    if digest(archive, "md5") != MD5:
        raise ValueError("Source archive MD5 differs from the published version")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(archive, "r|gz") as ar:
        for member in ar:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != ASM or ".." in parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError("Archive links are not allowed")
            if len(parts) < 3 or parts[1] not in {"genome", "for_participants"}:
                continue
            if not member.isfile():
                if not member.isdir():
                    raise ValueError("Unsupported archive entry")
                continue
            target = root.joinpath(*parts).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Extraction would leave the data directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Stream and compare existing files as well: a stale/corrupt extracted
            # file must not silently become the source of a reproducible result.
            source = ar.extractfile(member)
            h = hashlib.sha256()
            if target.exists():
                with target.open("rb") as old:
                    for block in iter(lambda: source.read(8 << 20), b""):
                        if old.read(len(block)) != block:
                            raise ValueError(f"Existing extracted file differs: {target}")
                        h.update(block)
                    if old.read(1):
                        raise ValueError("Existing extracted file has trailing data")
            else:
                with target.open("xb") as out:
                    for block in iter(lambda: source.read(8 << 20), b""):
                        out.write(block)
                        h.update(block)
            records.append({"name": member.name, "bytes": member.size, "sha256": h.hexdigest()})
    return root / ASM, records


class Whitelist:
    def __init__(self, rows, seqs):
        self.rows = rows
        self.names = np.array([r[0] for r in rows])
        self.starts = np.array([r[1] for r in rows], dtype=np.int64)
        self.ends = np.array([r[2] for r in rows], dtype=np.int64)
        lengths = self.ends - self.starts
        if len(rows) == 0 or np.any(lengths <= 0):
            raise ValueError("Empty or invalid whitelist interval")
        self.offsets = np.r_[0, np.cumsum(lengths)[:-1]]
        self.total = int(lengths.sum())
        self.groups = {}
        for name in np.unique(self.names):
            ix = np.flatnonzero(self.names == name)
            s, e = self.starts[ix], self.ends[ix]
            if name not in seqs or np.any(s < 0) or np.any(e > len(seqs[name])):
                raise ValueError("Whitelist outside FASTA bounds")
            if np.any(s[1:] < e[:-1]):
                raise ValueError("Unsorted or overlapping whitelist intervals")
            self.groups[str(name)] = ix

    @classmethod
    def read(cls, path, seqs):
        with gzip.open(path, "rt", encoding="ascii") as f:
            return cls([(p[0], int(p[1]), int(p[2])) for line in f if (p := line.split())], seqs)

    def encode_interval(self, name, start, end, strand):
        if strand not in {"+", "-"} or end <= start:
            raise ValueError("Invalid BED strand or interval")
        if name not in self.groups:
            raise ValueError("Signal contig not in whitelist")
        group = self.groups[name]
        local = np.searchsorted(self.starts[group], start, side="right") - 1
        if local < 0:
            raise ValueError("Signal before whitelist")
        row = group[local]
        if start < self.starts[row] or end > self.ends[row]:
            raise ValueError("Signal interval not fully inside whitelist")
        first = self.offsets[row] + start - self.starts[row]
        first += self.total if strand == "-" else 0
        return np.arange(first, first + end - start, dtype=np.int64)

    def decode(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= 2 * self.total):
            raise ValueError("Template index out of bounds")
        minus = indices >= self.total
        local = indices % self.total
        rows = np.searchsorted(self.offsets, local, side="right") - 1
        return self.names[rows], self.starts[rows] + local - self.offsets[rows], minus

    def audit_template(self, path):
        count = 0
        with gzip.open(path, "rt", encoding="ascii") as f:
            for line in f:
                p = line.split()
                if count >= 2 * len(self.rows):
                    raise ValueError("Extra template rows")
                expected = self.rows[count % len(self.rows)]
                strand = "+" if count < len(self.rows) else "-"
                if (p[0], int(p[1]), int(p[2])) != expected or p[5] != strand:
                    raise ValueError("Template does not match plus-then-minus whitelist order")
                count += 1
        if count != 2 * len(self.rows):
            raise ValueError("Truncated template")


def read_bed(path, whitelist):
    indices, counts = [], []
    spans = Counter()
    records = 0
    with gzip.open(path, "rt", encoding="ascii") as f:
        for line in f:
            p = line.split()
            if len(p) != 6:
                raise ValueError("Expected BED6")
            start, end, value = int(p[1]), int(p[2]), int(p[4])
            if value <= 0:
                raise ValueError("Expected positive integer sparse signal")
            ix = whitelist.encode_interval(p[0], start, end, p[5])
            indices.extend(ix.tolist())
            counts.extend([value] * len(ix))
            spans[end - start] += 1
            records += 1
    ix, val = np.array(indices, dtype=np.int64), np.array(counts, dtype=np.int64)
    order = np.argsort(ix)
    ix, val = ix[order], val[order]
    if np.any(np.diff(ix) == 0):
        raise ValueError("Duplicate position-strand key within one replicate")
    return ix, val, {"bed_rows": records, "positive_positions": len(ix), "read_sum": int(val.sum()),
                     "interval_length_histogram": dict(sorted(spans.items())), "duplicate_keys": 0}


def merge_tracks(left, right):
    ix = np.union1d(left[0], right[0])
    a, b = np.zeros(len(ix), dtype=np.int64), np.zeros(len(ix), dtype=np.int64)
    a[np.searchsorted(ix, left[0])] = left[1]
    b[np.searchsorted(ix, right[0])] = right[1]
    return ix, a + b, a, b


def audit_dense_txt(path, expected_ix, expected_values, expected_length, block_bytes=8 << 20):
    """Compare every dense TXT value with independently expanded sparse BED."""
    cursor, nonzero, total = 0, 0, 0
    with gzip.open(path, "rb") as f:
        while block := f.read(block_bytes):
            block += f.readline()  # complete the final partial line
            parsed = np.fromstring(block, sep="\n", dtype=np.float64)
            lines = block.count(b"\n") + int(not block.endswith(b"\n"))
            if (len(parsed) != lines or not np.all(np.isfinite(parsed))
                    or np.any(parsed < 0) or np.any(parsed >= 2**53)
                    or np.any(parsed != np.floor(parsed))):
                raise ValueError("Invalid integer count TXT")
            # EPIC replicate TXT files can spell integer counts as 1.0 / 1e0.
            values = parsed.astype(np.int64)
            end = cursor + len(values)
            lo, hi = np.searchsorted(expected_ix, [cursor, end])
            nz = np.flatnonzero(values)
            if not np.array_equal(cursor + nz, expected_ix[lo:hi]):
                raise ValueError(f"BED/TXT nonzero-coordinate mismatch in {path.name}")
            if not np.array_equal(values[nz], expected_values[lo:hi]):
                raise ValueError(f"BED/TXT count mismatch in {path.name}")
            nonzero += len(nz)
            total += int(values.sum())
            cursor = end
    if cursor != expected_length:
        raise ValueError("TXT length differs from template")
    return {"lines": cursor, "nonzero": nonzero, "read_sum": total, "mismatches": 0}


def sequence_summary(seqs, w):
    all_counts, by_contig = np.zeros(9, dtype=np.int64), {}
    for name, group in w.groups.items():
        counts = np.zeros(9, dtype=np.int64)
        for i in group:
            counts += np.bincount(seqs[name][w.starts[i]:w.ends[i]], minlength=9)
        by_contig[name] = counts
        all_counts += counts
    gc = int(all_counts[[1, 2, 5, 6]].sum())
    return {"bp": w.total, "intervals": len(w.rows), "contigs": len(w.groups),
            "code_counts": all_counts.tolist(), "gc_fraction_all_bp": gc / w.total,
            "gc_fraction_acgt": gc / int(all_counts[:8].sum()),
            "lower_fraction": float(all_counts[4:8].sum() / w.total),
            "n_or_other_fraction": float(all_counts[8] / w.total)}, by_contig


def oriented_context(seqs, decoded, offsets):
    """Compatibility adapter around the shared production implementation."""

    names, positions, minus = decoded
    strands = np.where(np.asarray(minus, dtype=bool), "-", "+")
    return extract_oriented_context_from_arrays(
        seqs,
        names,
        positions,
        strands,
        offsets,
    )


def window_stats(seqs, decoded, radius=100):
    names, positions, _ = decoded
    gc, lower, unknown, edge = (np.empty(len(names), dtype=float) for _ in range(4))
    for name in np.unique(names):
        take = np.flatnonzero(names == name)
        seq = seqs[name]
        lo, hi = np.maximum(0, positions[take] - radius), np.minimum(len(seq), positions[take] + radius + 1)
        flags = [(seq == 1) | (seq == 2) | (seq == 5) | (seq == 6), (seq >= 4) & (seq < 8), seq == 8]
        values = []
        for flag in flags:
            prefix = np.r_[0, np.cumsum(flag, dtype=np.int32)]
            values.append(prefix[hi] - prefix[lo])
        valid = hi - lo - values[2]
        gc[take] = np.divide(values[0], valid, out=np.full(len(take), np.nan), where=valid > 0)
        lower[take] = values[1] / (hi - lo)
        unknown[take] = values[2] / (hi - lo)
        edge[take] = hi - lo < 2 * radius + 1
    return {"gc": gc, "lower": lower, "unknown": unknown, "edge": edge}


def full_dinucleotides(seqs, w):
    total = np.zeros(17, dtype=np.int64)
    for name, group in w.groups.items():
        letters = BASE[seqs[name]]
        upstream = np.r_[np.uint8(4), letters[:-1]]
        pair_plus = np.where((upstream < 4) & (letters < 4), upstream * 4 + letters, 16)
        letters = COMP[letters]
        upstream = np.r_[letters[1:], np.uint8(4)]
        pair_minus = np.where((upstream < 4) & (letters < 4), upstream * 4 + letters, 16)
        for i in group:
            s, e = w.starts[i], w.ends[i]
            total += np.bincount(pair_plus[s:e], minlength=17)
            total += np.bincount(pair_minus[s:e], minlength=17)
    if total.sum() != 2 * w.total:
        raise ValueError("Dinucleotide denominator mismatch")
    return total


def corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_counts(values, n):
    quantiles = np.quantile(values, [0, .25, .5, .75, .9, .95, .99, .999, 1])
    histogram = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 100), (101, int(values.max()))]
    return {"positions": n, "positive": len(values), "zero": n - len(values),
            "positive_fraction": len(values) / n, "read_sum": int(values.sum()),
            "positive_mean": float(values.mean()), "positive_quantiles": dict(zip(["min", "q25", "median", "q75", "q90", "q95", "q99", "q999", "max"], quantiles.tolist())),
            "singletons": int(np.count_nonzero(values == 1)),
            "singletons_fraction": float(np.mean(values == 1)),
            "top_1pct_read_share": float(np.sort(values)[-max(1, int(np.ceil(.01 * len(values)))):].sum() / values.sum()),
            "bins": [{"label": str(lo) if lo == hi else f"{lo}-{hi}", "positions": int(np.sum((values >= lo) & (values <= hi)))} for lo, hi in histogram]}


def plot_results(result, context_pos, context_bg, windows_pos, windows_bg, a, b, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    bins = result["target"]["bins"]
    axes[0].bar([r["label"] for r in bins], [100*r["positions"]/result["target"]["positive"] for r in bins], color="#247b92")
    axes[0].set(xlabel="Pooled csRNA count (positive positions)", ylabel="Share of positive positions, %", title="Nematostella train: target distribution")
    axes[0].tick_params(axis="x", rotation=25)
    both = (a > 0) & (b > 0)
    axes[1].hexbin(np.log10(1+a), np.log10(1+b), gridsize=55, bins="log", mincnt=1, cmap="viridis")
    axes[1].set(xlabel="log10(1 + replicate 1)", ylabel="log10(1 + replicate 2)", title=f"Both replicates: {100*both.mean():.1f}% of positive union")
    fig.savefig(folder / "target_replicates.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    by = result["contigs"]
    x = np.arange(len(by))
    axes[0].bar(x, [100*r["positive_fraction"] for r in by], color="#247b92")
    axes[0].set_xticks(x, [r["contig"].replace("NC_", "") for r in by], rotation=65, ha="right")
    axes[0].set(ylabel="Positive positions, %", xlabel="Training contig", title="Rate differs across contigs")
    axes[1].hist(windows_bg["gc"][np.isfinite(windows_bg["gc"])], bins=np.linspace(0,1,31), density=True, histtype="step", linewidth=2, label="Uniform train sample", color="#247b92")
    axes[1].hist(windows_pos["gc"][np.isfinite(windows_pos["gc"])], bins=np.linspace(0,1,31), density=True, histtype="step", linewidth=2, label="All train positives", color="#d77835")
    axes[1].set(xlabel="GC fraction among A/C/G/T in 201 bp window", ylabel="Density", title="Sequence context (descriptive association)")
    axes[1].legend(frameon=False)
    fig.savefig(folder / "contigs_gc.png")
    plt.close(fig)

    freq_pos = np.stack([(context_pos == i).mean(axis=0) for i in range(4)])
    freq_bg = np.stack([(context_bg == i).mean(axis=0) for i in range(4)])
    enrichment = np.log2((freq_pos + 1e-5) / (freq_bg + 1e-5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
    vmax = max(.5, float(np.abs(enrichment).max()))
    im = axes[0].imshow(enrichment, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[-50.5,50.5,3.5,-.5])
    axes[0].set_yticks(range(4), list("ACGT"))
    axes[0].axvline(0, color="black", linewidth=.8)
    axes[0].set(xlabel="Offset from signal position, oriented by strand", title="log2 base-frequency enrichment at positives")
    fig.colorbar(im, ax=axes[0], label="log2 enrichment")
    pairs = sorted(result["dinucleotides"][:16], key=lambda r:r["positive_fraction"], reverse=True)
    axes[1].bar(range(16), [100*r["positive_fraction"] for r in pairs], color="#247b92")
    axes[1].set_xticks(range(16), [r["pair"] for r in pairs], rotation=60)
    axes[1].set(ylabel="Positive positions, %", title="Exact train rates by dinucleotide [-1, 0]")
    fig.savefig(folder / "sequence_context.png")
    plt.close(fig)


def run(root, background_size=200_000, seed=42):
    started = time.monotonic()
    root = root.resolve()
    archive = root / "data/raw" / f"{ASM}.tgz"
    output = root / "artifacts/eda/nematostella"
    output.mkdir(parents=True, exist_ok=True)
    log("Verify archive and extract participant/genome files only")
    source, files = extract_inputs(archive, root / "data/raw/unpacked")
    expected = json.loads((root / "data/raw/genome_metrics.json").read_text())["assemblies"][ASM]
    fp = source / "for_participants"
    log("Read all FASTA records and check FAI lengths")
    seqs, alphabet = read_fasta(source / "genome/genome.fa.gz")
    fai = {p[0]:int(p[1]) for line in (source / "genome/genome.fa.fai").read_text().splitlines() if (p:=line.split())}
    lengths = {k:len(v) for k,v in seqs.items()}
    if lengths != fai or sum(lengths.values()) != expected["genome"]["size_bp"] or len(seqs) != expected["genome"]["contig_count"]:
        raise ValueError("FASTA/FAI/metadata length mismatch")
    whitelists = {sub:Whitelist.read(fp / f"whitelist.{sub}.bed.gz", seqs) for sub in ("train", "test")}
    if set(whitelists["train"].groups) & set(whitelists["test"].groups):
        raise ValueError("Train and test share contigs")
    sequence, contig_codes = {}, {}
    for sub, w in whitelists.items():
        w.audit_template(fp / f"template.{sub}.bed.gz")
        sequence[sub], contig_codes[sub] = sequence_summary(seqs, w)
        exp = expected["whitelist"][sub]
        if w.total != exp["size_bp"] or len(w.groups) != exp["contig_count"] or 2*w.total != expected["template"][sub]["length"]:
            raise ValueError("Whitelist/template sizes differ from metadata")
        sequence[sub]["published_gc_fraction"] = exp["gc_fraction"]
        sequence[sub]["gc_all_bp_matches_published"] = bool(np.isclose(sequence[sub]["gc_fraction_all_bp"], exp["gc_fraction"], atol=2e-6, rtol=0))
        sequence[sub]["gc_acgt_matches_published"] = bool(np.isclose(sequence[sub]["gc_fraction_acgt"], exp["gc_fraction"], atol=2e-6, rtol=0))
        if not (sequence[sub]["gc_all_bp_matches_published"] or sequence[sub]["gc_acgt_matches_published"]):
            raise ValueError("Neither documented GC denominator matches metadata")
    w = whitelists["train"]
    beds, txts, pooled = {}, {}, {}
    for kind in ("csRNA", "rnaseq"):
        log(f"Expand {kind} replicate BED, then compare every train TXT value")
        tracks = [read_bed(fp / f"{kind}-r{r}-train.bed.gz", w) for r in (1,2)]
        for r, track in enumerate(tracks, 1):
            key = f"{kind}-r{r}"
            beds[key] = track[2]
            txts[key] = audit_dense_txt(fp / f"{key}-train.txt.gz", track[0], track[1], 2*w.total)
            log(f"Validated {key}: {txts[key]['lines']} positions")
        pooled[kind] = merge_tracks(tracks[0], tracks[1])
        ix, values, a, b = pooled[kind]
        txts[kind] = audit_dense_txt(fp / f"{kind}-train.txt.gz", ix, values, 2*w.total)
        log(f"Validated pooled {kind}: exact sum of replicates")
    ix, values, a, b = pooled["csRNA"]
    if len(ix) != expected["whitelist"]["nonzeros"]["train"]:
        raise ValueError("Nonzero train count differs from metadata")
    decoded = w.decode(ix)
    names, positions, minus = decoded
    log("Describe counts, replicates, strands and per-contig sequence segments")
    center_codes = np.empty(len(ix), dtype=np.uint8)
    contigs = []
    for name in w.groups:
        take = names == name
        center_codes[take] = seqs[name][positions[take]]
        codes = contig_codes["train"][name]
        positive_codes = np.bincount(center_codes[take], minlength=9)
        contigs.append({"contig":name, "positions":int(2*codes.sum()), "positive":int(take.sum()),
                        "positive_fraction":float(take.sum()/(2*codes.sum())), "read_sum":int(values[take].sum()),
                        "gc_fraction":float(codes[[1,2,5,6]].sum()/codes.sum()),
                        "lower_positions":int(2*codes[4:8].sum()), "lower_positive":int(positive_codes[4:8].sum()),
                        "upper_positions":int(2*codes[:4].sum()), "upper_positive":int(positive_codes[:4].sum())})
    all_codes = np.array(sequence["train"]["code_counts"])*2
    positive_codes = np.bincount(center_codes, minlength=9)
    segments = []
    for label, group in [("uppercase ACGT", range(4)), ("lowercase acgt", range(4,8)), ("N or other", [8])]:
        n, p = int(all_codes[list(group)].sum()), int(positive_codes[list(group)].sum())
        segments.append({"segment":label, "positions":n, "positive":p, "positive_fraction":p/n if n else None})
    for flag, label in [(False,"plus strand"),(True,"minus strand")]:
        take = minus == flag
        segments.append({"segment":label, "positions":w.total, "positive":int(take.sum()),
                         "positive_fraction":float(take.sum()/w.total), "read_sum":int(values[take].sum())})
    both = (a>0)&(b>0)
    result = {"assembly":ASM, "created_utc":datetime.now(timezone.utc).isoformat(),
              "archive_md5":MD5, "script_sha256":digest(__file__), "seed":seed,
              "background_size":background_size, "test_labels_read":False, "model_fitted":False,
              "environment":{"python":platform.python_version(), "numpy":np.__version__, "matplotlib":importlib.metadata.version("matplotlib")},
              "source_files":files, "genome":{"bp":sum(lengths.values()), "contigs":len(seqs), "alphabet":alphabet},
              "sequence":sequence, "bed_audit":beds, "txt_audit":txts,
              "target":summarize_counts(values, 2*w.total), "segments":segments, "contigs":contigs,
              "replicates":{"intersection":int(both.sum()), "union":len(ix), "jaccard":float(both.mean()),
                            "r1_only":int(np.sum((a>0)&(b==0))), "r2_only":int(np.sum((b>0)&(a==0))),
                            "log1p_pearson_positive_union":corr(np.log1p(a),np.log1p(b)),
                            "log1p_pearson_shared_positives":corr(np.log1p(a[both]),np.log1p(b[both]))}}
    # Control association is descriptive, and only uses supplied train tracks.
    rx, rv, _, _ = pooled["rnaseq"]
    union = np.union1d(ix, rx)
    cs, rna = np.zeros(len(union)), np.zeros(len(union))
    cs[np.searchsorted(union,ix)], rna[np.searchsorted(union,rx)] = values, rv
    result["control"] = {"positive_positions":len(rx), "shared_with_csrna":int(np.intersect1d(ix,rx).size),
                         "log1p_pearson_positive_union":corr(np.log1p(cs),np.log1p(rna))}
    log("Describe oriented sequence contexts; uniform train sample, no test labels")
    rng = np.random.default_rng(seed)
    bg_ix = rng.choice(2*w.total, size=background_size, replace=False)
    bg_decoded = w.decode(bg_ix)
    ctx_p = oriented_context(seqs, decoded, np.arange(-50,51))
    ctx_b = oriented_context(seqs, bg_decoded, np.arange(-50,51))
    win_p, win_b = window_stats(seqs,decoded), window_stats(seqs,bg_decoded)
    result["windows"] = {"length":201, "gc_denominator":"ACGT within the clipped contig window", "groups":{}}
    for label, windows in [("all_train_positives",win_p),("uniform_train_background",win_b)]:
        result["windows"]["groups"][label] = {"n":len(windows["gc"]), "gc_mean":float(np.nanmean(windows["gc"])),
            "gc_median":float(np.nanmedian(windows["gc"])), "gc_undefined":int(np.isnan(windows["gc"]).sum()),
            "edge_windows":int(windows["edge"].sum()), "lower_mean":float(windows["lower"].mean()),
            "gc_histogram":np.histogram(windows["gc"][np.isfinite(windows["gc"])], bins=np.linspace(0,1,21))[0].tolist()}
    result["background_positive_positions"] = int(np.isin(bg_ix,ix).sum())
    # This is a sample of exact sequence duplicates, not an exhaustive genome audit.
    sample = ctx_b[:min(100_000,len(ctx_b))]
    _, repeat_counts = np.unique(sample, axis=0, return_counts=True)
    result["sequence_duplicates"] = {"sample_windows":len(sample), "length":101, "unique_windows":len(repeat_counts),
        "excess_duplicate_windows":int(len(sample)-len(repeat_counts)), "largest_multiplicity":int(repeat_counts.max()),
        "orientation":"strand-oriented uppercase ACGTN; outside-contig context encoded N"}
    denom = full_dinucleotides(seqs,w)
    before, at = ctx_p[:,49],ctx_p[:,50]
    pairs = np.where((before<4)&(at<4), before*4+at, 16)
    numer = np.bincount(pairs,minlength=17)
    labels = [a+b for a in "ACGT" for b in "ACGT"]+["N/edge"]
    result["dinucleotides"] = [{"pair":label,"positions":int(n),"positive":int(p),
        "positive_fraction":float(p/n) if n else None,"enrichment":float((p/n)/(len(ix)/(2*w.total))) if n else None}
        for label,n,p in zip(labels,denom,numer)]
    result["elapsed_seconds"] = time.monotonic()-started
    result_path = output / "summary.json"
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plot_results(result,ctx_p,ctx_b,win_p,win_b,a,b,root/"assets/eda/nematostella")
    log(f"Done. Summary: {result_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--background-size",type=int,default=200_000)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.background_size < 1:
        parser.error("--background-size must be positive")
    run(args.root,args.background_size,args.seed)
