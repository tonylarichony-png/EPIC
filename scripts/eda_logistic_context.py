"""Interpretable, train-only case-control pilot for jaNemVect1.1.

Run through notebooks/02_eda_logistic_context.ipynb, or execute this module.
The population unit is a whitelist coordinate AND strand. All model decisions
are exploratory: 12 contigs, three fixed folds and a deliberately small sample.
No final fit, test-label access, probability calibration or champion promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import sklearn
from threadpoolctl import threadpool_limits

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_nematostella as eda
from ml_project.sequence_features import (
    encode_dinucleotide,
    encode_positional_bases,
)

OUT = "artifacts/eda/logistic_context"
BASES = "ACGTN"
NONREFERENCE = (0, 1, 2, 4)  # T is absent, so there is no full one-hot alias.
SCALES = (11, 21, 51, 101, 201, 401, 801)


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _save_tables(tables, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(folder / f"{name}.csv", index=False)


def length_folds(lengths, n_folds=3):
    """Descending length, deterministic lexical ties, round-robin; no labels."""
    return {name: i % n_folds for i, name in enumerate(
        sorted(lengths, key=lambda name: (-lengths[name], name)))}


def sample_positions(w, positive_ix, counts, lengths, seed=42,
                     positives_per_contig=1500, negatives_per_contig=6000):
    """Simple random sample without replacement in each contig/class stratum."""
    rng = np.random.default_rng(seed)
    positive_names, _, _ = w.decode(positive_ix)
    folds = length_folds(lengths)
    samples, populations = [], []
    for name in sorted(w.groups):
        group = w.groups[name]
        widths = w.ends[group] - w.starts[group]
        starts = np.r_[0, np.cumsum(widths)[:-1]]
        bp = int(widths.sum())
        positive_mask = positive_names == name
        available = positive_ix[positive_mask]
        n_pos, n_neg = len(available), 2 * bp - len(available)
        take_pos = rng.choice(available, min(positives_per_contig, n_pos), replace=False)
        need = min(negatives_per_contig, n_neg)
        selected = np.empty(0, dtype=np.int64)
        while len(selected) < need:
            # Rejection sampling is uniform; unlike a dense population mask this
            # takes memory proportional to the sampled positions.
            draw = rng.choice(2 * bp, min(2 * bp, max(64, 2 * (need-len(selected)))), replace=False)
            local = draw % bp
            row = np.searchsorted(starts, local, side="right") - 1
            global_ix = w.offsets[group[row]] + local - starts[row] + (draw >= bp) * w.total
            global_ix = global_ix[~np.isin(global_ix, available)]
            global_ix = global_ix[~np.isin(global_ix, selected)]
            selected = np.r_[selected, global_ix[:need-len(selected)]]
        for values, label, population in ((take_pos, 1, n_pos), (selected, 0, n_neg)):
            if not len(values):
                raise ValueError(f"Empty sample class {label} in {name}")
            names, coordinates, minus = w.decode(values)
            sample_counts = np.zeros(len(values), dtype=np.int64)
            if label:
                sample_counts = counts[np.searchsorted(positive_ix, values)]
            samples.append(pd.DataFrame({"template_index": values, "contig": names,
                "coordinate_0based": coordinates, "strand": np.where(minus, "-", "+"),
                "fold": folds[name], "target": label, "count": sample_counts,
                "population_weight": population / len(values)}))
        populations.append({"contig": name, "contig_length_bp": lengths[name],
            "whitelist_bp": bp, "population_positions": 2 * bp,
            "population_positive": n_pos, "population_negative": n_neg,
            "population_prevalence": n_pos / (2 * bp), "fold": folds[name],
            "sample_positive": len(take_pos), "sample_negative": len(selected)})
    rows = pd.concat(samples, ignore_index=True).sort_values("template_index").reset_index(drop=True)
    rows.insert(0, "row_id", np.arange(len(rows)))
    if rows.template_index.duplicated().any():
        raise AssertionError("Duplicate coordinate/strand samples")
    if rows.groupby("contig").fold.nunique().max() != 1:
        raise AssertionError("A contig was split across folds")
    return rows, pd.DataFrame(populations)


def prepare(root, seed=42):
    root = Path(root).resolve()
    folder = root / OUT
    folder.mkdir(parents=True, exist_ok=True)
    previous_path = root / "artifacts/eda/nematostella/summary.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    archive = root / "data/raw" / f"{eda.ASM}.tgz"
    eda.log("Logistic EDA: verify original archive and every consumed source hash")
    archive_md5 = eda.digest(archive, "md5")
    if archive_md5 != eda.MD5 or archive_md5 != previous["archive_md5"]:
        raise ValueError("Archive differs from the previous audited source")
    source = root / "data/raw/unpacked" / eda.ASM
    names = ["genome/genome.fa.gz", "genome/genome.fa.fai",
             "for_participants/whitelist.train.bed.gz", "for_participants/whitelist.test.bed.gz",
             "for_participants/template.train.bed.gz",
             "for_participants/csRNA-r1-train.bed.gz", "for_participants/csRNA-r2-train.bed.gz"]
    recorded = {r["name"]: r for r in previous["source_files"]}
    source_records = []
    for name in names:
        path = source / name
        actual = eda.digest(path)
        expected = recorded[f"{eda.ASM}/{name}"]
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Source changed since previous audit: {name}")
        source_records.append({"name": name, "sha256": actual, "bytes": path.stat().st_size})
    seqs, alphabet = eda.read_fasta(source / "genome/genome.fa.gz")
    lengths = {name: len(seq) for name, seq in seqs.items()}
    fai = {p[0]: int(p[1]) for line in (source / "genome/genome.fa.fai").read_text().splitlines()
           if (p := line.split())}
    if lengths != fai:
        raise ValueError("FASTA and FAI lengths differ")
    w = eda.Whitelist.read(source / "for_participants/whitelist.train.bed.gz", seqs)
    test_w = eda.Whitelist.read(source / "for_participants/whitelist.test.bed.gz", seqs)
    if set(w.groups) & set(test_w.groups):
        raise ValueError("Train/test contig overlap")
    if len(w.groups) != 12:
        raise ValueError("This preregistered pilot expects twelve training contigs")
    w.audit_template(source / "for_participants/template.train.bed.gz")
    tracks = [eda.read_bed(source / f"for_participants/csRNA-r{i}-train.bed.gz", w)
              for i in (1, 2)]
    ix, counts, r1, r2 = eda.merge_tracks(*tracks)
    if len(ix) != previous["target"]["positive"] or int(counts.sum()) != previous["target"]["read_sum"]:
        raise ValueError("Train count totals differ from previous audit")
    train_lengths = {name: lengths[name] for name in w.groups}
    rows, contigs = sample_positions(w, ix, counts, train_lengths, seed)
    decoded = w.decode(rows.template_index.to_numpy())
    # Only train contigs remain in the working state after split validation.
    seqs = {name: seqs[name] for name in w.groups}
    context = eda.oriented_context(seqs, decoded, np.arange(-400, 401))
    provenance = {"assembly": eda.ASM, "seed": seed, "archive_md5": archive_md5,
        "archive_sha256": eda.digest(archive), "source_files": source_records,
        "previous_summary_sha256": eda.digest(previous_path),
        "parser_sha256": eda.digest(eda.__file__), "engine_sha256": eda.digest(__file__),
        "test_labels_read": False, "source_hashes_match_previous_audit": True,
        "previous_dense_train_audit_reused": True, "sample_positions": len(rows),
        "sample_positive": int(rows.target.sum()),
        "population_positions": int(contigs.population_positions.sum()),
        "population_positive": int(contigs.population_positive.sum()),
        "population_prevalence": float(contigs.population_positive.sum()/contigs.population_positions.sum()),
        "sample_prevalence": float(rows.target.mean()),
        "split_rule": "sort training contigs by descending FASTA length, lexical ties; round-robin into 3 folds",
        "sample_rule": "SRS without replacement per contig/class: 1500 positives and 6000 negatives, seed 42",
        "coordinate_unit": "0-based genomic coordinate AND strand inside train whitelist",
        "context_policy": "centered on coordinate, oriented by strand, can extend beyond whitelist but never beyond FASTA contig; outside=N",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__}}
    state = {"rows": rows, "contigs": contigs, "seqs": seqs, "decoded": decoded,
             "context": context, "provenance": provenance, "seed": seed, "composition": {}}
    # Refuse a changed source/configuration before overwriting any artifacts from
    # an earlier run. run_experiments repeats the same idempotent check.
    preregister(state, root)
    rows.to_csv(folder / "sample_rows.csv.gz", index=False, compression="gzip")
    contigs.to_csv(folder / "contigs.csv", index=False)
    _json(folder / "provenance.json", provenance)
    eda.log(f"Prepared {len(rows):,} rows on 12 whole-contig groups")
    return state


def composition_features(seqs, decoded, width):
    """GC / valid ACGT; N, lowercase and outside-contig fractions / full width."""
    radius = width // 2
    if width % 2 != 1 or width < 1:
        raise ValueError("Window width must be positive and odd")
    names, positions, _ = decoded
    result = np.zeros((len(names), 5), dtype=np.float64)
    for name in np.unique(names):
        take = np.flatnonzero(names == name)
        seq = seqs[name]
        lo, hi = np.maximum(0, positions[take]-radius), np.minimum(len(seq), positions[take]+radius+1)
        values = []
        for flag in (((seq == 1) | (seq == 2) | (seq == 5) | (seq == 6)),
                     seq == 8, (seq >= 4) & (seq < 8)):
            prefix = np.empty(len(seq)+1, dtype=np.int32)
            prefix[0] = 0
            np.cumsum(flag, dtype=np.int32, out=prefix[1:])
            values.append(prefix[hi]-prefix[lo])
        valid = hi-lo-values[1]
        result[take, 0] = np.divide(values[0], valid, out=np.zeros(len(take)), where=valid > 0)
        result[take, 1] = values[1] / width
        result[take, 2] = (width - (hi-lo)) / width
        result[take, 3] = valid == 0
        result[take, 4] = values[2] / width
    columns = [f"gc_acgt_{width}", f"unknown_fraction_{width}", f"edge_fraction_{width}",
               f"gc_missing_{width}", f"lower_fraction_{width}"]
    return pd.DataFrame(result, columns=columns)


def _composition(state, width):
    if width not in state["composition"]:
        state["composition"][width] = composition_features(state["seqs"], state["decoded"], width)
    return state["composition"][width]


def weighted_quantile(values, weights, quantiles=(.1, .5, .9)):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return np.interp(quantiles, (np.cumsum(weights)-.5*weights)/weights.sum(), values)


def duplicate_context_audit(context, rows):
    """Exact equality only, using bytes without lossy hashes or case indicators."""
    seen = {}
    for i, sequence in enumerate(context):
        seen.setdefault(sequence.tobytes(), []).append(i)
    repeated = [indices for indices in seen.values() if len(indices) > 1]
    cross = [indices for indices in repeated if rows.iloc[indices].fold.nunique() > 1]
    mask = np.zeros(len(rows), dtype=bool)
    for indices in cross:
        mask[indices] = True
    return {"sample_windows": len(rows), "window_bp": context.shape[1],
        "unique_windows": len(seen), "duplicate_groups": len(repeated),
        "excess_duplicate_windows": sum(len(indices)-1 for indices in repeated),
        "cross_fold_duplicate_groups": len(cross), "cross_fold_duplicate_rows": int(mask.sum()),
        "cross_fold_duplicate_fraction": float(mask.mean()),
        "largest_multiplicity": max(map(len, seen.values()), default=0),
        "scope": "sampled exact oriented ACGTN windows; distant homologous/near-identical repeats unmeasured"}, mask


def feature_audit():
    return pd.DataFrame([
        ("positional A/C/G/N", "include", "one indicator per non-T base per strand-oriented offset", "T reference; N also codes padding, composition separates edge"),
        ("dinucleotide [-1,0]", "baseline", "categorical pair, TT reference; unknown/edge separate level", "interpretable pair association; count labels never enter X"),
        ("GC composition", "ablation", "GC / valid ACGT in centered window; zero if none, gc_missing=1", "fit scaler on training folds only; correlated with positional sequence"),
        ("N / edge composition", "include with GC", "N within contig / full width; outside contig / full width", "technical context availability indicators"),
        ("adjacent dinucleotide interactions +/-5", "ablation", "products of non-T indicators for adjacent bases", "conditional interaction odds ratio; excludes redundant full-pair encoding"),
        ("lowercase", "separate ablation", "lowercase ACGT / full width", "assembly masking annotation proxy; not proven causal or portable"),
        ("csRNA counts and replicates", "exclude from X", "pooled count>0 forms y; counts only in diagnostics", "target leakage"),
        ("rnaseq controls", "exclude from X", "train-only measured control not guaranteed at inference", "availability and assay leakage risk"),
        ("contig ID / coordinate / strand", "metadata only", "grouping, orientation, diagnostics", "avoid contig memorization; strand used only to orient sequence"),
        ("dense embeddings / arbitrary k-mers", "exclude", "outside interpretable pilot", "too many correlated or opaque parameters for this EDA"),
    ], columns=["feature_family", "decision", "meaning", "caution"])


def descriptive(state, root):
    rows = state["rows"]
    y, w = rows.target.to_numpy(), rows.population_weight.to_numpy()
    context = state["context"][:, 200:601]
    records = []
    for j, offset in enumerate(range(-200, 201)):
        for base, label in enumerate(BASES):
            positive = float(np.average(context[y == 1, j] == base, weights=w[y == 1]))
            negative = float(np.average(context[y == 0, j] == base, weights=w[y == 0]))
            records.append({"offset": offset, "base": label, "positive_fraction": positive,
                "negative_fraction": negative, "log2_enrichment": np.log2((positive+1e-6)/(negative+1e-6)),
                "pseudocount_probability": 1e-6})
    multiscale = []
    for width in SCALES:
        frame = _composition(state, width)
        for column in frame:
            for target in (0, 1):
                mask = y == target
                values = frame[column].to_numpy()[mask]
                q10, q50, q90 = weighted_quantile(values, w[mask])
                multiscale.append({"window_bp": width, "feature": column.rsplit("_", 1)[0],
                    "target": target, "sample_n": int(mask.sum()),
                    "weighted_mean": np.average(values, weights=w[mask]),
                    "weighted_q10": q10, "weighted_q50": q50, "weighted_q90": q90,
                    "sample_nonzero": int(np.count_nonzero(values))})
    duplicate, duplicate_mask = duplicate_context_audit(state["context"][:, 350:451], rows)
    state["cross_fold_duplicate_mask"] = duplicate_mask
    tables = {"position_enrichment": pd.DataFrame(records), "multiscale_summary": pd.DataFrame(multiscale),
        "feature_audit": feature_audit(), "contigs": state["contigs"],
        "duplicate_audit": pd.DataFrame([duplicate])}
    _save_tables(tables, Path(root)/OUT)
    _json(Path(root)/OUT/"duplicate_audit.json", duplicate)
    return tables


def reference_features(context, offsets):
    """Compatibility adapter around the shared positional encoder."""

    encoded = encode_positional_bases(context, offsets, reference="T")
    return encoded.matrix, list(encoded.names)


def pair_baseline(context):
    """Compatibility adapter around the shared dinucleotide encoder."""

    encoded = encode_dinucleotide(context, offsets=(-1, 0), reference="TT")
    return encoded.matrix, list(encoded.names)


def interaction_features(context):
    parts, names = [], []
    for pos in range(10):
        first, second = context[:, pos], context[:, pos+1]
        mask = (first != 3) & (second != 3)
        mapping = np.array([0, 1, 2, -1, 3])
        column = 4*mapping[first[mask]] + mapping[second[mask]]
        parts.append(sparse.csr_matrix((np.ones(mask.sum()), (np.flatnonzero(mask), column)),
                                       shape=(len(context), 16)))
        names.extend([f"interaction[{pos-5:+d},{pos-4:+d}]={BASES[a]}:{BASES[b]}"
                      for a in NONREFERENCE for b in NONREFERENCE])
    return sparse.hstack(parts, format="csr"), names


def candidates():
    result = [{"model": "dinuc_baseline", "family": "baseline", "position_bp": 2,
               "composition_bp": 0, "interactions": False, "lowercase": False}]
    result += [{"model": f"pos{width}", "family": "position", "position_bp": width,
                "composition_bp": 0, "interactions": False, "lowercase": False}
               for width in (11, 21, 51, 101, 201, 401)]
    result += [{"model": f"pos51_comp{width}", "family": "composition", "position_bp": 51,
                "composition_bp": width, "interactions": False, "lowercase": False}
               for width in (101, 201, 401, 801)]
    result += [{"model": "pos51_interactions11", "family": "interaction", "position_bp": 51,
                "composition_bp": 0, "interactions": True, "lowercase": False},
               {"model": "pos51_comp201_lower", "family": "lowercase", "position_bp": 51,
                "composition_bp": 201, "interactions": False, "lowercase": True}]
    return result


def design_matrix(state, config):
    if config["family"] == "baseline":
        X, names = pair_baseline(state["context"][:, 399:401])
    else:
        radius = config["position_bp"] // 2
        X, names = reference_features(state["context"][:, 400-radius:401+radius], range(-radius, radius+1))
    if config["interactions"]:
        interactions, interaction_names = interaction_features(state["context"][:, 395:406])
        X, names = sparse.hstack([X, interactions], format="csr"), names + interaction_names
    dense = None
    if config["composition_bp"]:
        dense = _composition(state, config["composition_bp"]).copy()
        if not config["lowercase"]:
            dense = dense.iloc[:, :4]
    return X, names, dense


def training_weights(y, population_weights):
    y = np.asarray(y)
    result = np.asarray(population_weights, dtype=float).copy()
    for label in (0, 1):
        mask = y == label
        result[mask] *= .5*len(y)/result[mask].sum()
    return result


def dense_rank_corr(counts, scores):
    positive = np.asarray(counts) > 0
    left = rankdata(np.asarray(counts)[positive], method="dense")
    right = rankdata(np.asarray(scores)[positive], method="dense")
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def metrics(y, score, weights, counts):
    return {"weighted_ap": float(average_precision_score(y, score, sample_weight=weights)),
            "weighted_roc_auc": float(roc_auc_score(y, score, sample_weight=weights)),
            "positive_dense_rank_corr": dense_rank_corr(counts, score),
            "population_prevalence": float(np.average(y, weights=weights))}


def preregister(state, root):
    config = {"seed": state["seed"], "source": state["provenance"], "models": candidates(),
        "folds": state["contigs"][["contig", "fold", "contig_length_bp"]].to_dict("records"),
        "fit": {"model": "LogisticRegression", "C": 1.0, "penalty": "L2", "solver": "lbfgs",
                "max_iter": 350, "tol": 1e-4, "threads": 2,
                "sample_weight": "inverse inclusion probabilities, balanced class totals, mean one",
                "scaler": "StandardScaler on dense composition, training fold only with same training weights"},
        "primary_metric": "inverse-inclusion weighted average precision; finite sample population approximation",
        "decision_rule": "exploratory shortlist only; positional window with fewest bases within one SE of best mean three-fold weighted AP; SE=fold SD/sqrt(3), not an inferential CI; compare all ablations separately",
        "uncertainty": "200 paired contig bootstrap replicates of macro per-contig AP differences; conditional on fitted OOF models and sampled rows, no refitting",
        "probability_calibration": False, "test_labels_read": False, "final_fit": False}
    folder = Path(root)/OUT
    path = folder/"preregistration.json"
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["config_sha256"] != config_hash:
            raise ValueError("Existing preregistration differs; use a fresh output directory before changing pilot")
        return old, eda.digest(path)
    record = {"created_utc_before_first_fit": datetime.now(timezone.utc).isoformat(),
              "config_sha256": config_hash, "config": config}
    _json(path, record)
    digest = eda.digest(path)
    (folder/"preregistration.sha256").write_text(digest+"\n", encoding="ascii")
    return record, digest


def _cached_results(state, folder, preregistration, preregistration_sha256):
    """Reuse only complete, hash-verified results for the identical pilot."""
    path = folder/"manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest["engine_sha256"] != eda.digest(__file__)
            or manifest["preregistration_sha256"] != preregistration_sha256):
        return None
    table_names = ("scores", "summary", "coefficients", "per_contig_scores", "paired_bootstrap",
                   "candidates", "scalers", "duplicate_sensitivity")
    files = [name+".csv" for name in table_names] + ["oof_predictions.csv.gz", "recommendation.json"]
    for filename in files:
        file_path = folder/filename
        if not file_path.exists() or eda.digest(file_path) != manifest["files"].get(filename):
            raise ValueError(f"Cached result failed integrity check: {filename}")
    oof = pd.read_csv(folder/"oof_predictions.csv.gz", float_precision="round_trip")
    rows = state["rows"]
    if len(oof) != len(rows):
        return None
    for name in ("row_id", "template_index", "contig", "coordinate_0based", "strand", "fold", "target", "count", "population_weight"):
        if not np.array_equal(oof[name].to_numpy(), rows[name].to_numpy()):
            return None
    result = {name: pd.read_csv(folder/f"{name}.csv", float_precision="round_trip") for name in table_names}
    result.update({"recommendation": json.loads((folder/"recommendation.json").read_text(encoding="utf-8")),
                   "manifest": manifest, "preregistration": preregistration, "oof": oof})
    eda.log("Reuse completed pilot: source, engine, preregistration, sampled rows and result hashes match")
    return result


def run_experiments(state, root):
    folder = Path(root)/OUT
    folder.mkdir(parents=True, exist_ok=True)
    preregistration, preregistration_sha256 = preregister(state, root)
    cached = _cached_results(state, folder, preregistration, preregistration_sha256)
    if cached is not None:
        return cached
    rows = state["rows"]
    y, weights = rows.target.to_numpy(), rows.population_weight.to_numpy()
    counts, folds = rows["count"].to_numpy(), rows.fold.to_numpy()
    scores, contig_scores, coefficient_rows, scaler_rows = [], [], [], []
    oof = rows.copy()
    model_records = candidates()
    if "cross_fold_duplicate_mask" not in state:
        _, state["cross_fold_duplicate_mask"] = duplicate_context_audit(state["context"][:, 350:451], rows)
    oof["exact101_cross_fold_duplicate"] = state["cross_fold_duplicate_mask"]
    total_started = time.monotonic()
    for config in model_records:
        model_name = config["model"]
        eda.log(f"Logistic pilot: {model_name}")
        X, sparse_names, dense = design_matrix(state, config)
        names = sparse_names + ([] if dense is None else list(dense.columns))
        predictions = np.empty(len(rows))
        all_coef, all_intercept = [], []
        for fold in range(3):
            train, valid = folds != fold, folds == fold
            train_contigs, valid_contigs = set(rows.contig[train]), set(rows.contig[valid])
            if train_contigs & valid_contigs:
                raise AssertionError("Train/validation contig overlap")
            train_w = training_weights(y[train], weights[train])
            X_train, X_valid = X[train], X[valid]
            scaler = None
            if dense is not None:
                scaler = StandardScaler().fit(dense.to_numpy()[train], sample_weight=train_w)
                X_train = sparse.hstack([X_train, sparse.csr_matrix(scaler.transform(dense.to_numpy()[train]))], format="csr")
                X_valid = sparse.hstack([X_valid, sparse.csr_matrix(scaler.transform(dense.to_numpy()[valid]))], format="csr")
                for name, mean, scale in zip(dense.columns, scaler.mean_, scaler.scale_):
                    scaler_rows.append({"model": model_name, "fold": fold, "feature": name,
                                        "train_mean": mean, "train_scale": scale})
            started = time.monotonic()
            estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=350, tol=1e-4,
                                           random_state=state["seed"])
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(X_train, y[train], sample_weight=train_w)
                predictions[valid] = estimator.decision_function(X_valid)
            elapsed = time.monotonic()-started
            convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            convergence = not convergence_warnings and int(estimator.n_iter_[0]) < 350
            record = {"model": model_name, "fold": fold, **metrics(y[valid], predictions[valid], weights[valid], counts[valid]),
                "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
                "train_contigs": ";".join(sorted(train_contigs)), "validation_contigs": ";".join(sorted(valid_contigs)),
                "n_features": X_train.shape[1], "n_iter": int(estimator.n_iter_[0]),
                "converged": convergence, "fit_seconds": elapsed,
                "convergence_warning": " | ".join(convergence_warnings)}
            scores.append(record)
            coef = estimator.coef_[0].copy()
            intercept = float(estimator.intercept_[0])
            if scaler is not None:
                coef[-len(dense.columns):] /= scaler.scale_
                intercept -= float(np.dot(coef[-len(dense.columns):], scaler.mean_))
            all_coef.append(coef)
            all_intercept.append(intercept)
            eda.log(f"  fold {fold}: weighted AP={record['weighted_ap']:.6f}; iter={record['n_iter']}; {elapsed:.1f}s")
            del X_train, X_valid, estimator
        coefs = np.stack(all_coef)
        for j, name in enumerate(names):
            numeric_fraction = name.startswith(("gc_acgt_", "unknown_fraction_", "edge_fraction_", "lower_fraction_"))
            unit = .1 if numeric_fraction else 1.
            effects = coefs[:, j]*unit
            coefficient_rows.append({"model": model_name, "feature": name,
                "coef_mean": float(effects.mean()), "coef_sd": float(effects.std(ddof=1)),
                "coef_min": float(effects.min()), "coef_max": float(effects.max()),
                "coef_fold0": effects[0], "coef_fold1": effects[1], "coef_fold2": effects[2],
                "sign_agreement": max(float(np.mean(effects > 0)), float(np.mean(effects < 0))),
                "odds_ratio_mean": float(np.exp(np.clip(effects.mean(), -700, 700))),
                "effect_unit": "fraction +0.10" if numeric_fraction else "indicator 0 to 1",
                "interpretation": "odds(A,B)*odds(T,T)/(odds(A,T)*odds(T,B)); conditional cross-product interaction" if name.startswith("interaction") else
                                  "conditional odds multiplier at T neighbors; mutation also changes active interactions" if name.startswith("base") and config["interactions"] else
                                  "conditional odds multiplier in class-balanced pilot; association, not causality",
                "reference": "TT" if name.startswith("pair") else "T at this offset" if name.startswith("base") else "other features held fixed"})
        for fold, intercept in enumerate(all_intercept):
            coefficient_rows.append({"model": model_name, "feature": f"intercept_fold{fold}", "coef_mean": intercept,
                "effect_unit": "class-balanced intercept; not natural log odds", "interpretation": "not calibrated"})
        oof[model_name] = predictions
        for contig in sorted(rows.contig.unique()):
            mask = rows.contig.to_numpy() == contig
            contig_scores.append({"model": model_name, "contig": contig, "fold": int(folds[mask][0]),
                                  **metrics(y[mask], predictions[mask], weights[mask], counts[mask])})
        # Incremental, inspectable outputs; a failed candidate leaves prior work.
        pd.DataFrame(scores).to_csv(folder/"scores.csv", index=False)
        oof.to_csv(folder/"oof_predictions.csv.gz", index=False, compression="gzip")
        del X
        gc.collect()
    score_frame, contig_frame = pd.DataFrame(scores), pd.DataFrame(contig_scores)
    summary_records = []
    for config in model_records:
        name = config["model"]
        local = score_frame[score_frame.model == name]
        per = contig_frame[contig_frame.model == name]
        summary_records.append({**config, **metrics(y, oof[name].to_numpy(), weights, counts),
            "fold_ap_mean": local.weighted_ap.mean(), "fold_ap_sd": local.weighted_ap.std(ddof=1),
            "fold_ap_se": local.weighted_ap.std(ddof=1)/np.sqrt(3),
            "mean_contig_ap": per.weighted_ap.mean(), "min_contig_ap": per.weighted_ap.min(),
            "n_features": int(local.n_features.iloc[0]), "all_folds_converged": bool(local.converged.all()),
            "total_fit_seconds": local.fit_seconds.sum()})
    summary_frame = pd.DataFrame(summary_records)
    pivot = contig_frame.pivot(index="contig", columns="model", values="weighted_ap")
    rng = np.random.default_rng(state["seed"]+100)
    boot = rng.integers(0, len(pivot), size=(200, len(pivot)))
    comparison_records = []
    for name in pivot.columns:
        if name == "dinuc_baseline":
            continue
        for reference in dict.fromkeys(("dinuc_baseline", "pos51",
                                       "pos51_comp201" if name == "pos51_comp201_lower" else "pos51")):
            if name == reference or reference not in pivot.columns:
                continue
            differences = (pivot[name]-pivot[reference]).to_numpy()
            simulated = differences[boot].mean(axis=1)
            low, high = np.quantile(simulated, [.025, .975])
            comparison_records.append({"model": name, "reference": reference,
                "mean_contig_ap_delta": differences.mean(), "ci_low": low, "ci_high": high,
                "contigs_improved": int((differences > 0).sum()), "contigs": len(differences),
                "bootstrap_replicates": 200,
                "estimand": "unweighted mean paired contig AP difference; fitted OOF/sample conditional"})
    positions = summary_frame[(summary_frame.family == "position") & summary_frame.all_folds_converged]
    if len(positions):
        best = positions.sort_values("fold_ap_mean", ascending=False).iloc[0]
        threshold = float(best.fold_ap_mean-best.fold_ap_se)
        eligible = positions[positions.fold_ap_mean >= threshold].sort_values("position_bp")
        selected = eligible.iloc[0]
        selected_window = int(selected.position_bp)
        window_shortlist = eligible.model.tolist()
        best_position = best.model
    else:
        threshold, selected_window, window_shortlist, best_position = None, None, [], None
    recommendation = {"status": "exploratory_shortlist_only", "champion_selected": False,
        "simplest_position_window_bp": selected_window, "best_position_model_by_fold_mean": best_position,
        "position_one_se_threshold": threshold, "position_shortlist": window_shortlist,
        "one_se_caution": "three folds overlap in training data; one-SE is a simplicity heuristic, not confidence guarantee",
        "baseline": "dinuc_baseline", "composition_anchor": "pos51",
        "ranked_ablation_candidates": summary_frame[summary_frame.family.isin(["composition", "interaction", "lowercase"]) & summary_frame.all_folds_converged].sort_values("fold_ap_mean", ascending=False).model.tolist(),
        "next_step": "confirm shortlisted contexts using larger negative samples and an independent development split; review exact duplicates and homologous repeats; then freeze design before final test",
        "limitations": ["18,000 positive / 72,000 negative sampled rows; weighted AP estimates population AP with sampling uncertainty",
            "all target-aware exploratory plots and model comparisons use training contigs; folds are reused for selection",
            "conditional bootstrap omits model-refit and position-sampling variance; 12 contigs is a small cluster sample",
            "exact101 duplicate check is sampled and does not detect distant homologous or near-identical repeats",
            "balanced training scores are not calibrated natural probabilities; no threshold or precision claim on enriched AP",
            "dense-rank correlation is only a diagnostic on sampled positives; binary LR does not model count magnitude",
            "lowercase is an assembly annotation proxy; coefficients are conditional associations under collinearity"]}
    sensitivity_records = []
    mask = ~state["cross_fold_duplicate_mask"]
    for config in model_records:
        name = config["model"]
        sensitivity_records.append({"model": name, "rows_kept": int(mask.sum()),
            "rows_removed": int((~mask).sum()), **metrics(y[mask], oof[name].to_numpy()[mask], weights[mask], counts[mask]),
            "scope": "evaluation rows only, no refit, retained-row estimand; does not establish repeat independence"})
    tables = {"scores": score_frame, "summary": summary_frame,
        "coefficients": pd.DataFrame(coefficient_rows), "per_contig_scores": contig_frame,
        "paired_bootstrap": pd.DataFrame(comparison_records), "candidates": pd.DataFrame(model_records),
        "scalers": pd.DataFrame(scaler_rows), "duplicate_sensitivity": pd.DataFrame(sensitivity_records)}
    _save_tables(tables, folder)
    _json(folder/"recommendation.json", recommendation)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": preregistration_sha256, "engine_sha256": eda.digest(__file__),
        "models_fitted": len(model_records), "fits": len(scores), "test_labels_read": False,
        "elapsed_seconds": time.monotonic()-total_started, "oof_score": "decision_function; uncalibrated logit",
        "files": {p.name: eda.digest(p) for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"}}
    _json(folder/"manifest.json", manifest)
    eda.log(f"Completed {len(scores)} grouped fits in {manifest['elapsed_seconds']:.1f}s")
    return {**tables, "recommendation": recommendation, "manifest": manifest,
            "preregistration": preregistration, "oof": oof}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepared = prepare(args.root, args.seed)
    descriptive(prepared, args.root)
    run_experiments(prepared, args.root)
