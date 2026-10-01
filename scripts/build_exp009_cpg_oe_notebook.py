"""Build the deliberately unexecuted EXP-009 CpG O/E201 notebook."""

from __future__ import annotations

from pathlib import Path
import copy
import textwrap

import nbformat

from build_exp008_gc_interaction_notebook import build_notebook as build_exp008_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "notebooks/experiments/EXP-009_cpg_oe201_logistic.ipynb"


def source(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


def build_notebook():
    notebook = copy.deepcopy(build_exp008_notebook())
    cells = {cell.id: cell for cell in notebook.cells}

    cells["title"].source = source(
        """
        # EXP-009 — EXP-007 + CpG O/E201

        **Статус:** подготовлен, но ни одна code cell не выполнена.

        EXP-008 отклонён, поэтому единственный основной reference — принятый
        EXP-007. К его `k2 + k4 + k6 + GC-bin` добавляется один новый
        категориальный признак расположения оснований: fixed CpG observed/expected bins в
        ориентированном окне 201 bp.

        Не используйте **Run All**. Выполняйте ячейки сверху вниз.
        """
    )
    cells["roadmap"].source = source(
        """
        ## Карта эксперимента

        1. Точно посчитать `6-mer × C × G × CpG × valid` на полном train.
        2. Посмотреть train-only enrichment по диагностическим CpG O/E bins.
        3. Собрать 4 379 признаков EXP-007 + семь CpG O/E201 contrasts.
        4. Выполнить 5 значений `C × 3 folds = 15 fits`.
        5. Сравнить выбранный `C` с EXP-007 на тех же folds.
        6. Один раз довести зафиксированный результат до validation и сохранить
           эксперимент даже при отрицательном исходе.
        """
    )
    cells["setup"].source = source(
        """
        from pathlib import Path
        import json
        import os
        import platform
        import sys
        import time

        _NUMERIC_STACK_ALREADY_LOADED = any(
            name == "numpy" or name.startswith("numpy.") for name in sys.modules
        )
        if (
            _NUMERIC_STACK_ALREADY_LOADED
            and os.environ.get("MKL_THREADING_LAYER", "").upper() != "SEQUENTIAL"
        ):
            raise RuntimeError(
                "Restart the kernel before EXP-009: the numeric stack was "
                "already loaded without MKL_THREADING_LAYER=SEQUENTIAL."
            )
        os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import Markdown, display

        CURRENT_DIR = Path.cwd().resolve()
        PROJECT_ROOT = next(
            candidate for candidate in (CURRENT_DIR, *CURRENT_DIR.parents)
            if (candidate / "README.md").is_file()
            and (candidate / "src/ml_project").is_dir()
        )
        SRC_DIR = PROJECT_ROOT / "src"
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        from ml_project.epic_baseline import load_baseline_sequences
        from ml_project.epic_cpg_logistic import (
            CPG_DIAGNOSTIC_LABELS,
            CpGLogisticConfig,
            aggregate_cpg_likelihood,
            cpg_enrichment_table,
            cross_validate_cpg_logistic,
            evaluate_cpg_logistic,
            fit_cpg_logistic,
            kmer_counts_from_cpg_counts,
            selected_cv_cpg_coefficients,
            split_kmer_cpg_counts,
        )
        from ml_project.epic_data import ensure_prepared_split
        from ml_project.epic_experiment import ExperimentSpec, build_run_record, save_run_record
        from ml_project.epic_kmer import KmerLookupConfig, evaluate_kmer_lookup, fit_kmer_lookup
        from ml_project.epic_kmer_logistic import (
            hierarchical_kmer_design,
            make_balanced_contig_folds,
            select_regularization,
            summarise_cv,
        )

        plt.rcParams.update({
            "figure.dpi": 120,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        })
        pd.set_option("display.max_columns", 80)
        pd.set_option("display.max_rows", 100)

        display(pd.Series({
            "python": platform.python_version(),
            "executable": sys.executable,
            "project_root": str(PROJECT_ROOT),
            "notebook_state": "prepared / unexecuted",
            "MKL_THREADING_LAYER": os.environ["MKL_THREADING_LAYER"],
            "threadpoolctl_used_for_fit": False,
        }, name="value").to_frame())
        """
    )
    cells["config-md"].source = source(
        """
        ## 1. Зафиксированный конфиг и формула

        ```text
        score = k2 + k4 + k6 + GC-bin + CpG-O/E-bin

        CpG O/E201 = CpG_count × valid_ACGT / (C_count × G_count)
        ```

        Окно inclusive `[-100,+100]` в ориентации цепи. CpG является
        reverse-complement invariant. Raw ratio используется без clipping, затем
        попадает в fixed bins `<0.25 … >=1.50`; `C×G=0` — отдельная категория.
        Reference — `[0.75,1.00)`, поэтому добавляется семь contrasts. Границы
        не обучаются и одинаковы на train, validation и test.

        Grid остаётся компактным: `0.0001, 0.0002, 0.0003, 0.0005, 0.001`.
        """
    )
    cells["config"].source = source(
        """
        RUN_NAME = "run_001"
        EXPERIMENT_ID = "EXP-009"
        NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/experiments/EXP-009_cpg_oe201_logistic.ipynb"
        ARTIFACT_DIR = PROJECT_ROOT / "artifacts/experiments/EXP-009" / RUN_NAME
        PLOT_DIR = ARTIFACT_DIR / "plots"

        CONFIG = CpGLogisticConfig()
        LOOKUP_CONFIG = KmerLookupConfig(ks=CONFIG.ks, smoothing="one_expected_positive")
        SPEC = ExperimentSpec(
            experiment_id=EXPERIMENT_ID,
            title="EXP-007 + fixed CpG O/E201 bins",
            hypothesis=(
                "CpG adjacency in the 201 bp sequence environment adds "
                "transferable information beyond local k-mers and GC201 bins"
            ),
            changed_variable="EXP-007 + seven reference-coded fixed CpG O/E201 bins",
            success_criterion=(
                "Before validation: mean AP > EXP-007, AP wins on >=2/3 identical "
                "CV folds, mean Spearman >= EXP-007; validation: relative AP gain "
                ">=1%, Spearman not lower, AP wins on >=2/3 contigs"
            ),
            seed=CONFIG.seed,
        )
        assert len(CONFIG.C_grid) * CONFIG.cv_folds == 15
        assert len(CONFIG.C_grid) == 5
        assert not (ARTIFACT_DIR / "metadata.json").exists(), (
            f"Run already exists: {ARTIFACT_DIR}. Use another run name; do not overwrite."
        )
        display(pd.Series(CONFIG.to_dict(), name="value").to_frame())
        """
    )
    cells["counting-md"].source = source(
        """
        ## 3. Точный подсчёт sufficient statistics

        Для каждой position-strand сохраняются не 201 букв, а совместные
        категории `6-mer`, `GC201-bin` и `CpG-O/E201-bin`. Raw `C`, `G`, `CpG`
        и `valid` считаются внутри bounded chunks и сразу переводятся в fixed bins.
        Одинаковые строки сворачиваются, multiplicity передаётся через
        `sample_weight`. Sampling отсутствует.
        """
    )
    cells["count-train"].source = source(
        """
        started = time.monotonic()
        train_cpg_counts = split_kmer_cpg_counts(
            split,
            sequences,
            "train",
            config=CONFIG,
            verbose=True,
        )
        train_count_seconds = time.monotonic() - started
        train_count_diagnostics = pd.Series({
            "represented_positions": int(train_cpg_counts.totals["total_positions"].sum()),
            "joint_rows": len(train_cpg_counts.totals),
            "positive_rows": len(train_cpg_counts.positives),
            "missing_C_or_G_positive_rows": int(
                train_cpg_counts.positives["cpg_oe201"].isna().sum()
            ),
        }, name="value").to_frame()
        display(train_count_diagnostics)
        print(f"Full train counted in {train_count_seconds:.2f} s")
        """
    )
    cells["check-train-counts"].source = source(
        """
        expected_train = int(split.summary.loc[
            split.summary["split"] == "train", "position_strand_count"
        ].iloc[0])
        expected_positive = int(split.summary.loc[
            split.summary["split"] == "train", "positive_positions"
        ].iloc[0])
        assert int(train_cpg_counts.totals["total_positions"].sum()) == expected_train
        assert len(train_cpg_counts.positives) == expected_positive
        assert train_cpg_counts.totals["gc_bin_code"].between(0, 8).all()
        assert train_cpg_counts.totals["cpg_bin_code"].between(0, 7).all()
        display(train_cpg_counts.totals.head(10))
        display(train_cpg_counts.positives.head(10))
        """
    )
    cells["folds"].source = source(
        """
        train_kmer_counts = kmer_counts_from_cpg_counts(train_cpg_counts)
        fold_assignment = make_balanced_contig_folds(
            train_kmer_counts,
            k=CONFIG.max_k,
            n_splits=CONFIG.cv_folds,
        )
        display(fold_assignment)
        assert not fold_assignment["contig"].duplicated().any()
        assert set(fold_assignment["fold"]) == set(range(CONFIG.cv_folds))
        """
    )
    cells["train-diagnostics-md"].source = source(
        """
        ## 4. Train-only диагностика CpG O/E201

        Это те же fixed bins, которые поступают в модель. График проверяет их
        численность и train-only enrichment до обучения.
        """
    )
    diagnostic = cells.pop("train-interaction-enrichment")
    diagnostic.id = "train-cpg-enrichment"
    diagnostic.source = source(
        """
        train_cpg_enrichment = cpg_enrichment_table(train_cpg_counts)
        display(train_cpg_enrichment)

        fig_train_cpg, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        x = np.arange(len(train_cpg_enrichment))
        axes[0].bar(x, train_cpg_enrichment["positions"], color="#78909C")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Train positions, log scale")
        axes[0].set_title("Support of diagnostic CpG O/E bins")
        axes[1].plot(x, train_cpg_enrichment["enrichment"], marker="o", color="#1967D2")
        axes[1].axhline(1.0, color="#5F6368", linestyle="--")
        axes[1].set_ylabel("Positive enrichment")
        axes[1].set_title("Train-only target enrichment")
        for ax in axes:
            ax.set_xticks(x, train_cpg_enrichment["cpg_oe_bin"], rotation=40, ha="right")
            ax.set_xlabel("CpG O/E201 diagnostic bin")
            ax.grid(axis="y", color="#E8EAED")
        plt.show()
        """
    )
    cells["build-aggregate"].source = source(
        """
        design = hierarchical_kmer_design(CONFIG)
        train_aggregate_preview = aggregate_cpg_likelihood(
            train_cpg_counts,
            design,
            config=CONFIG,
        )
        aggregate_report = pd.Series({
            "EXP-007_main_effect_columns": 4379,
            "CpG_OE201_bin_columns": 7,
            "candidate_columns": train_aggregate_preview.matrix.shape[1],
            "aggregate_binary_rows": len(train_aggregate_preview.rows),
            "represented_positions": int(train_aggregate_preview.rows["population_count"].sum()),
            "matrix_memory_mib": (
                train_aggregate_preview.matrix.data.nbytes
                + train_aggregate_preview.matrix.indices.nbytes
                + train_aggregate_preview.matrix.indptr.nbytes
            ) / 2**20,
        }, name="value").to_frame()
        display(aggregate_report)
        assert train_aggregate_preview.matrix.shape[1] == 4386
        assert int(train_aggregate_preview.rows["population_count"].sum()) == expected_train
        del train_aggregate_preview
        """
    )
    cells["show-grid"].source = source(
        """
        cv_plan = pd.Series({
            "folds": CONFIG.cv_folds,
            "C_values": len(CONFIG.C_grid),
            "planned_fits": CONFIG.cv_folds * len(CONFIG.C_grid),
            "C_grid": CONFIG.C_grid,
            "max_iter": CONFIG.max_iter,
            "tol": CONFIG.tol,
            "candidate_columns": 4386,
            "threadpoolctl_during_fit": False,
        }, name="value").to_frame()
        display(cv_plan)
        assert CONFIG.cv_folds * len(CONFIG.C_grid) == 15
        """
    )
    cells["stop-before-training"].source = source(
        """
        # STOP 1 — перед обучением

        Проверьте формулу CpG O/E, train enrichment, exact population,
        `candidate_columns=4386` и план `5 × 3 = 15 fits`. Следующая ячейка
        действительно обучает модели и печатает прогресс `01/15 … 15/15`.
        """
    )
    cells["run-train-cv"].source = source(
        """
        started = time.monotonic()
        cv_results = cross_validate_cpg_logistic(
            train_cpg_counts,
            fold_assignment,
            config=CONFIG,
            verbose=True,
        )
        cv_seconds = time.monotonic() - started
        assert len(cv_results) == 15
        print(f"Train-only CV finished in {cv_seconds:.2f} s")
        display(cv_results.head())
        """
    )
    cells["paired-cv-gate"].source = source(
        """
        EXP007_DIR = PROJECT_ROOT / "artifacts/experiments/EXP-007/run_001"
        exp007_metadata = json.loads((EXP007_DIR / "metadata.json").read_text(encoding="utf-8"))
        exp007_cv_results = pd.read_csv(EXP007_DIR / "cv_results.csv")
        EXP007_SELECTED_C = float(exp007_metadata["parameters"]["selected_C"])

        candidate_cv_selected = cv_results.loc[
            np.isclose(cv_results["C"], SELECTED_C),
            ["fold", "validation_contigs", "average_precision", "epic_spearman", "converged"],
        ].rename(columns={
            "average_precision": "candidate_average_precision",
            "epic_spearman": "candidate_epic_spearman",
            "converged": "candidate_converged",
        })
        exp007_cv_selected = exp007_cv_results.loc[
            np.isclose(exp007_cv_results["C"], EXP007_SELECTED_C),
            ["fold", "validation_contigs", "average_precision", "epic_spearman"],
        ].rename(columns={
            "validation_contigs": "exp007_validation_contigs",
            "average_precision": "exp007_average_precision",
            "epic_spearman": "exp007_epic_spearman",
        })
        paired_cv_gate = candidate_cv_selected.merge(
            exp007_cv_selected, on="fold", validate="one_to_one"
        )
        assert (
            paired_cv_gate["validation_contigs"]
            == paired_cv_gate["exp007_validation_contigs"]
        ).all(), "EXP-009 and EXP-007 must be compared on identical CV contigs"
        paired_cv_gate["ap_delta"] = (
            paired_cv_gate["candidate_average_precision"]
            - paired_cv_gate["exp007_average_precision"]
        )
        paired_cv_gate["relative_ap_delta"] = (
            paired_cv_gate["candidate_average_precision"]
            / paired_cv_gate["exp007_average_precision"] - 1
        )
        paired_cv_gate["spearman_delta"] = (
            paired_cv_gate["candidate_epic_spearman"]
            - paired_cv_gate["exp007_epic_spearman"]
        )
        display(paired_cv_gate.style.format({
            "candidate_average_precision": "{:.9f}",
            "exp007_average_precision": "{:.9f}",
            "ap_delta": "{:+.9f}",
            "relative_ap_delta": "{:+.2%}",
            "candidate_epic_spearman": "{:.6f}",
            "exp007_epic_spearman": "{:.6f}",
            "spearman_delta": "{:+.6f}",
        }))
        cv_gate_checks = pd.Series({
            "all candidate fits converged": paired_cv_gate["candidate_converged"].all(),
            "mean AP > EXP-007": (
                paired_cv_gate["candidate_average_precision"].mean()
                > paired_cv_gate["exp007_average_precision"].mean()
            ),
            "AP wins on at least 2/3 folds": int((paired_cv_gate["ap_delta"] > 0).sum()) >= 2,
            "mean Spearman >= EXP-007": (
                paired_cv_gate["candidate_epic_spearman"].mean()
                >= paired_cv_gate["exp007_epic_spearman"].mean()
            ),
        }, name="passed")
        CV_GATE_PASSED = bool(cv_gate_checks.all())
        display(cv_gate_checks.to_frame())

        fig_paired_cv, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(paired_cv_gate))
        width = 0.36
        ax.bar(x - width/2, paired_cv_gate["exp007_average_precision"], width,
               label="EXP-007", color="#7E57C2")
        ax.bar(x + width/2, paired_cv_gate["candidate_average_precision"], width,
               label="EXP-009", color="#1967D2")
        ax.set_xticks(x, [f"fold {value}" for value in paired_cv_gate["fold"]])
        ax.set_ylabel("Average Precision")
        ax.set_title("Paired train-CV gate against EXP-007")
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#E8EAED")
        plt.show()
        print("CV gate passed:", CV_GATE_PASSED)
        """
    )
    coefficient = cells.pop("cv-interaction-coefficients")
    coefficient.id = "cv-cpg-coefficients"
    coefficient.source = source(
        """
        selected_cv_cpg_coefs = selected_cv_cpg_coefficients(
            cv_results,
            C=SELECTED_C,
        )
        display(selected_cv_cpg_coefs.style.format({
            "coefficient": "{:+.6f}",
            "odds_multiplier": "{:.4f}",
        }))

        coefficient_grid = selected_cv_cpg_coefs.pivot(
            index="cpg_oe_bin", columns="fold", values="coefficient"
        ).reindex(CPG_DIAGNOSTIC_LABELS)
        limit = max(0.01, float(np.nanmax(np.abs(coefficient_grid.to_numpy()))))
        fig_cv_cpg, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        image = ax.imshow(coefficient_grid, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(coefficient_grid.columns)), coefficient_grid.columns)
        ax.set_yticks(range(len(coefficient_grid.index)), coefficient_grid.index)
        ax.set_xlabel("Fold")
        ax.set_ylabel("CpG O/E201 bin")
        ax.set_title("CpG-bin coefficient stability")
        fig_cv_cpg.colorbar(image, ax=ax, label="log-odds coefficient")
        plt.show()
        """
    )
    cells["stop-before-full-fit"].source = source(
        """
        # STOP 2 — paired CV gate

        Проверьте mean AP, число fold-побед, mean Spearman и устойчивость
        CpG-bin коэффициентов. Даже при непройденном gate эксперимент можно один раз
        довести до validation и сохранить, но такой run не получит `adopt`.
        """
    )
    cells["fit-full-train"].source = source(
        """
        print(f"Starting full-train fit with C={SELECTED_C:g} ...", flush=True)
        started = time.monotonic()
        final_fit = fit_cpg_logistic(
            train_cpg_counts,
            C=SELECTED_C,
            config=CONFIG,
            design=design,
        )
        full_fit_seconds = time.monotonic() - started
        display(final_fit.fit_report)
        assert bool(final_fit.fit_report.iloc[0]["converged"]), (
            "Final candidate did not converge; record the technical failure."
        )
        print(f"Full-train candidate fit in {full_fit_seconds:.2f} s")
        """
    )
    final_coef = cells.pop("inspect-final-interactions")
    final_coef.id = "inspect-final-cpg"
    final_coef.source = source(
        """
        final_cpg_coefficient = final_fit.coefficient_table.loc[
            final_fit.coefficient_table["feature_group"] == "CpG O/E201 bin"
        ].copy()
        final_cpg_coefficient["odds_multiplier"] = np.exp(
            final_cpg_coefficient["coefficient"]
        )
        display(final_cpg_coefficient)
        """
    )
    cells["validation-gate"].source = source(
        """
        # STOP 3 — зафиксированный одноразовый validation

        Формула CpG O/E, fixed bins, reference bin, grid, выбранный `C`,
        paired gate и final fit зафиксированы. После следующей ячейки ничего
        менять нельзя; новая форма CpG потребует EXP-010.
        """
    )
    cells["count-validation"].source = source(
        """
        started = time.monotonic()
        validation_cpg_counts = split_kmer_cpg_counts(
            split,
            sequences,
            "validation",
            config=CONFIG,
            verbose=True,
        )
        validation_count_seconds = time.monotonic() - started
        validation_kmer_counts = kmer_counts_from_cpg_counts(validation_cpg_counts)
        validation_count_diagnostics = pd.Series({
            "represented_positions": int(
                validation_cpg_counts.totals["total_positions"].sum()
            ),
            "joint_rows": len(validation_cpg_counts.totals),
            "positive_rows": len(validation_cpg_counts.positives),
        }, name="value").to_frame()
        display(validation_count_diagnostics)
        print(f"Validation counted in {validation_count_seconds:.2f} s")
        """
    )
    cells["evaluate-candidate"].source = source(
        """
        candidate_evaluation = evaluate_cpg_logistic(
            validation_cpg_counts,
            final_fit,
            config=CONFIG,
            variant="EXP-009 EXP-007 + CpG O/E201",
        )
        candidate_row = candidate_evaluation.metric_summary.iloc[0]
        display(candidate_evaluation.metric_summary)
        display(candidate_evaluation.per_contig_metrics)
        """
    )
    cells["compare-metrics"].source = source(
        """
        metric_summary = pd.DataFrame([
            {
                "variant": "EXP-003 6-mer lookup",
                "average_precision": float(exp003_row["average_precision"]),
                "epic_spearman": float(exp003_row["epic_spearman"]),
            },
            {
                "variant": "EXP-007 binned GC201",
                "average_precision": exp007_ap,
                "epic_spearman": exp007_spearman,
            },
            {
                "variant": "EXP-009 CpG O/E201",
                "average_precision": float(candidate_row["average_precision"]),
                "epic_spearman": float(candidate_row["epic_spearman"]),
            },
        ])
        display(metric_summary)

        exp003_contigs = lookup_evaluation.per_contig_metrics.loc[
            lookup_evaluation.per_contig_metrics["k"] == 6,
            ["segment_value", "average_precision", "epic_spearman"],
        ].rename(columns={
            "segment_value": "contig",
            "average_precision": "exp003_average_precision",
            "epic_spearman": "exp003_epic_spearman",
        })
        exp007_contigs = exp007_per_contig.rename(columns={
            "candidate_average_precision": "exp007_average_precision",
            "candidate_epic_spearman": "exp007_epic_spearman",
        })[["contig", "exp007_average_precision", "exp007_epic_spearman"]]
        candidate_contigs = candidate_evaluation.per_contig_metrics.rename(columns={
            "average_precision": "candidate_average_precision",
            "epic_spearman": "candidate_epic_spearman",
        })[["contig", "candidate_average_precision", "candidate_epic_spearman"]]
        per_contig_metrics = (
            exp003_contigs
            .merge(exp007_contigs, on="contig", validate="one_to_one")
            .merge(candidate_contigs, on="contig", validate="one_to_one")
        )
        display(per_contig_metrics)
        """
    )
    cells["validation-plots"].source = source(
        """
        colors = ["#9AA0A6", "#7E57C2", "#1967D2"]
        fig_metrics, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        for ax, metric, title in [
            (axes[0], "average_precision", "Average Precision"),
            (axes[1], "epic_spearman", "EPIC Spearman"),
        ]:
            bars = ax.bar(metric_summary["variant"], metric_summary[metric], color=colors)
            ax.bar_label(bars, fmt="%.7f", padding=3)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=12)
            ax.grid(axis="y", color="#E8EAED")
        plt.show()

        fig_contigs, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
        for ax, metric, label in [
            (axes[0], "average_precision", "Average Precision"),
            (axes[1], "epic_spearman", "EPIC Spearman"),
        ]:
            columns = [f"exp003_{metric}", f"exp007_{metric}", f"candidate_{metric}"]
            for _, row in per_contig_metrics.iterrows():
                values = [row[column] for column in columns]
                ax.plot(range(3), values, color="#AAB2C0")
                ax.scatter(range(3), values, color=colors)
                ax.text(2.04, values[-1], row["contig"].replace("NC_", ""), va="center")
            ax.set_xticks(range(3), ["EXP-003", "EXP-007", "EXP-009"])
            ax.set_ylabel(label)
            ax.set_title(f"Per-contig — {label}")
            ax.grid(axis="y", color="#E8EAED")
        plt.show()
        """
    )
    cells["pr-and-coefficients"].source = source(
        """
        reference_pr = lookup_evaluation.precision_recall.loc[
            lookup_evaluation.precision_recall["k"] == 6
        ].copy()
        candidate_pr = candidate_evaluation.precision_recall.copy()

        fig_pr, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot(reference_pr["recall"], reference_pr["precision"],
                color="#9AA0A6", label="EXP-003")
        ax.plot(exp007_pr["recall"], exp007_pr["precision"],
                color="#7E57C2", label="EXP-007")
        ax.plot(candidate_pr["recall"], candidate_pr["precision"],
                color="#1967D2", label="EXP-009 CpG O/E201")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Validation precision–recall")
        ax.legend(frameon=False)
        ax.grid(color="#E8EAED")
        plt.show()

        fig_coefficients, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        cv_mean_coefficients = selected_cv_cpg_coefs.groupby(
            "cpg_oe_bin", as_index=False, sort=False
        )["coefficient"].mean()
        axes[0].bar(
            cv_mean_coefficients["cpg_oe_bin"],
            cv_mean_coefficients["coefficient"],
            color="#00897B",
        )
        axes[0].axhline(0, color="#5F6368", linewidth=1)
        axes[0].set_title("Mean selected-CV CpG-bin coefficients")
        axes[1].bar(
            final_cpg_coefficient["category"],
            final_cpg_coefficient["coefficient"],
            color="#1967D2",
        )
        axes[1].axhline(0, color="#5F6368", linewidth=1)
        axes[1].set_title("Full-train CpG-bin coefficients")
        for ax in axes:
            ax.set_ylabel("log-odds contribution")
            ax.tick_params(axis="x", rotation=40)
            ax.grid(axis="y", color="#E8EAED")
        plt.show()
        """
    )
    cells["decision-md"].source = source(
        """
        ## 7. Решение

        EXP-007 — champion reference. EXP-008 показан только как сохранённый
        отрицательный эксперимент и не входит в candidate pipeline.
        """
    )
    cells["decision"].source = source(
        """
        candidate_ap = float(candidate_row["average_precision"])
        candidate_spearman = float(candidate_row["epic_spearman"])
        exp003_ap = float(exp003_row["average_precision"])
        exp003_spearman = float(exp003_row["epic_spearman"])

        champion_ap_wins = int((
            per_contig_metrics["candidate_average_precision"]
            > per_contig_metrics["exp007_average_precision"]
        ).sum())
        validation_checks = pd.Series({
            "relative AP gain vs EXP-007 >= 1%": candidate_ap / exp007_ap - 1 >= 0.01,
            "Spearman >= EXP-007": candidate_spearman >= exp007_spearman,
            "AP wins vs EXP-007 on at least 2/3 contigs": champion_ap_wins >= 2,
        }, name="passed")
        champion_checks = pd.concat([
            pd.Series({"train-CV gate passed": CV_GATE_PASSED}, name="passed"),
            validation_checks,
        ])
        display(Markdown("### Champion gate vs EXP-007"))
        display(champion_checks.to_frame())

        VALIDATION_CHECKS_PASSED = bool(validation_checks.all())
        if CV_GATE_PASSED and VALIDATION_CHECKS_PASSED:
            DECISION = "adopt"
        elif VALIDATION_CHECKS_PASSED:
            DECISION = "iterate"
        else:
            DECISION = "reject"
        INTERPRETATION = (
            f"CpG O/E201 changed AP vs EXP-007 from {exp007_ap:.9f} "
            f"to {candidate_ap:.9f} ({candidate_ap / exp007_ap - 1:+.2%}) and "
            f"Spearman from {exp007_spearman:.6f} to {candidate_spearman:.6f}; "
            f"train-CV gate passed: {CV_GATE_PASSED}; protocol: {PROTOCOL_STATUS}. "
            f"Decision: {DECISION}."
        )
        display(Markdown(f"**Decision: `{DECISION}`.** {INTERPRETATION}"))
        """
    )
    cells["save-md"].source = source(
        """
        ## 8. Сохранение

        Эта ячейка создаёт immutable `EXP-009/run_001` при любом содержательном
        исходе: `adopt`, `iterate` или `reject`.
        """
    )
    cells["save"].source = source(
        """
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        plot_objects = {
            "01_train_cpg_enrichment.png": fig_train_cpg,
            "02_train_cv_diagnostics.png": fig_cv,
            "03_paired_cv_gate.png": fig_paired_cv,
            "04_cv_cpg_coefficients.png": fig_cv_cpg,
            "05_validation_metrics.png": fig_metrics,
            "06_per_contig_metrics.png": fig_contigs,
            "07_precision_recall.png": fig_pr,
            "08_final_cpg_coefficient.png": fig_coefficients,
        }
        for filename, figure in plot_objects.items():
            figure.savefig(PLOT_DIR / filename, dpi=170, bbox_inches="tight", facecolor="white")

        criteria_table = champion_checks.rename("passed").to_frame().assign(
            comparison="EXP-007"
        ).rename_axis("criterion").reset_index()
        precision_recall = pd.concat([
            reference_pr.assign(variant="EXP-003 6-mer lookup"),
            exp007_pr.assign(variant="EXP-007 binned GC201"),
            candidate_pr.assign(variant="EXP-009 CpG O/E201"),
        ], ignore_index=True)
        tables = {
            "train_count_diagnostics.csv": train_count_diagnostics,
            "validation_count_diagnostics.csv": validation_count_diagnostics,
            "train_cpg_enrichment.csv": train_cpg_enrichment,
            "cv_fold_assignment.csv": fold_assignment,
            "cv_results.csv": cv_results,
            "cv_summary.csv": cv_summary,
            "paired_cv_gate.csv": paired_cv_gate,
            "cv_gate_checks.csv": cv_gate_checks.rename("passed").reset_index(names="criterion"),
            "selected_cv_cpg_coefficients.csv": selected_cv_cpg_coefs,
            "metric_summary.csv": metric_summary,
            "per_contig_metrics.csv": per_contig_metrics,
            "criteria.csv": criteria_table,
            "fit_report.csv": final_fit.fit_report,
            "coefficient_table.csv": final_fit.coefficient_table,
            "final_cpg_coefficient.csv": final_cpg_coefficient,
            "precision_recall.csv": precision_recall,
        }
        for filename, table in tables.items():
            table.to_csv(ARTIFACT_DIR / filename, index=False)
        candidate_evaluation.positive_predictions.to_csv(
            ARTIFACT_DIR / "positive_predictions.csv.gz", index=False, compression="gzip"
        )

        METRICS = {
            "reference_average_precision": exp007_ap,
            "candidate_average_precision": candidate_ap,
            "delta_average_precision": candidate_ap - exp007_ap,
            "reference_epic_spearman": exp007_spearman,
            "candidate_epic_spearman": candidate_spearman,
            "delta_epic_spearman": candidate_spearman - exp007_spearman,
            "historical_exp003_average_precision": exp003_ap,
            "historical_exp003_epic_spearman": exp003_spearman,
            "selected_cv_candidate_mean_average_precision": float(
                paired_cv_gate["candidate_average_precision"].mean()
            ),
            "selected_cv_reference_mean_average_precision": float(
                paired_cv_gate["exp007_average_precision"].mean()
            ),
        }
        PARAMETERS = CONFIG.to_dict() | {
            "selected_C": SELECTED_C,
            "reference_experiment": "EXP-007/run_001",
            "rejected_predecessor": "EXP-008/run_001",
            "historical_reference": "EXP-003/run_001",
            "cv_gate": (
                "mean AP > EXP-007; AP wins >=2/3 folds; "
                "mean Spearman >= EXP-007; all fits converged"
            ),
            "cv_gate_passed": CV_GATE_PASSED,
            "allow_exploratory_completion": ALLOW_EXPLORATORY_COMPLETION,
            "protocol_status": PROTOCOL_STATUS,
            "validation_population": "all whitelist positions",
            "validation_used_for_C_selection": False,
            "score": "raw float64 logistic decision_function",
        }
        record = build_run_record(
            SPEC,
            split,
            run_name=RUN_NAME,
            notebook_path=NOTEBOOK_PATH,
            parameters=PARAMETERS,
            metrics=METRICS,
            interpretation=INTERPRETATION,
            decision=DECISION,
        )
        output = save_run_record(PROJECT_ROOT, record)
        (output / "config.json").write_text(
            json.dumps(PARAMETERS, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        print("Saved run:", output)
        """
    )
    cells["sync-report-md"].source = source(
        """
        ## 9. Синхронизация отчёта

        Финальная ячейка читает сохранённые артефакты и обновляет карточку
        EXP-009, включая отрицательный результат.
        """
    )
    cells["sync-report"].source = source(
        """
        from ml_project.epic_cpg_report import sync_exp009_report

        experiment_note = sync_exp009_report(PROJECT_ROOT, run_name=RUN_NAME)
        print("Updated experiment note:", experiment_note)
        """
    )

    notebook.metadata["epic"] = {
        "experiment_id": "EXP-009",
        "status": "prepared_unexecuted",
        "reference_experiment": "EXP-007",
        "rejected_predecessor": "EXP-008",
        "historical_reference": "EXP-003",
        "candidate_feature": "fixed CpG O/E bins in 201 bp window",
        "planned_cv_fits": 15,
        "training_cells": ["run-train-cv", "fit-full-train"],
        "validation_gate": "validation-gate",
        "threadpoolctl_during_fit": False,
    }
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbformat.validate(notebook)
    return notebook


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(nbformat.writes(build_notebook()), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
