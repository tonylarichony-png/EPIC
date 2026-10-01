"""Build the deliberately unexecuted EXP-008 interaction notebook."""

from __future__ import annotations

from pathlib import Path
import copy
import textwrap

import nbformat

from build_exp007_gc201_bins_notebook import build_notebook as build_exp007_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "notebooks/experiments/EXP-008_k2_gc_interaction.ipynb"


def source(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


def build_notebook():
    notebook = copy.deepcopy(build_exp007_notebook())
    cells = {cell.id: cell for cell in notebook.cells}

    cells["title"].source = source(
        """
        # EXP-008 — Hierarchical LR + interaction(k2, GC-bin)

        **Статус:** подготовлен, но ни одна code cell не выполнена.

        EXP-007 становится новым champion. Здесь сохраняются все его main
        effects и добавляются только 128 reference-coded взаимодействий между
        strand-oriented динуклеотидом `[-1,0]` и GC201-bin.

        Не используйте **Run All**. Выполняйте ячейки сверху вниз.
        """
    )
    cells["roadmap"].source = source(
        """
        ## Карта эксперимента

        1. Воспроизвести exact `6-mer × GC201` counts EXP-007.
        2. Проверить train-only support и enrichment `k2 × GC-bin`.
        3. Собрать 4 379 main-effect и 128 interaction-столбцов.
        4. Выполнить только 5 значений `C × 3 folds = 15 fits`.
        5. Сравнить каждый fold с новым champion EXP-007.
        6. Открыть validation только после paired CV gate.
        7. Сохранить метрики, heatmaps взаимодействий и отчёт.
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

        # This must happen before NumPy/SciPy/sklearn are imported. The
        # environment used for EXP-007 can otherwise load incompatible Intel
        # and LLVM OpenMP runtimes into one process.
        _NUMERIC_STACK_ALREADY_LOADED = any(
            name == "numpy" or name.startswith("numpy.") for name in sys.modules
        )
        if (
            _NUMERIC_STACK_ALREADY_LOADED
            and os.environ.get("MKL_THREADING_LAYER", "").upper() != "SEQUENTIAL"
        ):
            raise RuntimeError(
                "Restart the kernel before EXP-008: the numeric stack was "
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
        from ml_project.epic_data import ensure_prepared_split
        from ml_project.epic_experiment import ExperimentSpec, build_run_record, save_run_record
        from ml_project.epic_gc_interaction_logistic import (
            GCInteractionLogisticConfig,
            K2_LABELS,
            aggregate_gc_interaction_likelihood,
            cross_validate_gc_interaction_logistic,
            evaluate_gc_interaction_logistic,
            fit_gc_interaction_logistic,
            interaction_enrichment_table,
            interaction_feature_table,
            selected_cv_interaction_coefficients,
        )
        from ml_project.epic_gc_binned_logistic import GC_BIN_LABELS
        from ml_project.epic_gc_logistic import (
            gc_count_diagnostics,
            kmer_counts_from_gc_counts,
            split_kmer_gc_counts,
        )
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
            "threadpoolctl_used_for_fit": False,
            "MKL_THREADING_LAYER": os.environ["MKL_THREADING_LAYER"],
            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        }, name="value").to_frame())
        """
    )
    cells["config-md"].source = source(
        """
        ## 1. Зафиксированный конфиг

        Main effects полностью совпадают с EXP-007. Interaction использует
        references `k2=TT` и `GC=[0.35,0.40)`, поэтому добавляется
        `(17−1) × (9−1) = 128` столбцов без линейной зависимости от main effects.

        Перед первой ячейкой перезапустите kernel, если в нём уже выполнялся
        EXP-007. Setup фиксирует безопасный MKL backend **до** импорта NumPy;
        это устраняет конфликт OpenMP, а не просто скрывает warning.

        Grid сокращён до пяти значений вокруг устойчивого оптимума `0.0003`:
        `0.0001, 0.0002, 0.0003, 0.0005, 0.001`.
        """
    )
    cells["config"].source = source(
        """
        RUN_NAME = "run_001"
        EXPERIMENT_ID = "EXP-008"
        NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/experiments/EXP-008_k2_gc_interaction.ipynb"
        ARTIFACT_DIR = PROJECT_ROOT / "artifacts/experiments/EXP-008" / RUN_NAME
        PLOT_DIR = ARTIFACT_DIR / "plots"

        CONFIG = GCInteractionLogisticConfig()
        LOOKUP_CONFIG = KmerLookupConfig(ks=CONFIG.ks, smoothing="one_expected_positive")
        SPEC = ExperimentSpec(
            experiment_id=EXPERIMENT_ID,
            title="Hierarchical LR + interaction(k2[-1,0], GC201-bin)",
            hypothesis=(
                "The usefulness of the start-site dinucleotide depends on the "
                "regional GC promoter class beyond additive EXP-007 effects"
            ),
            changed_variable=(
                "EXP-007 + 128 reference-coded k2[-1,0] x GC201-bin interactions"
            ),
            success_criterion=(
                "Before validation: AP win over EXP-007 on all 3 identical CV folds; "
                "validation: relative AP gain >= 1%, Spearman not lower, "
                "AP wins on at least 2/3 validation contigs"
            ),
            seed=CONFIG.seed,
        )
        assert len(CONFIG.C_grid) * CONFIG.cv_folds == 15
        assert not (ARTIFACT_DIR / "metadata.json").exists(), (
            f"Run already exists: {ARTIFACT_DIR}. Use another run name; do not overwrite."
        )
        display(pd.Series(CONFIG.to_dict(), name="value").to_frame())
        display(interaction_feature_table(CONFIG))
        """
    )
    cells["split-md"].source = source(
        """
        ## 2. Неизменный split

        Используется тот же `contig_holdout_v1`. Все diagnostics, выбор `C` и
        paired gate до validation выполняются только на train-contig.
        """
    )
    cells["counting-md"].source = source(
        """
        ## 3. Данные уже достаточны для interaction

        Новый проход для отдельного признака не нужен. `k2[-1,0]` является
        suffix уже посчитанного strand-oriented 6-mer, а GC-bin получается из
        `gc_count/valid_count`. Exact multiplicity по-прежнему передаётся через
        `sample_weight`.
        """
    )
    cells["train-diagnostics-md"].source = source(
        """
        ## 4. Train-only support `k2 × GC-bin`

        Heatmaps показывают численность и marginal enrichment сочетаний. Они не
        заменяют CV, но помогают обнаружить пустые и крайне редкие клетки до
        добавления interaction-параметров.
        """
    )
    diagnostic_cell = cells["train-bin-enrichment"]
    diagnostic_cell.id = "train-interaction-enrichment"
    diagnostic_cell.source = source(
        """
        train_interaction_enrichment = interaction_enrichment_table(
            train_gc_counts,
            design=hierarchical_kmer_design(CONFIG),
        )
        display(
            train_interaction_enrichment.sort_values(
                ["positives", "positions"], ascending=False
            ).head(40)
        )

        support_heatmap = train_interaction_enrichment.pivot(
            index="k2", columns="gc_bin", values="positives"
        ).reindex(index=K2_LABELS, columns=GC_BIN_LABELS)
        enrichment_heatmap = train_interaction_enrichment.pivot(
            index="k2", columns="gc_bin", values="enrichment"
        ).reindex(index=K2_LABELS, columns=GC_BIN_LABELS)

        fig_train_interactions, axes = plt.subplots(
            1, 2, figsize=(16, 7), constrained_layout=True
        )
        images = [
            axes[0].imshow(np.log10(support_heatmap.to_numpy() + 1), aspect="auto", cmap="Greys"),
            axes[1].imshow(enrichment_heatmap.to_numpy(), aspect="auto", cmap="viridis"),
        ]
        titles = ["log10(positive support + 1)", "Positive enrichment"]
        for ax, image, title in zip(axes, images, titles):
            ax.set_xticks(range(len(GC_BIN_LABELS)), GC_BIN_LABELS, rotation=45, ha="right")
            ax.set_yticks(range(len(K2_LABELS)), K2_LABELS)
            ax.set_title(title)
            ax.set_xlabel("GC201 bin")
            ax.set_ylabel("k2[-1,0]")
            fig_train_interactions.colorbar(image, ax=ax, shrink=0.8)
        plt.show()
        """
    )
    cells["build-aggregate"].source = source(
        """
        design = hierarchical_kmer_design(CONFIG)
        train_aggregate_preview = aggregate_gc_interaction_likelihood(
            train_gc_counts,
            design,
            config=CONFIG,
        )
        aggregate_report = pd.Series({
            "EXP-007_main_effect_columns": 4379,
            "interaction_columns": len(interaction_feature_table(CONFIG)),
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
        assert train_aggregate_preview.matrix.shape[1] == 4507
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
            "threadpoolctl_during_fit": False,
        }, name="value").to_frame()
        display(cv_plan)
        assert CONFIG.cv_folds * len(CONFIG.C_grid) == 15
        """
    )
    cells["stop-before-training"].source = source(
        """
        # STOP 1 — дальше ровно 15 обучений

        До этой точки модели не обучались. Проверьте heatmaps, support редких
        сочетаний, `candidate_columns=4507` и grid из пяти `C`. Следующая ячейка
        напечатает фактический прогресс `1/15…15/15`.
        """
    )
    cells["run-train-cv"].source = source(
        """
        started = time.monotonic()
        cv_results = cross_validate_gc_interaction_logistic(
            train_gc_counts,
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
    cells["select-c"].source = source(
        """
        selected = select_regularization(
            cv_summary,
            relative_ap_tolerance=CONFIG.near_best_relative_ap,
        )
        SELECTED_C = float(selected["C"])
        display(selected.to_frame("selected_value"))
        print(f"Frozen candidate C before validation: {SELECTED_C:g}")
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
            ["fold", "average_precision", "epic_spearman"],
        ].rename(columns={
            "average_precision": "exp007_average_precision",
            "epic_spearman": "exp007_epic_spearman",
        })
        paired_cv_gate = candidate_cv_selected.merge(
            exp007_cv_selected, on="fold", validate="one_to_one"
        )
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
        CV_GATE_PASSED = bool(
            paired_cv_gate["candidate_converged"].all()
            and (paired_cv_gate["ap_delta"] > 0).all()
            and paired_cv_gate["candidate_epic_spearman"].mean()
                >= paired_cv_gate["exp007_epic_spearman"].mean()
        )

        fig_paired_cv, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(paired_cv_gate))
        width = 0.36
        ax.bar(x - width/2, paired_cv_gate["exp007_average_precision"], width,
               label="EXP-007", color="#7E57C2")
        ax.bar(x + width/2, paired_cv_gate["candidate_average_precision"], width,
               label="EXP-008", color="#1967D2")
        ax.set_xticks(x, [f"fold {value}" for value in paired_cv_gate["fold"]])
        ax.set_ylabel("Average Precision")
        ax.set_title("Paired train-CV gate против EXP-007")
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#E8EAED")
        plt.show()
        print("CV gate passed:", CV_GATE_PASSED)
        """
    )
    coefficient_cell = cells["cv-bin-coefficients"]
    coefficient_cell.id = "cv-interaction-coefficients"
    coefficient_cell.source = source(
        """
        selected_cv_interaction_coefs = selected_cv_interaction_coefficients(
            cv_results,
            C=SELECTED_C,
            config=CONFIG,
        )
        interaction_stability = (
            selected_cv_interaction_coefs
            .groupby(["interaction_index", "k2", "gc_bin"], as_index=False)
            .agg(
                mean_coefficient=("coefficient", "mean"),
                std_coefficient=("coefficient", "std"),
                min_coefficient=("coefficient", "min"),
                max_coefficient=("coefficient", "max"),
            )
        )
        interaction_stability["mean_absolute_coefficient"] = interaction_stability[
            "mean_coefficient"
        ].abs()
        display(interaction_stability.nlargest(30, "mean_absolute_coefficient"))

        fig_cv_interactions, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        coefficient_limit = selected_cv_interaction_coefs["coefficient"].abs().max()
        nonreference_k2 = [label for label in K2_LABELS if label != "TT"]
        nonreference_gc = [label for label in GC_BIN_LABELS if label != "[0.35,0.40)"]
        for ax, (fold, frame) in zip(axes, selected_cv_interaction_coefs.groupby("fold")):
            pivot = frame.pivot(index="k2", columns="gc_bin", values="coefficient").reindex(
                index=nonreference_k2, columns=nonreference_gc
            )
            image = ax.imshow(
                pivot.to_numpy(), aspect="auto", cmap="coolwarm",
                vmin=-coefficient_limit, vmax=coefficient_limit,
            )
            ax.set_xticks(range(len(nonreference_gc)), nonreference_gc, rotation=45, ha="right")
            ax.set_yticks(range(len(nonreference_k2)), nonreference_k2)
            ax.set_title(f"fold {fold}")
            ax.set_xlabel("GC201 bin")
            ax.set_ylabel("k2[-1,0]")
        fig_cv_interactions.colorbar(image, ax=axes, shrink=0.75, label="Interaction coefficient")
        plt.show()
        """
    )
    cells["stop-before-full-fit"].source = source(
        """
        # STOP 2 — paired CV gate

        Проверьте AP каждого fold, mean Spearman и три heatmap interaction.
        Если строгий gate не пройден, EXP-008 нельзя автоматически принять.
        После явного решения можно один раз довести уже зафиксированный
        candidate до validation в режиме `exploratory_after_failed_cv_gate`,
        чтобы отрицательный эксперимент также был полностью сохранён.
        """
    )
    cells["enforce-cv-gate"].source = source(
        """
        ALLOW_EXPLORATORY_COMPLETION = True
        PROTOCOL_STATUS = (
            "confirmatory"
            if CV_GATE_PASSED
            else "exploratory_after_failed_cv_gate"
        )
        if not CV_GATE_PASSED:
            assert ALLOW_EXPLORATORY_COMPLETION
            display(Markdown(
                "**CV gate не пройден.** Продолжаем ровно один раз только для "
                "сохранения полного отрицательного результата. Такой run не "
                "может получить решение `adopt`."
            ))
        print("Protocol status:", PROTOCOL_STATUS)
        """
    )
    cells["fit-full-train"].source = source(
        """
        print(f"Starting full-train fit with C={SELECTED_C:g} ...", flush=True)
        started = time.monotonic()
        final_fit = fit_gc_interaction_logistic(
            train_gc_counts,
            C=SELECTED_C,
            config=CONFIG,
            design=design,
        )
        full_fit_seconds = time.monotonic() - started
        display(final_fit.fit_report)
        assert bool(final_fit.fit_report.iloc[0]["converged"]), (
            "Final candidate did not converge; do not open validation."
        )
        print(f"Full-train candidate fit in {full_fit_seconds:.2f} s")
        """
    )
    final_coef_cell = cells["inspect-final-bin-coefficients"]
    final_coef_cell.id = "inspect-final-interactions"
    final_coef_cell.source = source(
        """
        final_interactions = interaction_feature_table(CONFIG).merge(
            final_fit.coefficient_table[
                ["feature_name", "coefficient", "absolute_coefficient"]
            ],
            on="feature_name",
            validate="one_to_one",
        )
        final_interactions["odds_multiplier"] = np.exp(final_interactions["coefficient"])
        display(final_interactions.nlargest(30, "absolute_coefficient"))

        final_interaction_heatmap = final_interactions.pivot(
            index="k2", columns="gc_bin", values="coefficient"
        ).reindex(index=nonreference_k2, columns=nonreference_gc)
        """
    )
    cells["validation-gate"].source = source(
        """
        # STOP 3 — зафиксированный одноразовый validation

        Зафиксированы vocabulary из 128 interactions, grid из пяти `C`,
        выбранный `C`, результат paired gate, protocol status и final fit.
        Если CV gate не пройден, validation интерпретируется только как
        exploratory. После следующей ячейки ничего менять нельзя; новая
        настройка потребует EXP-009.
        """
    )
    cells["evaluate-candidate"].source = source(
        """
        candidate_evaluation = evaluate_gc_interaction_logistic(
            validation_gc_counts,
            final_fit,
            config=CONFIG,
            variant="EXP-008 LR + k2 x GC-bin",
        )
        candidate_row = candidate_evaluation.metric_summary.iloc[0]
        display(candidate_evaluation.metric_summary)
        display(candidate_evaluation.per_contig_metrics)
        """
    )
    cells["load-references"].source = source(
        """
        exp007_per_contig = pd.read_csv(EXP007_DIR / "per_contig_metrics.csv")
        exp007_pr_all = pd.read_csv(EXP007_DIR / "precision_recall.csv")
        exp007_pr = exp007_pr_all.loc[
            exp007_pr_all["variant"] == "EXP-007 binned GC201"
        ].copy()
        exp007_ap = float(exp007_metadata["metrics"]["candidate_average_precision"])
        exp007_spearman = float(exp007_metadata["metrics"]["candidate_epic_spearman"])
        display(pd.Series({
            "EXP-007 AP": exp007_ap,
            "EXP-007 Spearman": exp007_spearman,
            "EXP-007 decision": exp007_metadata["run"]["decision"],
        }, name="value").to_frame())
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
                "variant": "EXP-008 k2 x GC-bin",
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
            ax.set_xticks(range(3), ["EXP-003", "EXP-007", "EXP-008"])
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
        ]
        candidate_pr = candidate_evaluation.precision_recall
        fig_pr, ax = plt.subplots(figsize=(9, 5))
        ax.step(reference_pr["recall"], reference_pr["precision"], where="post",
                color="#9AA0A6", label="EXP-003 lookup")
        ax.step(exp007_pr["recall"], exp007_pr["precision"], where="post",
                color="#7E57C2", label="EXP-007")
        ax.step(candidate_pr["recall"], candidate_pr["precision"], where="post",
                color="#1967D2", label="EXP-008 interaction")
        ax.axhline(candidate_row["prevalence"], color="#5F6368", linestyle="--",
                   label=f"prevalence={candidate_row['prevalence']:.7f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Полная validation Precision–Recall")
        ax.legend(frameon=False)
        ax.grid(color="#E8EAED")
        plt.show()

        strongest = final_interactions.nlargest(20, "absolute_coefficient").sort_values("coefficient")
        fig_coefficients, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
        limit = np.nanmax(np.abs(final_interaction_heatmap.to_numpy()))
        image = axes[0].imshow(
            final_interaction_heatmap.to_numpy(), aspect="auto", cmap="coolwarm",
            vmin=-limit, vmax=limit,
        )
        axes[0].set_xticks(range(len(nonreference_gc)), nonreference_gc, rotation=45, ha="right")
        axes[0].set_yticks(range(len(nonreference_k2)), nonreference_k2)
        axes[0].set_title("Финальные k2 × GC-bin interactions")
        axes[0].set_xlabel("GC201 bin")
        axes[0].set_ylabel("k2[-1,0]")
        fig_coefficients.colorbar(image, ax=axes[0], shrink=0.8)
        axes[1].barh(
            strongest["feature_name"], strongest["coefficient"],
            color=np.where(strongest["coefficient"] >= 0, "#1967D2", "#D93025"),
        )
        axes[1].axvline(0, color="#202124", linewidth=1)
        axes[1].set_xlabel("Additional log-odds beyond main effects")
        axes[1].set_title("20 сильнейших interactions")
        plt.show()
        """
    )
    cells["decision-md"].source = source(
        """
        ## 7. Формальное решение

        EXP-007 — новый champion и единственный основной ablation reference.
        EXP-003 остаётся исторической точкой отсчёта.
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
            f"k2 x GC-bin interactions changed AP vs EXP-007 from {exp007_ap:.9f} "
            f"to {candidate_ap:.9f} ({candidate_ap / exp007_ap - 1:+.2%}) and "
            f"Spearman from {exp007_spearman:.6f} to {candidate_spearman:.6f}; "
            f"vs historical EXP-003 AP change is {candidate_ap / exp003_ap - 1:+.2%}. "
            f"Train-CV gate passed: {CV_GATE_PASSED}; protocol: {PROTOCOL_STATUS}. "
            f"Decision: {DECISION}."
        )
        display(Markdown(f"**Decision: `{DECISION}`.** {INTERPRETATION}"))
        """
    )
    cells["save-md"].source = source(
        """
        ## 8. Сохранение

        Только эта ячейка создаёт immutable run `EXP-008/run_001`.
        """
    )
    cells["save"].source = source(
        """
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        plot_objects = {
            "01_train_interaction_enrichment.png": fig_train_interactions,
            "02_train_cv_diagnostics.png": fig_cv,
            "03_paired_cv_gate.png": fig_paired_cv,
            "04_cv_interaction_coefficients.png": fig_cv_interactions,
            "05_validation_metrics.png": fig_metrics,
            "06_per_contig_metrics.png": fig_contigs,
            "07_precision_recall.png": fig_pr,
            "08_final_interactions.png": fig_coefficients,
        }
        for filename, figure in plot_objects.items():
            figure.savefig(PLOT_DIR / filename, dpi=170, bbox_inches="tight", facecolor="white")

        criteria_table = champion_checks.rename("passed").to_frame().assign(
            comparison="EXP-007"
        ).rename_axis("criterion").reset_index()
        precision_recall = pd.concat([
            reference_pr.assign(variant="EXP-003 6-mer lookup"),
            exp007_pr.assign(variant="EXP-007 binned GC201"),
            candidate_pr.assign(variant="EXP-008 k2 x GC-bin"),
        ], ignore_index=True)
        tables = {
            "train_gc_diagnostics.csv": train_gc_diagnostics,
            "validation_gc_diagnostics.csv": validation_gc_diagnostics,
            "train_interaction_enrichment.csv": train_interaction_enrichment,
            "cv_fold_assignment.csv": fold_assignment,
            "cv_results.csv": cv_results,
            "cv_summary.csv": cv_summary,
            "paired_cv_gate.csv": paired_cv_gate,
            "selected_cv_interaction_coefficients.csv": selected_cv_interaction_coefs,
            "interaction_stability.csv": interaction_stability,
            "metric_summary.csv": metric_summary,
            "per_contig_metrics.csv": per_contig_metrics,
            "criteria.csv": criteria_table,
            "fit_report.csv": final_fit.fit_report,
            "coefficient_table.csv": final_fit.coefficient_table,
            "final_interactions.csv": final_interactions,
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
            "historical_reference": "EXP-003/run_001",
            "cv_gate": "AP > EXP-007 on every fold and mean Spearman >= EXP-007",
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

        Финальная ячейка только читает сохранённые артефакты и обновляет
        автоматический блок карточки EXP-008.
        """
    )
    cells["sync-report"].source = source(
        """
        from ml_project.epic_gc_interaction_report import sync_exp008_report

        experiment_note = sync_exp008_report(PROJECT_ROOT, run_name=RUN_NAME)
        print("Updated experiment note:", experiment_note)
        """
    )

    notebook.metadata["epic"] = {
        "experiment_id": "EXP-008",
        "status": "prepared_unexecuted",
        "reference_experiment": "EXP-007",
        "historical_reference": "EXP-003",
        "candidate_feature": "k2[-1,0] x fixed GC201-bin interaction",
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
    # Use an explicit encoding: this project contains Russian explanatory text
    # and the Windows system locale may otherwise pass a legacy code page to
    # file-opening helpers.
    OUTPUT.write_text(nbformat.writes(build_notebook()), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
