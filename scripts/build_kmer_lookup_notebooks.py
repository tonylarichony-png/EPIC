"""Build the three readable whole-k-mer lookup experiment notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    {
        "id": "EXP-002",
        "candidate_k": 4,
        "reference_k": 2,
        "title": "4-mer lookup",
        "hypothesis": (
            "Цельное ориентированное слово [-3,0] содержит переносимый сигнал "
            "сверх динуклеотида [-1,0]."
        ),
        "change": "Только длина цельного слова: 2-mer → 4-mer.",
        "filename": "EXP-002_4mer_lookup.ipynb",
        "reference_experiment": "EXP-001 / 2-mer bridge-control",
    },
    {
        "id": "EXP-003",
        "candidate_k": 6,
        "reference_k": 4,
        "title": "6-mer lookup",
        "hypothesis": (
            "Две дополнительные upstream-буквы в слове [-5,0] содержат "
            "переносимый сигнал сверх 4-mer."
        ),
        "change": "Только длина цельного слова: 4-mer → 6-mer.",
        "filename": "EXP-003_6mer_lookup.ipynb",
        "reference_experiment": "EXP-002",
    },
    {
        "id": "EXP-004",
        "candidate_k": 8,
        "reference_k": 6,
        "title": "8-mer lookup",
        "hypothesis": (
            "Цельное слово [-7,0] содержит полезные локальные мотивы сверх "
            "6-mer, а fixed hierarchical backoff сдерживает редкие категории."
        ),
        "change": "Только длина цельного слова: 6-mer → 8-mer.",
        "filename": "EXP-004_8mer_lookup.ipynb",
        "reference_experiment": "EXP-003",
    },
)


def markdown(cell_id: str, source: str):
    cell = nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")
    cell["id"] = cell_id
    return cell


def code(cell_id: str, source: str):
    cell = nbf.v4.new_code_cell(dedent(source).strip() + "\n")
    cell["id"] = cell_id
    return cell


def render(source: str, experiment: dict[str, object]) -> str:
    replacements = {
        "__EXPERIMENT_ID__": str(experiment["id"]),
        "__TITLE__": str(experiment["title"]),
        "__HYPOTHESIS__": str(experiment["hypothesis"]),
        "__CHANGE__": str(experiment["change"]),
        "__FILENAME__": str(experiment["filename"]),
        "__REFERENCE_EXPERIMENT__": str(experiment["reference_experiment"]),
        "__REFERENCE_K__": str(experiment["reference_k"]),
        "__CANDIDATE_K__": str(experiment["candidate_k"]),
        "__WINDOW_START__": str(-(int(experiment["candidate_k"]) - 1)),
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def build_notebook(experiment: dict[str, object]):
    candidate_k = int(experiment["candidate_k"])
    reference_k = int(experiment["reference_k"])
    exp_id = str(experiment["id"])
    title = str(experiment["title"])
    hypothesis = str(experiment["hypothesis"])
    change = str(experiment["change"])

    def md(cell_id: str, source: str):
        return markdown(cell_id, render(source, experiment))

    def py(cell_id: str, source: str):
        return code(cell_id, render(source, experiment))

    cells = [
        md(
            "title",
            """
            # __EXPERIMENT_ID__ — __TITLE__

            **Гипотеза.** __HYPOTHESIS__

            **Единственное изменение.** __CHANGE__

            **Reference.** __REFERENCE_EXPERIMENT__.

            **Критерий принятия, зафиксированный до validation:** candidate
            принимается, только если полный AP вырос минимум на 1% относительно,
            EPIC Spearman не снизился и AP улучшился хотя бы на двух из трёх
            validation-contig.

            Официальный test не читается. Все основные метрики вычисляются на
            полном `contig_holdout_v1`, без отрицательного subsampling.
            """,
        ),
        md(
            "protocol",
            """
            ## 1. Протокол серии до запуска

            В EXP-002/003/004 меняется только длина `k`. Во всех notebooks
            зафиксированы одинаковые:

            - target `int(pooled_count > 0)`;
            - train/validation-contig и seed 42;
            - ориентация окна по предсказываемой цепи;
            - цельное категориальное слово с последней буквой в offset `0`;
            - `N/edge` как отдельная категория;
            - train-only hierarchical backoff;
            - точные full-population AP и dense-rank EPIC Spearman.

            Здесь нет one-hot, solver, эпох и loss-кривой. Обучение — точный
            подсчёт категорий и применение заранее заданной формулы сглаживания.
            Вместо фиктивной training loss показываются support, shrinkage,
            coverage и время полного прохода.
            """,
        ),
        py(
            "setup",
            r"""
            from pathlib import Path
            import hashlib
            import json
            import platform
            import sys
            import time

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

            from ml_project.epic_baseline import (
                DinucleotideBaselineConfig,
                category_summary as dinucleotide_category_summary,
                evaluate_dinucleotide_baseline,
                fit_dinucleotide_baseline,
                load_baseline_sequences,
            )
            from ml_project.epic_data import ensure_prepared_split
            from ml_project.epic_experiment import (
                ExperimentSpec,
                build_run_record,
                save_run_record,
            )
            from ml_project.epic_kmer import (
                KmerLookupConfig,
                SUPPORTED_K,
                evaluate_kmer_lookup,
                fit_kmer_lookup,
                kmer_category_labels,
                split_kmer_counts,
            )

            EXPERIMENT_ID = "__EXPERIMENT_ID__"
            REFERENCE_K = __REFERENCE_K__
            CANDIDATE_K = __CANDIDATE_K__
            CHAIN_KS = tuple(k for k in SUPPORTED_K if k <= CANDIDATE_K)
            RUN_NAME = "run_001"

            SPEC = ExperimentSpec(
                experiment_id=EXPERIMENT_ID,
                title="__TITLE__",
                hypothesis="__HYPOTHESIS__",
                changed_variable="__CHANGE__",
                success_criterion=(
                    "relative validation AP gain >= 1%; candidate EPIC Spearman "
                    ">= reference; AP improves on at least 2 of 3 validation contigs"
                ),
                seed=42,
            )
            CONFIG = KmerLookupConfig(
                ks=CHAIN_KS,
                smoothing="one_expected_positive",
            )
            NOTEBOOK_PATH = (
                PROJECT_ROOT / "notebooks/experiments/__FILENAME__"
            )
            ARTIFACT_DIR = (
                PROJECT_ROOT / "artifacts/experiments" / EXPERIMENT_ID / RUN_NAME
            )
            PLOT_DIR = ARTIFACT_DIR / "plots"
            PLOT_DIR.mkdir(parents=True, exist_ok=True)

            plt.rcParams.update({
                "figure.dpi": 120,
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titleweight": "bold",
            })
            pd.set_option("display.max_columns", 40)
            pd.set_option("display.max_rows", 40)
            COLORS = {"reference": "#9AA0A6", "candidate": "#1967D2"}

            def show_plot(fig, filename):
                path = PLOT_DIR / filename
                fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
                plt.show()
                plt.close(fig)
                return path

            display(pd.Series({
                "python": platform.python_version(),
                "executable": sys.executable,
                "experiment": EXPERIMENT_ID,
                "reference_k": REFERENCE_K,
                "candidate_k": CANDIDATE_K,
                "chain": str(CHAIN_KS),
                "run": RUN_NAME,
            }, name="value").to_frame())
            """,
        ),
        md(
            "data-contract-md",
            """
            ## 2. Неизменный data contract

            Загружается тот же проверенный `contig_holdout_v1`, что в EXP-001.
            Обе цепи каждого contig остаются вместе. Test показан только как
            часть manifest; его target отсутствует и нигде далее не используется.
            """,
        ),
        py(
            "data-contract",
            """
            split = ensure_prepared_split(PROJECT_ROOT)
            display(split.contig_assignment.sort_values(["split", "contig"]))
            display(split.summary)
            assert split.config.split_version == "contig_holdout_v1"
            """,
        ),
        md(
            "estimator-md",
            r"""
            ## 3. Что именно означает lookup и backoff

            Для `k=__CANDIDATE_K__` одна категория — всё ориентированное слово
            offsets `[__WINDOW_START__, …, 0]`. На minus-цепи extractor
            делает reverse-complement, поэтому смысл offsets одинаков на обеих
            цепях. `N/edge` означает неизвестную букву либо край **contig**, а не
            край whitelist-интервала.

            Пусть `p` — train prevalence. До validation фиксируем
            `tau = 1/p`, то есть prior содержит одну ожидаемую positive-позицию:

            `score = (positive + tau × parent_score) / (total + tau)`.

            Канонический backoff удаляет самые дальние upstream-буквы:
            `8 → 6 → 4 → 2 → p`. Невстреченное слово получает ровно parent score.
            У `N/edge` нет однозначного suffix, поэтому его prior всегда равен
            глобальной train prevalence, но сама категория обучается по своим
            counts. Validation не выбирает `tau` и не меняет формулу.
            """,
        ),
        py(
            "exact-counts",
            """
            runtime_rows = []

            started = time.monotonic()
            sequences, fasta_alphabet = load_baseline_sequences(split)
            runtime_rows.append({
                "stage": "load_fasta", "seconds": time.monotonic() - started
            })

            started = time.monotonic()
            train_counts = split_kmer_counts(
                split,
                sequences,
                "train",
                ks=CHAIN_KS,
                batch_size=CONFIG.count_batch_size,
            )
            runtime_rows.append({
                "stage": "count_full_train", "seconds": time.monotonic() - started
            })

            started = time.monotonic()
            validation_counts = split_kmer_counts(
                split,
                sequences,
                "validation",
                ks=CHAIN_KS,
                batch_size=CONFIG.count_batch_size,
            )
            runtime_rows.append({
                "stage": "count_full_validation",
                "seconds": time.monotonic() - started,
            })

            train_population = (
                train_counts.totals.groupby("k")["total_positions"].sum()
            )
            validation_population = (
                validation_counts.totals.groupby("k")["total_positions"].sum()
            )
            count_audit = pd.DataFrame({
                "k": CHAIN_KS,
                "possible_categories": [4**k + 1 for k in CHAIN_KS],
                "train_positions": [int(train_population.loc[k]) for k in CHAIN_KS],
                "train_positives": [len(train_counts.positives)] * len(CHAIN_KS),
                "validation_positions": [
                    int(validation_population.loc[k]) for k in CHAIN_KS
                ],
                "validation_positives": [
                    len(validation_counts.positives)
                ] * len(CHAIN_KS),
            })
            display(count_audit)

            expected = split.summary.set_index("split")
            assert count_audit["train_positions"].eq(
                int(expected.loc["train", "position_strand_count"])
            ).all()
            assert count_audit["validation_positions"].eq(
                int(expected.loc["validation", "position_strand_count"])
            ).all()
            print("Audit passed: каждый k покрывает полный train и validation.")
            """,
        ),
        md(
            "fit-md",
            """
            ## 4. Train-only score tables

            Эта операция не оптимизирует коэффициенты. Она применяет одну
            зафиксированную формулу к полным category counts. В таблице support
            особенно важны `zero_positive_categories`: их становится много у
            8-mer, поэтому сырой rate без backoff был бы нестабилен.
            """,
        ),
        py(
            "fit",
            """
            started = time.monotonic()
            fit = fit_kmer_lookup(train_counts, CONFIG)
            runtime_rows.append({
                "stage": "fit_lookup_tables", "seconds": time.monotonic() - started
            })

            display(fit.smoothing_report)
            display(fit.support_summary)

            expected_tau = 1.0 / fit.global_prevalence
            np.testing.assert_allclose(
                fit.smoothing_report["strength"].to_numpy(), expected_tau
            )
            assert fit.smoothing_report["method"].eq(
                "one_expected_positive"
            ).all()
            print(f"Train prevalence: {fit.global_prevalence:.9f}")
            print(f"Fixed tau: {expected_tau:.6f}")
            """,
        ),
        md(
            "evaluation-md",
            """
            ## 5. Точная полная validation-оценка

            AP считается по category histogram с объединением одинаковых scores,
            поэтому он в точности соответствует развёрнутым десяткам миллионов
            строк. Spearman использует все положительные validation-позиции и
            dense ranks исходного pooled count.

            Scores остаются `float64` и не квантуются: это сохраняет протокол
            EXP-001. Submission-aware rank transform и округление до пяти знаков
            должны проверяться отдельным экспериментом.
            """,
        ),
        py(
            "evaluation",
            """
            started = time.monotonic()
            evaluation = evaluate_kmer_lookup(validation_counts, fit)
            runtime_rows.append({
                "stage": "evaluate_full_validation",
                "seconds": time.monotonic() - started,
            })
            runtime_report = pd.DataFrame(runtime_rows)

            overall = evaluation.metric_summary.set_index("k")
            reference_row = overall.loc[REFERENCE_K]
            candidate_row = overall.loc[CANDIDATE_K]
            pair_metric_summary = pd.DataFrame([
                {
                    "metric": "Average Precision",
                    "direction": "maximize",
                    "reference": float(reference_row["average_precision"]),
                    "candidate": float(candidate_row["average_precision"]),
                    "delta": float(
                        candidate_row["average_precision"]
                        - reference_row["average_precision"]
                    ),
                    "relative_delta": float(
                        candidate_row["average_precision"]
                        / reference_row["average_precision"] - 1
                    ),
                },
                {
                    "metric": "EPIC Spearman",
                    "direction": "maximize",
                    "reference": float(reference_row["epic_spearman"]),
                    "candidate": float(candidate_row["epic_spearman"]),
                    "delta": float(
                        candidate_row["epic_spearman"]
                        - reference_row["epic_spearman"]
                    ),
                    "relative_delta": float(
                        candidate_row["epic_spearman"]
                        / reference_row["epic_spearman"] - 1
                    ),
                },
            ])
            display(
                pair_metric_summary.style.format({
                    "reference": "{:.9f}",
                    "candidate": "{:.9f}",
                    "delta": "{:+.9f}",
                    "relative_delta": "{:+.2%}",
                })
            )
            display(runtime_report.style.format({"seconds": "{:.3f}"}))
            """,
        ),
        md(
            "bridge-md",
            """
            ### Bridge к EXP-001

            В EXP-002 отдельно восстанавливается исходная L2 logistic regression
            по тому же динуклеотиду. Если 2-mer lookup сохраняет тот же порядок
            17 категорий, AP и Spearman совпадут. Эта проверка нужна, чтобы
            изменение estimator не маскировалось под эффект длины слова.

            В EXP-003/004 bridge уже не нужен: reference и candidate используют
            один и тот же lookup estimator.
            """,
        ),
        py(
            "bridge",
            """
            bridge_control = pd.DataFrame()
            if EXPERIMENT_ID == "EXP-002":
                labels2 = dict(enumerate(kmer_category_labels(2)))
                train_dinuc_totals = train_counts.totals.loc[
                    train_counts.totals["k"] == 2
                ].copy()
                train_dinuc_totals["category"] = (
                    train_dinuc_totals["category_code"].map(labels2)
                )
                train_dinuc_positives = train_counts.positives.rename(
                    columns={"kmer_2_code": "category_code"}
                ).copy()
                train_dinuc_positives["category"] = (
                    train_dinuc_positives["category_code"].map(labels2)
                )
                train_dinuc_summary = dinucleotide_category_summary(
                    train_dinuc_totals,
                    train_dinuc_positives,
                )
                baseline_fit = fit_dinucleotide_baseline(
                    train_dinuc_summary,
                    DinucleotideBaselineConfig(),
                )

                validation_dinuc_totals = validation_counts.totals.loc[
                    validation_counts.totals["k"] == 2
                ].copy()
                validation_dinuc_totals["category"] = (
                    validation_dinuc_totals["category_code"].map(labels2)
                )
                validation_dinuc_positives = validation_counts.positives.rename(
                    columns={"kmer_2_code": "category_code"}
                ).copy()
                validation_dinuc_positives["category"] = (
                    validation_dinuc_positives["category_code"].map(labels2)
                )
                baseline_evaluation = evaluate_dinucleotide_baseline(
                    validation_dinuc_totals,
                    validation_dinuc_positives,
                    baseline_fit,
                    DinucleotideBaselineConfig(),
                )
                lr_metrics = baseline_evaluation.metric_summary.set_index("metric")
                lookup2 = overall.loc[2]
                bridge_control = pd.DataFrame([
                    {
                        "metric": "Average Precision",
                        "EXP-001 logistic": lr_metrics.loc[
                            "Average Precision", "candidate"
                        ],
                        "2-mer lookup": lookup2["average_precision"],
                    },
                    {
                        "metric": "EPIC Spearman",
                        "EXP-001 logistic": lr_metrics.loc[
                            "EPIC Spearman", "candidate"
                        ],
                        "2-mer lookup": lookup2["epic_spearman"],
                    },
                ])
                bridge_control["absolute_difference"] = (
                    bridge_control["2-mer lookup"]
                    - bridge_control["EXP-001 logistic"]
                ).abs()
                display(bridge_control.style.format({
                    "EXP-001 logistic": "{:.12f}",
                    "2-mer lookup": "{:.12f}",
                    "absolute_difference": "{:.3g}",
                }))
                assert bridge_control["absolute_difference"].max() < 1e-12
            else:
                print("Reference и candidate уже используют один lookup estimator.")
            """,
        ),
        py(
            "coverage",
            """
            coverage_rows = []
            for k in CHAIN_KS:
                train_table = fit.category_summary.loc[
                    fit.category_summary["k"] == k
                ]
                validation_table = evaluation.category_summary.loc[
                    evaluation.category_summary["k"] == k
                ]
                unseen = (
                    (validation_table["train_total_positions"] == 0)
                    & (validation_table["total_positions"] > 0)
                )
                coverage_rows.append({
                    "k": k,
                    "possible_categories": len(train_table),
                    "train_observed_categories": int(train_table["observed"].sum()),
                    "train_zero_positive_categories": int(
                        ((train_table["total_positions"] > 0)
                         & (train_table["positive_positions"] == 0)).sum()
                    ),
                    "validation_observed_categories": int(
                        validation_table["observed"].sum()
                    ),
                    "validation_unseen_categories": int(unseen.sum()),
                    "validation_positions_in_unseen": int(
                        validation_table.loc[unseen, "total_positions"].sum()
                    ),
                    "unique_validation_score_levels": int(
                        overall.loc[k, "unique_score_levels"]
                    ),
                })
            coverage_summary = pd.DataFrame(coverage_rows)
            display(coverage_summary)
            """,
        ),
        md(
            "metric-plots-md",
            """
            ## 6. Основные метрики и стабильность по contig

            AP и Spearman имеют разные шкалы, поэтому изображаются отдельно.
            Per-contig линии соединяют один и тот же contig — это проверка, что
            pooled-прирост не создаётся одной удержанной последовательностью.
            """,
        ),
        py(
            "metric-bars",
            """
            def metric_bar(metric, source_column, filename, decimals):
                values = [
                    float(reference_row[source_column]),
                    float(candidate_row[source_column]),
                ]
                labels = [f"{REFERENCE_K}-mer", f"{CANDIDATE_K}-mer"]
                colors = [COLORS["reference"], COLORS["candidate"]]
                fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
                bars = ax.bar(labels, values, color=colors, width=0.58)
                margin = max(values) * 0.20 if max(values) else 0.1
                ax.set_ylim(0, max(values) + margin)
                ax.set(ylabel=metric, title=f"{EXPERIMENT_ID} — {metric}")
                ax.grid(axis="y", color="#E8EAED", linewidth=0.8)
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        value + margin*0.08,
                        f"{value:.{decimals}f}",
                        ha="center", va="bottom", fontweight="bold",
                    )
                show_plot(fig, filename)

            metric_bar(
                "Average Precision", "average_precision",
                "01_average_precision.png", 7,
            )
            metric_bar(
                "EPIC Spearman", "epic_spearman",
                "02_epic_spearman.png", 5,
            )
            """,
        ),
        py(
            "precision-recall",
            """
            fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
            for k, color, label in (
                (REFERENCE_K, COLORS["reference"], f"{REFERENCE_K}-mer"),
                (CANDIDATE_K, COLORS["candidate"], f"{CANDIDATE_K}-mer"),
            ):
                curve = evaluation.precision_recall.loc[
                    evaluation.precision_recall["k"] == k
                ].sort_values("recall")
                ap_value = float(overall.loc[k, "average_precision"])
                ax.step(
                    np.r_[0.0, curve["recall"].to_numpy()],
                    np.r_[curve["precision"].iloc[0], curve["precision"].to_numpy()],
                    where="post", linewidth=2.0, color=color,
                    label=f"{label} · AP={ap_value:.7f}",
                )
            prevalence = float(candidate_row["prevalence"])
            ax.axhline(
                prevalence, color="#5F6368", linestyle="--",
                label=f"prevalence={prevalence:.7f}",
            )
            ax.set(
                xlim=(0, 1), xlabel="Recall", ylabel="Precision",
                title="Полная validation Precision–Recall",
            )
            ax.grid(color="#E8EAED", linewidth=0.8)
            ax.legend(frameon=False)
            show_plot(fig, "03_precision_recall.png")
            """,
        ),
        py(
            "contig-plots",
            """
            per_contig = evaluation.per_contig_metrics.copy()

            def paired_contig_plot(metric, filename, display_name):
                local = per_contig.loc[
                    per_contig["k"].isin([REFERENCE_K, CANDIDATE_K])
                ]
                pivot = local.pivot(
                    index="segment_value", columns="k", values=metric
                )
                fig, ax = plt.subplots(figsize=(8.4, 4.7), layout="constrained")
                for index, (contig, row) in enumerate(pivot.iterrows()):
                    y = [row[REFERENCE_K], row[CANDIDATE_K]]
                    ax.plot([0, 1], y, color="#B0B7C3", linewidth=1.2)
                    ax.scatter(
                        [0, 1], y,
                        color=[COLORS["reference"], COLORS["candidate"]],
                        s=44,
                    )
                    offset = (index - (len(pivot)-1)/2) * max(np.ptp(y), 1e-8) * 0.08
                    ax.text(
                        1.03, y[1] + offset,
                        str(contig).replace("NC_", ""), va="center", fontsize=8,
                    )
                means = [pivot[REFERENCE_K].mean(), pivot[CANDIDATE_K].mean()]
                ax.plot([0, 1], means, color="#202124", linewidth=2.2)
                ax.scatter(
                    [0, 1], means, color="#202124", marker="D", s=65,
                    label="mean contig",
                )
                ax.set_xticks([0, 1], [f"{REFERENCE_K}-mer", f"{CANDIDATE_K}-mer"])
                ax.set_xlim(-0.15, 1.30)
                ax.set(ylabel=display_name, title=f"Per-contig — {display_name}")
                ax.grid(axis="y", color="#E8EAED", linewidth=0.8)
                ax.legend(frameon=False)
                show_plot(fig, filename)
                return pivot

            contig_ap = paired_contig_plot(
                "average_precision", "04_contig_average_precision.png",
                "Average Precision",
            )
            contig_spearman = paired_contig_plot(
                "epic_spearman", "05_contig_epic_spearman.png",
                "EPIC Spearman",
            )
            display(
                per_contig.loc[
                    per_contig["k"].isin([REFERENCE_K, CANDIDATE_K])
                ]
            )
            """,
        ),
        md(
            "support-md",
            """
            ## 7. Support и действие сглаживания

            У цельного k-mer нет переноса информации между всеми похожими
            словами. Backoff делится только с конечным более коротким suffix.
            `data_weight = n/(n+tau)` показывает, насколько итоговый score
            определяется собственной статистикой категории, а не prior.
            """,
        ),
        py(
            "support-plots",
            """
            candidate_train = fit.category_summary.loc[
                fit.category_summary["k"] == CANDIDATE_K
            ].copy()
            observed_candidate = candidate_train.loc[
                candidate_train["total_positions"] > 0
            ].copy()

            fig, ax = plt.subplots(figsize=(8.2, 4.5), layout="constrained")
            ax.hist(
                np.log10(observed_candidate["total_positions"]),
                bins=35, color="#5E97F6", edgecolor="white",
            )
            ax.axvline(
                np.log10(expected_tau), color="#D93025", linestyle="--",
                label=f"tau={expected_tau:.0f}",
            )
            ax.set(
                xlabel="log10(train support категории)",
                ylabel="Количество категорий",
                title=f"Support распределение {CANDIDATE_K}-mer",
            )
            ax.legend(frameon=False)
            show_plot(fig, "06_category_support.png")

            ordered_support = observed_candidate.sort_values("total_positions")
            fig, ax = plt.subplots(figsize=(8.2, 4.5), layout="constrained")
            ax.plot(
                ordered_support["total_positions"],
                ordered_support["data_weight"],
                color="#188038", linewidth=1.8,
            )
            ax.axvline(expected_tau, color="#D93025", linestyle="--")
            ax.set_xscale("log")
            ax.set(
                xlabel="Train support категории (log scale)",
                ylabel="Вес собственных данных n/(n+tau)",
                title="Редкие категории сильнее опираются на backoff",
                ylim=(0, 1.02),
            )
            ax.grid(color="#E8EAED", linewidth=0.8)
            show_plot(fig, "07_shrinkage_vs_support.png")
            """,
        ),
        py(
            "top-bottom",
            """
            minimum_support = max(int(np.ceil(expected_tau)), 1_000)
            stable_categories = candidate_train.loc[
                (candidate_train["total_positions"] >= minimum_support)
                & (candidate_train["category"] != "N/edge")
            ].copy()
            stable_categories["score_enrichment"] = (
                stable_categories["posterior_score"] / fit.global_prevalence
            )
            stable_categories["log2_score_enrichment"] = np.log2(
                stable_categories["score_enrichment"]
            )
            top_bottom = pd.concat([
                stable_categories.nsmallest(10, "posterior_score").assign(group="bottom"),
                stable_categories.nlargest(10, "posterior_score").assign(group="top"),
            ]).sort_values("log2_score_enrichment")

            fig, ax = plt.subplots(figsize=(9.0, 6.3), layout="constrained")
            colors = np.where(
                top_bottom["log2_score_enrichment"] >= 0,
                "#1967D2", "#D93025",
            )
            ax.barh(
                top_bottom["category"],
                top_bottom["log2_score_enrichment"],
                color=colors,
            )
            ax.axvline(0, color="#202124", linewidth=1)
            ax.set(
                xlabel="log2(score / train prevalence)",
                ylabel=f"{CANDIDATE_K}-mer",
                title=f"Top/bottom {CANDIDATE_K}-mer при support ≥ {minimum_support:,}",
            )
            ax.grid(axis="x", color="#E8EAED", linewidth=0.8)
            show_plot(fig, "08_top_bottom_kmers.png")
            display(top_bottom[[
                "category", "group", "total_positions", "positive_positions",
                "positive_rate", "prior_score", "posterior_score",
                "score_enrichment",
            ]])
            """,
        ),
        md(
            "external-md",
            """
            ## 8. История серии и внешний ориентир `epic_solution`

            External числа читаются из единой таблицы
            `phase2_nematostella_results.csv`. Они получены на official test
            halves с другим split и score transform. Поэтому они показаны
            отдельно, не участвуют в delta, критерии или решении и не позволяют
            утверждать, что одна локальная модель «победила» другую.
            """,
        ),
        py(
            "external-history",
            """
            external_path = (
                PROJECT_ROOT.parent
                / "epic_solution/results/phase2_nematostella_results.csv"
            )
            external_expected = pd.DataFrame({
                "k": [2, 4, 6, 8],
                "model": ["dinucleotide (challenge baseline)", "4-mer", "6-mer", "8-mer"],
                "external_average_precision": [0.00122, 0.00134, 0.00148, 0.00150],
                "external_epic_spearman": [0.0810, 0.1023, 0.1183, 0.1316],
            })
            if external_path.is_file():
                raw_external = pd.read_csv(external_path)
                selected = raw_external.loc[
                    raw_external["model"].isin(external_expected["model"])
                ].set_index("model")
                external_benchmark = external_expected[["k", "model"]].copy()
                external_benchmark["external_average_precision"] = (
                    external_benchmark["model"].map(selected["prauc_mean"])
                )
                external_benchmark["external_epic_spearman"] = (
                    external_benchmark["model"].map(selected["spear_mean"])
                )
                external_sha256 = hashlib.sha256(external_path.read_bytes()).hexdigest()
            else:
                external_benchmark = external_expected.copy()
                external_sha256 = None
            external_benchmark["protocol"] = (
                "external official test halves; different split and transform"
            )
            display(external_benchmark)

            local_history = evaluation.metric_summary.loc[
                evaluation.metric_summary["k"] > 0,
                ["k", "average_precision", "epic_spearman"],
            ].copy()
            local_history["protocol"] = "local contig_holdout_v1 full validation"

            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
            axes[0].plot(
                local_history["k"], local_history["average_precision"],
                marker="o", linewidth=2.2, color="#1967D2", label="local",
            )
            axes[0].plot(
                external_benchmark["k"],
                external_benchmark["external_average_precision"],
                marker="o", linestyle="--", color="#9AA0A6",
                label="epic_solution · другой split",
            )
            axes[0].set(
                xlabel="k", ylabel="Average Precision",
                title="AP по длине цельного слова",
            )
            axes[1].plot(
                local_history["k"], local_history["epic_spearman"],
                marker="o", linewidth=2.2, color="#1967D2", label="local",
            )
            axes[1].plot(
                external_benchmark["k"],
                external_benchmark["external_epic_spearman"],
                marker="o", linestyle="--", color="#9AA0A6",
                label="epic_solution · другой split",
            )
            axes[1].set(
                xlabel="k", ylabel="EPIC Spearman",
                title="Spearman по длине цельного слова",
            )
            for ax in axes:
                ax.set_xticks([2, 4, 6, 8])
                ax.grid(color="#E8EAED", linewidth=0.8)
                ax.legend(frameon=False)
            show_plot(fig, "09_kmer_history_and_external.png")
            """,
        ),
        md(
            "decision-md",
            """
            ## 9. Механическое решение

            Decision не выбирается по впечатлению от графика. Ниже буквально
            применяются три условия, записанные в начале notebook. Даже если
            более длинное слово имеет чуть лучший pooled AP, оно отклоняется,
            если прирост меньше заранее заданного порога или ухудшает вторичную
            обязательную метрику.
            """,
        ),
        py(
            "decision",
            """
            reference_ap = float(reference_row["average_precision"])
            candidate_ap = float(candidate_row["average_precision"])
            reference_spearman = float(reference_row["epic_spearman"])
            candidate_spearman = float(candidate_row["epic_spearman"])
            relative_ap_gain = candidate_ap / reference_ap - 1
            contig_ap_wins = int((
                contig_ap[CANDIDATE_K] > contig_ap[REFERENCE_K]
            ).sum())

            success_checks = pd.Series({
                "relative AP gain >= 1%": relative_ap_gain >= 0.01,
                "EPIC Spearman не ниже reference": (
                    candidate_spearman >= reference_spearman
                ),
                "AP выше минимум на 2/3 contig": contig_ap_wins >= 2,
            }, name="passed")
            display(success_checks.to_frame())

            success = bool(success_checks.all())
            DECISION = "adopt" if success else "reject"
            INTERPRETATION = (
                f"{CANDIDATE_K}-mer изменил полный validation AP с "
                f"{reference_ap:.9f} до {candidate_ap:.9f} "
                f"({relative_ap_gain:+.2%}) и EPIC Spearman с "
                f"{reference_spearman:.6f} до {candidate_spearman:.6f}. "
                f"AP улучшился на {contig_ap_wins} из 3 validation-contig. "
                f"По заранее заданному правилу решение: {DECISION}."
            )

            current_best = local_history.loc[
                local_history["average_precision"].idxmax()
            ]
            within_one_percent = local_history.loc[
                local_history["average_precision"]
                >= 0.99 * current_best["average_precision"]
            ]
            series_candidates = within_one_percent.loc[
                within_one_percent["epic_spearman"]
                >= current_best["epic_spearman"]
            ]
            recommended_k_so_far = (
                int(series_candidates["k"].min())
                if not series_candidates.empty else None
            )
            display(Markdown(
                f"**Decision: `{DECISION}`.** {INTERPRETATION}  \\n"
                f"Текущий shortest-near-best выбор в рассчитанной цепочке: "
                f"`k={recommended_k_so_far}`."
            ))
            """,
        ),
        md(
            "save-md",
            """
            ## 10. Сохранение запуска

            Сохраняются обе обязательные метрики, category lookup, coverage,
            support, positive predictions, PR points, runtime, внешний ориентир,
            все графики и metadata с полным data contract. `run_001` не
            перезаписывается: изменение протокола требует нового run name.
            """,
        ),
        py(
            "save",
            """
            train_candidate_summary = fit.category_summary.loc[
                fit.category_summary["k"] == CANDIDATE_K
            ].copy()
            validation_candidate_summary = evaluation.category_summary.loc[
                evaluation.category_summary["k"] == CANDIDATE_K
            ].copy()
            lookup_table = train_candidate_summary[[
                "k", "category_code", "category", "total_positions",
                "positive_positions", "positive_rate", "parent_k",
                "parent_category_code", "prior_score", "smoothing_strength",
                "data_weight", "posterior_score",
            ]]

            tables = {
                "metric_summary.csv": pair_metric_summary,
                "all_k_metrics.csv": evaluation.metric_summary,
                "per_contig_metrics.csv": evaluation.per_contig_metrics,
                "train_kmer_summary.csv": train_candidate_summary,
                "validation_kmer_summary.csv": validation_candidate_summary,
                "lookup_table.csv": lookup_table,
                "coverage_summary.csv": coverage_summary,
                "support_summary.csv": fit.support_summary,
                "smoothing_report.csv": fit.smoothing_report,
                "top_bottom_kmers.csv": top_bottom,
                "precision_recall.csv": evaluation.precision_recall,
                "runtime_report.csv": runtime_report,
                "kmer_history.csv": local_history,
                "external_benchmark.csv": external_benchmark,
            }
            if not bridge_control.empty:
                tables["exp001_bridge_control.csv"] = bridge_control
            for filename, table in tables.items():
                table.to_csv(ARTIFACT_DIR / filename, index=False)
            evaluation.positive_predictions.to_csv(
                ARTIFACT_DIR / "positive_predictions.csv.gz",
                index=False,
                compression="gzip",
            )

            METRICS = {
                "reference_average_precision": reference_ap,
                "candidate_average_precision": candidate_ap,
                "delta_average_precision": candidate_ap - reference_ap,
                "reference_epic_spearman": reference_spearman,
                "candidate_epic_spearman": candidate_spearman,
                "delta_epic_spearman": candidate_spearman - reference_spearman,
            }
            EXPERIMENT_PARAMETERS = CONFIG.to_dict() | {
                "reference_k": REFERENCE_K,
                "candidate_k": CANDIDATE_K,
                "backoff_chain": list(CHAIN_KS),
                "actual_tau": float(expected_tau),
                "tau_rule": "1 / full train prevalence; one expected positive",
                "n_edge_prior": "global train prevalence",
                "reference_experiment": "__REFERENCE_EXPERIMENT__",
                "external_benchmark_path": str(external_path),
                "external_benchmark_sha256": external_sha256,
                "external_benchmark_used_for_decision": False,
                "validation_population": "all whitelist positions",
                "score_quantisation": "none; raw float64 train-only posterior rates",
            }
            record = build_run_record(
                SPEC,
                split,
                run_name=RUN_NAME,
                notebook_path=NOTEBOOK_PATH,
                parameters=EXPERIMENT_PARAMETERS,
                metrics=METRICS,
                interpretation=INTERPRETATION,
                decision=DECISION,
            )
            output = save_run_record(PROJECT_ROOT, record)
            (output / "config.json").write_text(
                json.dumps(EXPERIMENT_PARAMETERS, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )
            print("Saved run:", output)
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    print(" -", path.relative_to(PROJECT_ROOT))
            """,
        ),
    ]

    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (EPIC EDA · Miniforge)",
                "language": "python",
                "name": "epic-eda",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "epic": {
                "experiment_id": exp_id,
                "data_contract": "ml_project.epic_data",
                "lookup_module": "ml_project.epic_kmer",
                "reference_k": reference_k,
                "candidate_k": candidate_k,
            },
        },
    )


def main() -> None:
    output_directory = ROOT / "notebooks/experiments"
    output_directory.mkdir(parents=True, exist_ok=True)
    for experiment in EXPERIMENTS:
        output = output_directory / str(experiment["filename"])
        nbf.write(build_notebook(experiment), output)
        print(output)


if __name__ == "__main__":
    main()
