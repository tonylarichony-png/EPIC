"""Build the readable, deliberately unexecuted EXP-005 notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/experiments/EXP-005_hierarchical_6mer_logistic.ipynb"


def markdown(cell_id: str, source: str):
    cell = nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")
    cell["id"] = cell_id
    return cell


def code(cell_id: str, source: str):
    cell = nbf.v4.new_code_cell(dedent(source).strip() + "\n")
    cell["id"] = cell_id
    return cell


def build_notebook():
    md = markdown
    py = code
    cells = [
        md(
            "title",
            """
            # EXP-005 — Hierarchical 2/4/6-mer Logistic Regression

            **Статус:** подготовлен, но ни одна code cell не выполнена.

            **Гипотеза.** L2 logistic regression сможет обучить вклад уровней
            `2-mer + 4-mer + 6-mer` и превзойти фиксированный hierarchical
            lookup из EXP-003 на новых contig.

            **Меняем только estimator:** фиксированная posterior-формула
            EXP-003 → обучаемая логистическая регрессия. Окно, target, split,
            ориентация и категории остаются прежними.

            Этот notebook специально разбит на маленькие шаги. Не используйте
            **Run All**: выполняйте ячейки сверху вниз и проверяйте вывод каждой.
            """,
        ),
        md(
            "roadmap",
            """
            ## Карта notebook

            1. Проверить окружение и неизменный data contract.
            2. Посчитать **только train** категории `2/4/6-mer`.
            3. Проверить sparse one-hot: 4 371 столбец, три активных признака.
            4. Построить train-contig folds и показать точный `C`-grid.
            5. Остановиться у первого checkpoint — до него модель не обучается.
            6. Вручную запустить train-only CV и выбрать `C` механическим правилом.
            7. Вручную обучить один final model на полном train.
            8. Только после фиксации `C` открыть validation и сравнить с EXP-003.
            9. Решение и артефакты сохранить отдельной последней ячейкой.

            > Validation намеренно расположен **после** выбора `C` и full-train
            > fit. Так параметры нельзя незаметно подобрать по финальным метрикам.
            """,
        ),
        py(
            "setup",
            r"""
            from pathlib import Path
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

            from ml_project.epic_baseline import load_baseline_sequences
            from ml_project.epic_data import ensure_prepared_split
            from ml_project.epic_experiment import (
                ExperimentSpec,
                build_run_record,
                save_run_record,
            )
            from ml_project.epic_kmer import (
                KmerLookupConfig,
                evaluate_kmer_lookup,
                fit_kmer_lookup,
                split_kmer_counts,
            )
            from ml_project.epic_kmer_logistic import (
                HierarchicalKmerLogisticConfig,
                aggregate_binary_likelihood,
                cross_validate_hierarchical_kmer_logistic,
                evaluate_category_scores,
                fit_hierarchical_kmer_logistic,
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
            pd.set_option("display.max_columns", 50)
            pd.set_option("display.max_rows", 60)

            display(pd.Series({
                "python": platform.python_version(),
                "executable": sys.executable,
                "project_root": str(PROJECT_ROOT),
                "notebook_state": "prepared / unexecuted",
            }, name="value").to_frame())
            """,
        ),
        md(
            "protocol-md",
            """
            ## 1. Протокол до обучения

            Главный reference — EXP-003, 6-mer lookup:

            - validation AP `0.0016005373755008387`;
            - EPIC Spearman `0.09718557212273755`.

            EXP-005 принимается только при выполнении всех условий:

            - относительный AP gain не меньше 1%;
            - EPIC Spearman не ниже EXP-003;
            - AP выше reference минимум на 2 из 3 validation-contig.

            `C` выбирается без validation: среди сошедшихся вариантов берём
            самый маленький `C` в пределах 1% от лучшего mean train-CV AP,
            если его mean CV Spearman не хуже Spearman лучшего по AP варианта.
            """,
        ),
        py(
            "protocol",
            """
            EXPERIMENT_ID = "EXP-005"
            RUN_NAME = "run_001"
            NOTEBOOK_PATH = (
                PROJECT_ROOT
                / "notebooks/experiments/EXP-005_hierarchical_6mer_logistic.ipynb"
            )
            ARTIFACT_DIR = (
                PROJECT_ROOT / "artifacts/experiments" / EXPERIMENT_ID / RUN_NAME
            )
            PLOT_DIR = ARTIFACT_DIR / "plots"

            SPEC = ExperimentSpec(
                experiment_id=EXPERIMENT_ID,
                title="Hierarchical 2/4/6-mer Logistic Regression",
                hypothesis=(
                    "Train-only tuned L2 logistic regression over hierarchical "
                    "2/4/6-mer one-hot features improves on EXP-003 lookup"
                ),
                changed_variable=(
                    "fixed posterior lookup -> L2 logistic regression; sequence "
                    "window and hierarchy remain 2/4/6-mer"
                ),
                success_criterion=(
                    "relative validation AP gain >= 1%; candidate EPIC Spearman "
                    ">= EXP-003; AP improves on at least 2 of 3 validation contigs"
                ),
                seed=42,
            )
            CONFIG = HierarchicalKmerLogisticConfig()
            display(pd.Series(CONFIG.to_dict(), name="fixed_value").to_frame())
            """,
        ),
        md(
            "data-contract-md",
            """
            ## 2. Неизменный data contract

            Загружается `contig_holdout_v1`. Обе цепи одного contig остаются
            вместе. Target равен `int(pooled_count > 0)`. Test target не
            существует и официальный test нигде в эксперименте не читается.
            """,
        ),
        py(
            "data-contract",
            """
            split = ensure_prepared_split(PROJECT_ROOT)
            assert split.config.split_version == "contig_holdout_v1"
            display(split.contig_assignment.sort_values(["split", "contig"]))
            display(split.summary)
            """,
        ),
        py(
            "load-fasta",
            """
            started = time.monotonic()
            sequences, fasta_alphabet = load_baseline_sequences(split)
            fasta_seconds = time.monotonic() - started
            display(pd.Series({
                "contigs_loaded": len(sequences),
                "alphabet": str(fasta_alphabet),
                "seconds": fasta_seconds,
            }, name="value").to_frame())
            """,
        ),
        md(
            "train-counts-md",
            """
            ## 3. Точный подсчёт train — ещё без модели

            Следующая ячейка проходит по полному train и считает категории.
            Это feature preparation, не обучение. Ожидаемое время на текущей
            машине — около 20 секунд. Validation пока не открывается.
            """,
        ),
        py(
            "train-counts",
            """
            started = time.monotonic()
            train_counts = split_kmer_counts(
                split,
                sequences,
                "train",
                ks=CONFIG.ks,
            )
            train_count_seconds = time.monotonic() - started
            print(f"Full train counted in {train_count_seconds:.2f} s")
            """,
        ),
        py(
            "train-count-audit",
            """
            train_population = (
                train_counts.totals.groupby("k")["total_positions"].sum()
            )
            train_positive_population = len(train_counts.positives)
            audit = pd.DataFrame({
                "positions": train_population,
                "positive_positions": train_positive_population,
                "prevalence": train_positive_population / train_population,
            })
            display(audit)
            assert train_population.nunique() == 1
            assert int(train_population.iloc[0]) == 228_064_160
            assert train_positive_population == 170_443
            """,
        ),
        md(
            "features-md",
            """
            ## 4. Признаки: обучаемая иерархия `2 + 4 + 6`

            Для каждого канонического 6-mer включаются три столбца:

            ```text
            ACGTCA → k2=CA + k4=GTCA + k6=ACGTCA
            ```

            Полный словарь содержит `17 + 257 + 4097 = 4371` столбец. Это
            полный sparse one-hot, а не порядковые числа. Благодаря общим
            2/4-mer столбцам редкие 6-mer могут опираться на устойчивый suffix,
            но силу этого эффекта теперь обучает LR.

            Консервативное правило неоднозначности: если 6-mer равен `N/edge`,
            активируется `N/edge` на каждом уровне. `strand` остаётся только в
            ключе; последовательность уже ориентирована extractor-ом.
            """,
        ),
        py(
            "build-design",
            """
            design = hierarchical_kmer_design(CONFIG)
            display(pd.Series({
                "possible_6mer_categories": design.matrix.shape[0],
                "feature_columns": design.matrix.shape[1],
                "nonzero_values": design.matrix.nnz,
                "active_features_per_category": design.matrix.getnnz(axis=1).min(),
            }, name="value").to_frame())
            assert design.matrix.shape == (4097, 4371)
            assert np.all(design.matrix.getnnz(axis=1) == 3)
            """,
        ),
        py(
            "inspect-design",
            """
            examples = design.category_table.loc[
                design.category_table["max_category"].isin(
                    ["AAAAAA", "ACGTCA", "TTTTTT", "N/edge"]
                )
            ]
            display(examples)
            display(design.feature_table.groupby("level_k").size().rename("columns"))
            """,
        ),
        md(
            "aggregate-md",
            """
            ## 5. Как 228 млн строк помещаются в память

            Все позиции одного 6-mer имеют одинаковые признаки. Поэтому binary
            likelihood можно **точно**, без sampling, представить двумя строками
            на категорию: negative и positive, а multiplicity передать через
            `sample_weight`.

            Максимум получается `4097 × 2 = 8194` агрегированные строки. Это
            не приближение: оптимизируется та же weighted logistic loss, что и
            при физическом повторении всех train-позиций.
            """,
        ),
        py(
            "aggregate-train",
            """
            aggregate_preview = aggregate_binary_likelihood(
                train_counts,
                design,
                balance_classes=CONFIG.balance_classes,
            )
            display(aggregate_preview.rows.head(10))
            display(pd.Series({
                "aggregate_rows": len(aggregate_preview.rows),
                "represented_positions": int(
                    aggregate_preview.rows["population_count"].sum()
                ),
                "represented_positives": int(
                    aggregate_preview.rows.loc[
                        aggregate_preview.rows["target"] == 1,
                        "population_count",
                    ].sum()
                ),
                "sparse_shape": str(aggregate_preview.matrix.shape),
                "sparse_nnz": aggregate_preview.matrix.nnz,
            }, name="value").to_frame())
            display(
                aggregate_preview.rows.groupby("target")[[
                    "population_count", "sample_weight"
                ]].sum()
            )
            """,
        ),
        md(
            "folds-md",
            """
            ## 6. Внутренняя validation только внутри train

            Девять train-contig делятся на три folds целиком, жадно балансируясь
            по числу position-strand. Обе цепи contig всегда остаются вместе.
            Внешние три validation-contig здесь не используются.
            """,
        ),
        py(
            "make-folds",
            """
            fold_assignment = make_balanced_contig_folds(
                train_counts,
                k=CONFIG.max_k,
                n_splits=CONFIG.cv_folds,
            )
            display(fold_assignment)
            display(
                fold_assignment.groupby("fold")[["positions", "positives"]]
                .sum()
                .assign(prevalence=lambda frame: frame.positives / frame.positions)
            )
            assert fold_assignment["contig"].nunique() == 9
            assert fold_assignment.groupby("contig")["fold"].nunique().max() == 1
            """,
        ),
        md(
            "solver-md",
            """
            ## 7. Solver и гиперпараметр

            Используется `LogisticRegression(penalty="l2", solver="lbfgs")`.
            У `lbfgs` нет пользовательского learning rate: длину шага определяет
            сам алгоритм. Единственный подбираемый model-параметр — `C`, обратная
            сила L2-регуляризации.

            Grid содержит 13 заранее заданных значений от `1e-4` до `100`.
            Это всего 39 fits (`13 C × 3 folds`) на матрице не более 8194 строк.
            Optuna здесь не нужна: одномерный grid прозрачен и проверяет весь
            диапазон.

            У `lbfgs` нет эпох, поэтому честного графика loss-by-epoch не будет.
            Вместо него после CV строятся balanced log-loss, AP, Spearman,
            `n_iter` и convergence по каждому `C`.
            """,
        ),
        py(
            "review-grid",
            """
            model_plan = pd.DataFrame({"C": CONFIG.C_grid})
            model_plan["regularization_strength_1_over_C"] = 1 / model_plan["C"]
            display(model_plan)
            print("Planned fits:", len(CONFIG.C_grid) * CONFIG.cv_folds)
            print("No model has been fitted yet.")
            """,
        ),
        md(
            "stop-before-training",
            """
            # ⛔ CHECKPOINT 1 — первое обучение ниже

            До этой точки выполнены только загрузка, точный подсчёт, построение
            признаков и folds. **Ни одной логистической регрессии ещё нет.**

            Перед следующей ячейкой вместе проверяем:

            - почему активны три признака;
            - почему агрегация точная;
            - зачем нужен class balancing;
            - почему validation ещё не открывался;
            - почему подбирается `C`, но не learning rate.
            """,
        ),
        md(
            "cv-training-md",
            """
            ## 8. TRAINING STEP 1 — train-only cross-validation

            Следующая ячейка действительно обучает 39 моделей. Запускайте её
            только после совместной проверки checkpoint. Она не читает внешний
            validation и не выбирает решение EXP-005.
            """,
        ),
        py(
            "run-train-cv",
            """
            started = time.monotonic()
            cv_results = cross_validate_hierarchical_kmer_logistic(
                train_counts,
                fold_assignment,
                config=CONFIG,
            )
            cv_seconds = time.monotonic() - started
            print(f"Train-only CV completed in {cv_seconds:.2f} s")
            display(cv_results)
            """,
        ),
        py(
            "summarise-cv",
            """
            cv_summary = summarise_cv(cv_results)
            display(cv_summary.style.format({
                "C": "{:.4g}",
                "mean_average_precision": "{:.9f}",
                "std_average_precision": "{:.9f}",
                "mean_epic_spearman": "{:.6f}",
                "std_epic_spearman": "{:.6f}",
                "mean_balanced_log_loss": "{:.6f}",
                "total_fit_seconds": "{:.2f}",
            }))
            assert cv_summary["all_folds_converged"].all(), (
                "Есть несошедшиеся C: сначала разбираем convergence, "
                "не продолжаем к validation."
            )
            """,
        ),
        py(
            "cv-plots",
            """
            fig_cv, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
            x = cv_summary["C"]
            panels = [
                ("mean_average_precision", "Train-CV Average Precision"),
                ("mean_epic_spearman", "Train-CV EPIC Spearman"),
                ("mean_balanced_log_loss", "Train-CV balanced log-loss"),
                ("max_iterations", "Максимум L-BFGS iterations"),
            ]
            for ax, (column, title) in zip(axes.flat, panels):
                ax.plot(x, cv_summary[column], marker="o", color="#1967D2")
                ax.set_xscale("log")
                ax.set_xlabel("C (log scale)")
                ax.set_ylabel(column)
                ax.set_title(title)
                ax.grid(color="#E8EAED")
            plt.show()
            """,
        ),
        py(
            "select-c",
            """
            selected_cv_row = select_regularization(
                cv_summary,
                relative_ap_tolerance=CONFIG.near_best_relative_ap,
            )
            SELECTED_C = float(selected_cv_row["C"])
            display(selected_cv_row.rename("selected_value").to_frame())
            print("Locked C:", SELECTED_C)
            """,
        ),
        md(
            "full-fit-md",
            """
            ## 9. TRAINING STEP 2 — один full-train fit

            `C` уже зафиксирован только по train-CV. Следующая ячейка обучает
            ровно одну модель на всех девяти train-contig. Validation по-прежнему
            не загружен и не посчитан.
            """,
        ),
        py(
            "fit-full-train",
            """
            final_fit = fit_hierarchical_kmer_logistic(
                train_counts,
                C=SELECTED_C,
                config=CONFIG,
                design=design,
            )
            display(final_fit.fit_report)
            assert bool(final_fit.fit_report.iloc[0]["converged"])
            """,
        ),
        py(
            "fit-audit",
            """
            coefficient_levels = (
                final_fit.coefficient_table.groupby("level_k")
                .agg(
                    features=("coefficient", "size"),
                    l1_norm=("absolute_coefficient", "sum"),
                    mean_absolute=("absolute_coefficient", "mean"),
                    max_absolute=("absolute_coefficient", "max"),
                )
            )
            display(coefficient_levels)
            display(
                final_fit.coefficient_table.nlargest(20, "absolute_coefficient")
            )
            """,
        ),
        md(
            "validation-gate",
            """
            # ⛔ CHECKPOINT 2 — `C` и final model уже зафиксированы

            Только теперь разрешено открыть validation. Если хочется изменить
            признаки, grid, solver, weights или правило выбора `C`, этот run
            прекращается и создаётся новый протокол — validation нельзя смотреть,
            а затем переписывать EXP-005.
            """,
        ),
        py(
            "count-validation",
            """
            started = time.monotonic()
            validation_counts = split_kmer_counts(
                split,
                sequences,
                "validation",
                ks=CONFIG.ks,
            )
            validation_count_seconds = time.monotonic() - started
            print(f"Full validation counted in {validation_count_seconds:.2f} s")
            """,
        ),
        py(
            "evaluate-overall",
            """
            lookup_config = KmerLookupConfig(
                ks=CONFIG.ks,
                smoothing="one_expected_positive",
            )
            lookup_fit = fit_kmer_lookup(train_counts, lookup_config)
            lookup_evaluation = evaluate_kmer_lookup(validation_counts, lookup_fit)
            reference_row = lookup_evaluation.metric_summary.loc[
                lookup_evaluation.metric_summary["k"] == 6
            ].iloc[0]
            assert np.isclose(
                reference_row["average_precision"], 0.0016005373755008387
            )
            assert np.isclose(
                reference_row["epic_spearman"], 0.09718557212273755
            )

            candidate_evaluation = evaluate_category_scores(
                validation_counts,
                final_fit.category_scores,
                k=CONFIG.max_k,
                variant="hierarchical_2_4_6mer_lr",
            )
            candidate_row = candidate_evaluation.metric_summary.iloc[0]
            metric_summary = pd.DataFrame([
                {
                    "variant": "EXP-003 6-mer lookup",
                    "average_precision": reference_row["average_precision"],
                    "epic_spearman": reference_row["epic_spearman"],
                },
                {
                    "variant": "EXP-005 hierarchical LR",
                    "average_precision": candidate_row["average_precision"],
                    "epic_spearman": candidate_row["epic_spearman"],
                },
            ])
            display(metric_summary)
            """,
        ),
        py(
            "evaluate-contigs",
            """
            reference_contigs = lookup_evaluation.per_contig_metrics.loc[
                lookup_evaluation.per_contig_metrics["k"] == 6,
                ["segment_value", "average_precision", "epic_spearman"],
            ].rename(columns={
                "segment_value": "contig",
                "average_precision": "reference_average_precision",
                "epic_spearman": "reference_epic_spearman",
            })
            candidate_contig_rows = []
            for contig in sorted(reference_contigs["contig"]):
                evaluated = evaluate_category_scores(
                    validation_counts,
                    final_fit.category_scores,
                    k=CONFIG.max_k,
                    contigs=(contig,),
                    variant="hierarchical_2_4_6mer_lr",
                )
                row = evaluated.metric_summary.iloc[0]
                candidate_contig_rows.append({
                    "contig": contig,
                    "candidate_average_precision": row["average_precision"],
                    "candidate_epic_spearman": row["epic_spearman"],
                })
            per_contig_metrics = reference_contigs.merge(
                pd.DataFrame(candidate_contig_rows), on="contig", validate="one_to_one"
            )
            display(per_contig_metrics)
            """,
        ),
        py(
            "metric-plots",
            """
            colors = ["#9AA0A6", "#1967D2"]
            fig_metrics, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
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
            """,
        ),
        py(
            "diagnostic-plots",
            """
            fig_contigs, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
            for ax, metric, label in [
                (axes[0], "average_precision", "Average Precision"),
                (axes[1], "epic_spearman", "EPIC Spearman"),
            ]:
                ref = per_contig_metrics[f"reference_{metric}"]
                cand = per_contig_metrics[f"candidate_{metric}"]
                for index, contig in enumerate(per_contig_metrics["contig"]):
                    ax.plot([0, 1], [ref.iloc[index], cand.iloc[index]], color="#AAB2C0")
                    ax.scatter([0], [ref.iloc[index]], color="#9AA0A6")
                    ax.scatter([1], [cand.iloc[index]], color="#1967D2")
                    ax.text(1.03, cand.iloc[index], contig.replace("NC_", ""), va="center")
                ax.set_xticks([0, 1], ["EXP-003 lookup", "EXP-005 LR"])
                ax.set_ylabel(label)
                ax.set_title(f"Per-contig — {label}")
                ax.grid(axis="y", color="#E8EAED")
            plt.show()

            reference_pr = lookup_evaluation.precision_recall.loc[
                lookup_evaluation.precision_recall["k"] == 6
            ]
            candidate_pr = candidate_evaluation.precision_recall
            fig_pr, ax = plt.subplots(figsize=(9, 5))
            ax.step(reference_pr["recall"], reference_pr["precision"], where="post",
                    color="#9AA0A6", label="EXP-003 lookup")
            ax.step(candidate_pr["recall"], candidate_pr["precision"], where="post",
                    color="#1967D2", label="EXP-005 LR")
            ax.axhline(candidate_row["prevalence"], color="#5F6368", linestyle="--",
                       label=f"prevalence={candidate_row['prevalence']:.7f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Полная validation Precision–Recall")
            ax.legend(frameon=False)
            ax.grid(color="#E8EAED")
            plt.show()
            """,
        ),
        py(
            "coefficient-plots",
            """
            coefficient_summary = (
                final_fit.coefficient_table.groupby("level_k")
                .agg(
                    mean_absolute=("absolute_coefficient", "mean"),
                    l1_norm=("absolute_coefficient", "sum"),
                )
                .reset_index()
            )
            strongest = final_fit.coefficient_table.nlargest(
                20, "absolute_coefficient"
            ).sort_values("coefficient")
            fig_coefficients, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            axes[0].bar(
                coefficient_summary["level_k"].astype(str),
                coefficient_summary["mean_absolute"],
                color="#1967D2",
            )
            axes[0].set_xlabel("k-mer level")
            axes[0].set_ylabel("Mean |coefficient|")
            axes[0].set_title("Средний вклад уровня")
            axes[0].grid(axis="y", color="#E8EAED")
            axes[1].barh(
                strongest["feature_name"],
                strongest["coefficient"],
                color=np.where(strongest["coefficient"] >= 0, "#1967D2", "#D93025"),
            )
            axes[1].axvline(0, color="#202124", linewidth=1)
            axes[1].set_xlabel("Log-odds coefficient")
            axes[1].set_title("20 самых сильных коэффициентов")
            plt.show()
            """,
        ),
        md(
            "decision-md",
            """
            ## 10. Механическое решение

            Эта ячейка не меняет критерий после просмотра результата. Все три
            условия должны выполниться одновременно. Даже хороший train-CV не
            заменяет внешний contig holdout.
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
                per_contig_metrics["candidate_average_precision"]
                > per_contig_metrics["reference_average_precision"]
            ).sum())
            success_checks = pd.Series({
                "relative AP gain >= 1%": relative_ap_gain >= 0.01,
                "EPIC Spearman >= EXP-003": candidate_spearman >= reference_spearman,
                "AP wins on at least 2/3 contigs": contig_ap_wins >= 2,
            }, name="passed")
            display(success_checks.to_frame())
            DECISION = "adopt" if bool(success_checks.all()) else "reject"
            INTERPRETATION = (
                f"Hierarchical LR changed AP from {reference_ap:.9f} to "
                f"{candidate_ap:.9f} ({relative_ap_gain:+.2%}) and EPIC Spearman "
                f"from {reference_spearman:.6f} to {candidate_spearman:.6f}; "
                f"AP wins on {contig_ap_wins}/3 contigs. Decision: {DECISION}."
            )
            display(Markdown(f"**Decision: `{DECISION}`.** {INTERPRETATION}"))
            """,
        ),
        md(
            "save-md",
            """
            ## 11. Сохранение — выполнять последней

            Ячейка создаёт `artifacts/experiments/EXP-005/run_001`. Она не
            перезаписывает существующий run. До проверки decision её запускать
            нельзя.
            """,
        ),
        py(
            "save",
            r"""
            PLOT_DIR.mkdir(parents=True, exist_ok=True)
            plot_objects = {
                "01_train_cv_diagnostics.png": fig_cv,
                "02_validation_metrics.png": fig_metrics,
                "03_per_contig_metrics.png": fig_contigs,
                "04_precision_recall.png": fig_pr,
                "05_coefficients.png": fig_coefficients,
            }
            for filename, figure in plot_objects.items():
                figure.savefig(
                    PLOT_DIR / filename,
                    dpi=170,
                    bbox_inches="tight",
                    facecolor="white",
                )
            tables = {
                "cv_fold_assignment.csv": fold_assignment,
                "cv_results.csv": cv_results,
                "cv_summary.csv": cv_summary,
                "metric_summary.csv": metric_summary,
                "per_contig_metrics.csv": per_contig_metrics,
                "fit_report.csv": final_fit.fit_report,
                "feature_table.csv": final_fit.design.feature_table,
                "coefficient_table.csv": final_fit.coefficient_table,
                "precision_recall.csv": candidate_evaluation.precision_recall,
            }
            for filename, table in tables.items():
                table.to_csv(ARTIFACT_DIR / filename, index=False)
            candidate_evaluation.positive_predictions.to_csv(
                ARTIFACT_DIR / "positive_predictions.csv.gz",
                index=False,
                compression="gzip",
            )

            METRICS = {
                "reference_average_precision": reference_ap,
                "candidate_average_precision": candidate_ap,
                "delta_average_precision": candidate_ap - reference_ap,
                "relative_average_precision_gain": relative_ap_gain,
                "reference_epic_spearman": reference_spearman,
                "candidate_epic_spearman": candidate_spearman,
                "delta_epic_spearman": candidate_spearman - reference_spearman,
            }
            PARAMETERS = CONFIG.to_dict() | {
                "selected_C": SELECTED_C,
                "reference_experiment": "EXP-003/run_001",
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
                json.dumps(PARAMETERS, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print("Saved run:", output)
            """,
        ),
        md(
            "sync-report-md",
            """
            ## 12. Обновить карточку эксперимента

            Эта финальная ячейка читает уже сохранённые CSV/JSON и графики,
            записывает метрики, CV, коэффициенты и решение в `experiments/EXP-005.md`.
            Модель повторно не обучается, а ручной анализ в карточке сохраняется.
            """,
        ),
        py(
            "sync-report",
            """
            from ml_project.epic_experiment_report import sync_exp005_report

            experiment_note = sync_exp005_report(PROJECT_ROOT, run_name=RUN_NAME)
            print("Updated experiment note:", experiment_note)
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
            "language_info": {"name": "python", "version": "3.12"},
            "epic": {
                "experiment_id": "EXP-005",
                "status": "prepared_unexecuted",
                "reference_experiment": "EXP-003",
                "feature_module": "ml_project.epic_kmer_logistic",
                "candidate_k": 6,
                "training_cells": ["run-train-cv", "fit-full-train"],
            },
        },
    )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
