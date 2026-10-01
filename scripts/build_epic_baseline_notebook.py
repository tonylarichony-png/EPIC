"""Build the readable and executable EXP-001 dinucleotide baseline notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/experiments/EXP-001_dinucleotide_baseline.ipynb"


def markdown(cell_id: str, source: str):
    cell = nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")
    cell["id"] = cell_id
    return cell


def code(cell_id: str, source: str):
    cell = nbf.v4.new_code_cell(dedent(source).strip() + "\n")
    cell["id"] = cell_id
    return cell


CELLS = [
    markdown(
        "title",
        r"""
        # EXP-001 — динуклеотидный baseline

        Первая контролируемая модель EPIC использует **ровно один категориальный
        признак**: две буквы ориентированной ДНК в offsets `[-1, 0]` — букву
        непосредственно перед предсказываемой позицией и букву в самой позиции.

        **Гипотеза.** Локальная ориентированная пара содержит воспроизводимый
        сигнал и превзойдёт константный no-skill reference на фиксированных
        validation-contig.

        **Меняем только одно.** Константный score заменяется на L2 logistic
        regression по 16 парам `AA…TT` и общей категории `N/edge`.

        **Критерий успеха до запуска.** На полном validation Average Precision
        выше prevalence reference, EPIC Spearman положителен, solver сошёлся без
        `ConvergenceWarning`. Официальный test не используется.
        """,
    ),
    markdown(
        "protocol",
        r"""
        ## 1. Что фиксируем до расчёта

        - Split: `contig_holdout_v1`, целые contig, seed 42.
        - Target: `csRNA-r1 + csRNA-r2 > 0`; исходный count остаётся только для
          вторичной ранговой метрики.
        - Признак: ориентированная по цепи пара `[-1, 0]`.
        - Кодирование: reference one-hot; `TT` — нулевая опорная категория,
          остальные 15 канонических пар и `N/edge` дают 16 столбцов.
        - Candidate: `LogisticRegression`, L2, `C=1`, solver `lbfgs`.
        - Reference: одинаковый score для всех строк.
        - Метрики: полный validation AP и EPIC Spearman по положительным target.

        Здесь нет Optuna, подбора threshold или выбора по validation. Этот запуск
        должен дать честную первую точку отсчёта для следующих экспериментов.
        """,
    ),
    code(
        "setup",
        r"""
        from pathlib import Path
        import json
        import platform
        import sys

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
            aggregate_training_data,
            category_summary,
            coefficient_table,
            evaluate_dinucleotide_baseline,
            fit_dinucleotide_baseline,
            load_baseline_sequences,
            split_category_counts,
        )
        from ml_project.epic_data import ensure_prepared_split
        from ml_project.epic_experiment import (
            ExperimentSpec,
            build_run_record,
            save_run_record,
        )
        from ml_project.sequence_features import DINUCLEOTIDE_LABELS

        SPEC = ExperimentSpec(
            experiment_id="EXP-001",
            title="Динуклеотидный baseline",
            hypothesis=(
                "Ориентированная пара DNA [-1,0] содержит воспроизводимый сигнал "
                "и превосходит константный reference на validation-contig."
            ),
            changed_variable=(
                "Константный score заменён на L2 logistic regression только по "
                "17 категориям ориентированного динуклеотида."
            ),
            success_criterion=(
                "validation AP > prevalence, EPIC Spearman > 0, solver сошёлся "
                "без ConvergenceWarning"
            ),
            seed=42,
        )
        CONFIG = DinucleotideBaselineConfig(seed=SPEC.seed)
        RUN_NAME = "run_001"
        NOTEBOOK_PATH = (
            PROJECT_ROOT
            / "notebooks/experiments/EXP-001_dinucleotide_baseline.ipynb"
        )
        ARTIFACT_DIR = PROJECT_ROOT / "artifacts/experiments/EXP-001" / RUN_NAME
        PLOT_DIR = ARTIFACT_DIR / "plots"
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update({
            "figure.dpi": 120,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        })
        pd.set_option("display.max_columns", 30)
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
            "experiment": SPEC.experiment_id,
            "seed": SPEC.seed,
            "run": RUN_NAME,
        }, name="value").to_frame())
        """,
    ),
    markdown(
        "data-contract-md",
        r"""
        ## 2. Единый data contract

        Следующая ячейка не создаёт новое разбиение. Она загружает сохранённый
        manifest, проверяет SHA-256 исходников и производных split-файлов и
        показывает фактический состав train / validation / test.

        В дальнейшем читаются только train-target и validation-target. У test
        нет меток, и он не участвует ни в обучении, ни в выборе решения.
        """,
    ),
    code(
        "data-contract",
        r"""
        split = ensure_prepared_split(PROJECT_ROOT)

        display(split.contig_assignment.sort_values(["split", "contig"]))
        display(split.summary)
        print("Split version:", split.config.split_version)
        print("Split seed:", split.config.seed)
        """,
    ),
    markdown(
        "aggregation-md",
        r"""
        ## 3. Почему 228 млн строк можно точно свернуть в 34

        У модели всего **17 возможных входов**:

        - 16 пар из `A/C/G/T`: `AA, AC, …, TT`;
        - одна общая категория `N/edge`, если хотя бы одна буква неизвестна либо
          контекст выходит за край contig.

        Все строки одной категории имеют абсолютно одинаковый вектор признаков.
        Поэтому для каждой категории достаточно двух строк: `target=0` и
        `target=1`, а их `sample_weight` равен числу представленных позиций с
        поправкой `class_weight="balanced"`.

        Получается `17 × 2 = 34` строки для solver. Это **не subsampling**:
        последовательность всё равно сканируется целиком один раз, а взвешенная
        log-loss и её градиент точно совпадают с развёрнутым датасетом. Такое
        сжатие возможно именно потому, что в EXP-001 только один признак с низкой
        кардинальностью.
        """,
    ),
    code(
        "sequence-and-counts",
        r"""
        sequences, fasta_alphabet = load_baseline_sequences(split)

        train_totals, train_positives = split_category_counts(
            split, sequences, "train"
        )
        validation_totals, validation_positives = split_category_counts(
            split, sequences, "validation"
        )

        count_audit = pd.DataFrame([
            {
                "split": "train",
                "all_position_strand": int(train_totals["total_positions"].sum()),
                "positive_position_strand": len(train_positives),
                "contigs": train_totals["contig"].nunique(),
                "categories": train_totals.loc[
                    train_totals["total_positions"] > 0, "category"
                ].nunique(),
            },
            {
                "split": "validation",
                "all_position_strand": int(validation_totals["total_positions"].sum()),
                "positive_position_strand": len(validation_positives),
                "contigs": validation_totals["contig"].nunique(),
                "categories": validation_totals.loc[
                    validation_totals["total_positions"] > 0, "category"
                ].nunique(),
            },
        ])
        display(count_audit)

        expected = split.summary.set_index("split")
        for row in count_audit.itertuples(index=False):
            assert row.all_position_strand == int(
                expected.loc[row.split, "position_strand_count"]
            )
            assert row.positive_position_strand == int(expected.loc[row.split, "positive_positions"])
        assert tuple(DINUCLEOTIDE_LABELS)[-1] == "N/edge"
        print("Audit passed: ни одна train/validation позиция не потеряна.")
        """,
    ),
    markdown(
        "train-summary-md",
        r"""
        ## 4. Что модель видит в train

        `positive_rate` — доля положительных среди всех позиций данной пары.
        `enrichment` делит эту долю на общую train-prevalence. Например,
        enrichment `2.9×` означает: положительный target у этой пары встречается
        примерно в 2.9 раза чаще, чем у случайной train-позиции. Это ассоциация,
        а не причинный эффект.
        """,
    ),
    code(
        "train-summary",
        r"""
        train_category_summary = category_summary(train_totals, train_positives)
        train_prevalence = (
            train_category_summary["positive_positions"].sum()
            / train_category_summary["total_positions"].sum()
        )
        display(
            train_category_summary.sort_values("enrichment", ascending=False)
            .style.format({
                "positive_rate": "{:.7f}",
                "enrichment": "{:.3f}×",
            })
        )
        print(f"Train prevalence: {train_prevalence:.9f}")
        """,
    ),
    code(
        "aggregate-training",
        r"""
        aggregate_rows, encoded_aggregate = aggregate_training_data(
            train_category_summary, CONFIG
        )

        aggregate_audit = pd.Series({
            "solver_rows": len(aggregate_rows),
            "feature_columns": encoded_aggregate.matrix.shape[1],
            "represented_positions": int(aggregate_rows["population_count"].sum()),
            "represented_positives": int(
                aggregate_rows.loc[
                    aggregate_rows["target"] == 1, "population_count"
                ].sum()
            ),
            "negative_weight_sum": aggregate_rows.loc[
                aggregate_rows["target"] == 0, "sample_weight"
            ].sum(),
            "positive_weight_sum": aggregate_rows.loc[
                aggregate_rows["target"] == 1, "sample_weight"
            ].sum(),
        }, name="value")
        display(aggregate_audit.to_frame())
        display(aggregate_rows.head(8))

        assert len(aggregate_rows) == 34
        assert int(aggregate_rows["population_count"].sum()) == int(
            train_totals["total_positions"].sum()
        )
        """,
    ),
    markdown(
        "fit-md",
        r"""
        ## 5. Обучение logistic regression

        У `lbfgs` нет настраиваемого `learning_rate`: solver сам выбирает длину
        шага во внутреннем line search. Настраиваемые здесь величины — сила L2
        (`C`), точность остановки (`tol`) и лимит итераций (`max_iter`). В
        baseline они заранее зафиксированы, а не подобраны по validation.

        Важное ограничение визуализации: scikit-learn не возвращает историю
        objective для каждой внутренней итерации `lbfgs`, поэтому честной кривой
        **loss vs epoch** здесь нет. Ниже показаны доступные диагностики
        сходимости; рисовать выдуманную loss-кривую нельзя.
        """,
    ),
    code(
        "fit",
        r"""
        fit = fit_dinucleotide_baseline(train_category_summary, CONFIG)
        display(fit.fit_report.T.rename(columns={0: "value"}))

        fit_row = fit.fit_report.iloc[0]
        assert bool(fit_row["converged"])
        assert fit_row["warnings"] == ""
        """,
    ),
    code(
        "convergence-plot",
        r"""
        used = int(fit_row["iterations"])
        budget = int(fit_row["max_iter"])

        fig, ax = plt.subplots(figsize=(8.5, 2.3), layout="constrained")
        ax.barh(["LBFGS"], [used], color="#188038", label="использовано")
        ax.barh(
            ["LBFGS"], [budget - used], left=[used], color="#E8EAED",
            label="не понадобилось"
        )
        ax.axvline(budget, color="#5F6368", linestyle="--", linewidth=1)
        ax.text(used + budget * 0.01, 0, f"{used} из {budget} итераций", va="center")
        ax.set(xlabel="Лимит итераций", title="Solver завершился задолго до max_iter")
        ax.legend(frameon=False, ncol=2, loc="lower right")
        show_plot(fig, "01_convergence.png")
        """,
    ),
    markdown(
        "evaluation-md",
        r"""
        ## 6. Полная validation-оценка

        AP вычисляется точно по всем validation-позициям, а не по выборке.
        Поскольку score постоянен внутри категории, одинаковые scores
        обрабатываются одной threshold-группой.

        EPIC Spearman считается только среди положительных позиций: dense rank
        pooled count коррелируется с dense rank score. У константного reference
        ранжирования нет; в таблице для сравнения записан `0` по явной конвенции
        официального fast-score, а не как математически определённая корреляция.
        """,
    ),
    code(
        "evaluation",
        r"""
        evaluation = evaluate_dinucleotide_baseline(
            validation_totals,
            validation_positives,
            fit,
            CONFIG,
        )
        display(
            evaluation.metric_summary.style.format({
                "reference": "{:.9f}",
                "candidate": "{:.9f}",
                "delta": "{:+.9f}",
            })
        )

        metric_by_name = evaluation.metric_summary.set_index("metric")
        reference_ap = float(metric_by_name.loc["Average Precision", "reference"])
        candidate_ap = float(metric_by_name.loc["Average Precision", "candidate"])
        candidate_spearman = float(metric_by_name.loc["EPIC Spearman", "candidate"])
        ap_ratio = candidate_ap / reference_ap

        display(Markdown(
            f"**Итог:** AP вырос с `{reference_ap:.9f}` до `{candidate_ap:.9f}` "
            f"(`{ap_ratio:.2f}×`, delta `{candidate_ap-reference_ap:+.9f}`); "
            f"EPIC Spearman = `{candidate_spearman:.6f}`."
        ))
        """,
    ),
    code(
        "overall-metric-plots",
        r"""
        def metric_bar(metric, filename, decimals):
            row = metric_by_name.loc[metric]
            values = [float(row["reference"]), float(row["candidate"])]
            labels = ["Constant reference", "Dinucleotide LR"]
            colors = [COLORS["reference"], COLORS["candidate"]]
            fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
            bars = ax.bar(labels, values, color=colors, width=0.58)
            margin = max(values) * 0.20 if max(values) else 0.1
            ax.set_ylim(0, max(values) + margin)
            ax.set_ylabel(metric)
            ax.set_title(f"Validation — {metric}")
            ax.grid(axis="y", color="#E8EAED", linewidth=0.8)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + margin * 0.08,
                    f"{value:.{decimals}f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
            show_plot(fig, filename)

        metric_bar("Average Precision", "02_average_precision.png", 7)
        metric_bar("EPIC Spearman", "03_epic_spearman.png", 4)
        """,
    ),
    markdown(
        "pr-md",
        r"""
        ### Precision–Recall, а не accuracy

        Положительный класс крайне редкий, поэтому accuracy вводила бы в
        заблуждение. Горизонтальная линия — prevalence: ожидаемый AP
        константного/случайного ранжирования. Ступеней немного, потому что модель
        имеет только 17 возможных score-уровней.
        """,
    ),
    code(
        "pr-plot",
        r"""
        pr = evaluation.precision_recall.sort_values("recall")
        fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
        ax.step(
            np.r_[0.0, pr["recall"].to_numpy()],
            np.r_[pr["precision"].iloc[0], pr["precision"].to_numpy()],
            where="post",
            color=COLORS["candidate"],
            linewidth=2.2,
            label=f"Dinucleotide LR · AP={candidate_ap:.7f}",
        )
        ax.axhline(
            reference_ap,
            color=COLORS["reference"],
            linestyle="--",
            linewidth=1.7,
            label=f"Prevalence · {reference_ap:.7f}",
        )
        ax.set(
            xlim=(0, 1),
            ylim=(0, max(pr["precision"].max(), reference_ap) * 1.12),
            xlabel="Recall",
            ylabel="Precision",
            title="Полная validation Precision–Recall кривая",
        )
        ax.grid(color="#E8EAED", linewidth=0.8)
        ax.legend(frameon=False)
        show_plot(fig, "04_precision_recall.png")
        """,
    ),
    markdown(
        "contigs-md",
        r"""
        ### Устойчивость по validation-contig

        Общая метрика может скрыть, что выигрыш создаёт один contig. Поэтому
        каждая тонкая линия ниже соединяет reference и candidate для одного и
        того же contig. Ромб показывает среднее по трём contig и не заменяет
        полную pooled-метрику выше.
        """,
    ),
    code(
        "contig-plots",
        r"""
        per_contig = evaluation.per_contig_metrics.copy()

        def paired_contig_plot(metric, filename, decimals):
            pivot = per_contig.pivot(
                index="segment_value", columns="variant", values=metric
            )
            x = np.array([0.0, 1.0])
            fig, ax = plt.subplots(figsize=(8.2, 4.7), layout="constrained")
            for contig, row in pivot.iterrows():
                y = [row["constant_reference"], row["dinucleotide_logistic"]]
                ax.plot(x, y, color="#B0B7C3", linewidth=1.2, alpha=0.8)
                ax.scatter(x, y, color=[COLORS["reference"], COLORS["candidate"]], s=42)
                ax.text(1.03, y[1], str(contig).replace("NC_", ""), va="center", fontsize=8)
            means = [
                pivot["constant_reference"].mean(),
                pivot["dinucleotide_logistic"].mean(),
            ]
            ax.plot(x, means, color="#202124", linewidth=2.3, zorder=3)
            ax.scatter(x, means, color="#202124", marker="D", s=70, label="mean contig")
            ax.set_xticks(x, ["Constant reference", "Dinucleotide LR"])
            ax.set_xlim(-0.15, 1.28)
            ax.set_ylabel(metric)
            ax.set_title(f"Validation-contig — {metric}")
            ax.grid(axis="y", color="#E8EAED", linewidth=0.8)
            ax.legend(frameon=False)
            show_plot(fig, filename)
            return pivot

        contig_ap = paired_contig_plot(
            "average_precision", "05_contig_average_precision.png", 7
        )
        contig_spearman = paired_contig_plot(
            "epic_spearman", "06_contig_epic_spearman.png", 4
        )
        display(per_contig)
        """,
    ),
    markdown(
        "coefficients-md",
        r"""
        ## 7. Вклад категорий

        Коэффициент — изменение log-odds относительно опорной пары `TT` при
        прочих равных. Здесь других признаков нет. Положительное значение
        повышает score, отрицательное понижает; `exp(coefficient)` показан как
        odds ratio относительно `TT`.

        Это удобная интерпретация baseline, но не причинный вывод. Кроме того,
        class balancing сдвигает intercept, поэтому сами scores не следует
        читать как откалиброванные вероятности.
        """,
    ),
    code(
        "coefficient-plot",
        r"""
        coefficients = coefficient_table(fit, CONFIG).sort_values("coefficient")

        fig, ax = plt.subplots(figsize=(9.2, 7.0), layout="constrained")
        colors = np.where(
            coefficients["coefficient"] >= 0, "#1967D2", "#D93025"
        )
        ax.barh(coefficients["category"], coefficients["coefficient"], color=colors)
        ax.axvline(0, color="#202124", linewidth=1)
        ax.set(
            xlabel="Коэффициент log-odds относительно TT",
            ylabel="Ориентированная пара [-1, 0]",
            title="Какие динуклеотиды повышают и понижают score",
        )
        ax.grid(axis="x", color="#E8EAED", linewidth=0.8)
        show_plot(fig, "07_dinucleotide_coefficients.png")

        category_effects = (
            coefficients.merge(
                train_category_summary[
                    ["category", "positive_rate", "enrichment"]
                ],
                on="category",
                how="left",
            )
            .sort_values("coefficient", ascending=False)
            .reset_index(drop=True)
        )
        display(
            category_effects.style.format({
                "coefficient": "{:+.4f}",
                "odds_ratio": "{:.3f}×",
                "positive_rate": "{:.7f}",
                "enrichment": "{:.3f}×",
            })
        )
        """,
    ),
    markdown(
        "decision-md",
        r"""
        ## 8. Решение

        Критерий применяется механически к тем значениям, которые были
        зафиксированы до запуска. Baseline принимается как первая рабочая точка,
        если обе метрики лучше reference и solver действительно сошёлся.

        Это не означает, что `C=1` оптимально или что двух букв достаточно для
        финального решения. Следующий отдельный эксперимент сможет менять ровно
        один компонент — например, регуляризацию или ширину контекста — и
        сравниваться с этим сохранённым baseline.
        """,
    ),
    code(
        "decision",
        r"""
        success_checks = pd.Series({
            "AP выше prevalence reference": candidate_ap > reference_ap,
            "EPIC Spearman положителен": candidate_spearman > 0,
            "LBFGS сошёлся": bool(fit_row["converged"]),
            "Нет ConvergenceWarning": fit_row["warnings"] == "",
        }, name="passed")
        display(success_checks.to_frame())

        success = bool(success_checks.all())
        DECISION = "adopt" if success else "iterate"
        INTERPRETATION = (
            f"На полном validation динуклеотидный baseline увеличил AP с "
            f"{reference_ap:.9f} до {candidate_ap:.9f} ({ap_ratio:.2f}x) и дал "
            f"EPIC Spearman {candidate_spearman:.6f}. LBFGS сошёлся за "
            f"{int(fit_row['iterations'])} итераций без предупреждений. "
            f"Решение: {DECISION}; использовать эту модель как reference для "
            f"следующего контролируемого эксперимента."
        )
        display(Markdown(f"**Decision: `{DECISION}`.** {INTERPRETATION}"))
        """,
    ),
    markdown(
        "save-md",
        r"""
        ## 9. Сохранение воспроизводимого запуска

        Сохраняются таблицы, predictions только для положительных validation
        позиций, PR points, коэффициенты, графики и `metadata.json` с параметрами,
        метриками, SHA-256 data contract и notebook.

        `run_001` намеренно не перезаписывается: после изменения параметров нужен
        новый `RUN_NAME`, иначе старый и новый результат смешались бы.
        """,
    ),
    code(
        "save",
        r"""
        tables = {
            "train_category_summary.csv": train_category_summary,
            "aggregate_training_rows.csv": fit.aggregate_training_rows,
            "fit_report.csv": fit.fit_report,
            "metric_summary.csv": evaluation.metric_summary,
            "per_contig_metrics.csv": evaluation.per_contig_metrics,
            "validation_category_summary.csv": evaluation.category_summary,
            "precision_recall.csv": evaluation.precision_recall,
            "coefficients.csv": category_effects,
        }
        for filename, table in tables.items():
            table.to_csv(ARTIFACT_DIR / filename, index=False)
        evaluation.positive_predictions.to_csv(
            ARTIFACT_DIR / "validation_positive_predictions.csv.gz",
            index=False,
            compression="gzip",
        )

        METRICS = {
            "reference_average_precision": reference_ap,
            "candidate_average_precision": candidate_ap,
            "delta_average_precision": candidate_ap - reference_ap,
            "reference_epic_spearman": 0.0,
            "candidate_epic_spearman": candidate_spearman,
            "delta_epic_spearman": candidate_spearman,
        }
        EXPERIMENT_PARAMETERS = CONFIG.to_dict() | {
            "feature": "oriented DNA dinucleotide at offsets [-1, 0]",
            "categories": list(DINUCLEOTIDE_LABELS),
            "class_weight_equivalent": "balanced through exact sample weights",
            "validation_population": "all whitelist positions",
            "reference": "constant score",
            "reference_spearman_policy": "0 by explicit no-ranking convention",
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
            json.dumps(EXPERIMENT_PARAMETERS, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Saved run:", output)
        print("Saved files:")
        for path in sorted(output.rglob("*")):
            if path.is_file():
                print(" -", path.relative_to(PROJECT_ROOT))
        """,
    ),
]


NOTEBOOK = nbf.v4.new_notebook(
    cells=CELLS,
    metadata={
        "kernelspec": {
            "display_name": "Python (EPIC EDA · Miniforge)",
            "language": "python",
            "name": "epic-eda",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "epic": {
            "experiment_id": "EXP-001",
            "data_contract": "ml_project.epic_data",
            "baseline_module": "ml_project.epic_baseline",
        },
    },
)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(NOTEBOOK, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
