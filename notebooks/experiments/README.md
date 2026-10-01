# Experiment notebooks

Здесь хранится по одному notebook на каждый новый геномный эксперимент:
`EXP-xxx_<slug>.ipynb`.

Не копируйте в них чтение BED, pooling повторностей и разбиение contig. Общая
подготовка находится в `src/ml_project/epic_data.py`. Новый notebook, карточку
и по умолчанию отдельный модуль нового признака создаёт команда из корня
проекта. Она наследует последний сохранённый `adopt`-champion, но ничего не
запускает:

Каждый результат обязательно показывает **две** метрики: Average Precision по
всем позициям и EPIC Spearman по dense ranks положительных target.

```bat
conda activate epic-eda
new-epic-experiment.cmd
```

Подробности: `docs/epic_experiment_workflow.md`.

## Выполненные notebooks

- `EXP-001_dinucleotide_baseline.ipynb` — полный точный baseline по
  ориентированной паре `[-1,0]`; решение и метрики находятся в
  `experiments/EXP-001.md`.
- `EXP-002_4mer_lookup.ipynb` — полный 4-mer lookup против 2-mer; `adopt`.
- `EXP-003_6mer_lookup.ipynb` — полный 6-mer lookup против 4-mer; `adopt`,
  champion ширины контекста.
- `EXP-004_8mer_lookup.ipynb` — полный 8-mer lookup против 6-mer; `reject`:
  прирост AP ниже порога и EPIC Spearman снизился.

EXP-002/003/004 — точные категориальные lookup-эксперименты, поэтому в них нет
solver, epochs и loss-графика. Их обучающая диагностика — support категорий,
сила shrinkage, coverage и время полного прохода. Добавление признаков и
обучение логистической регрессии начинается только в следующем эксперименте.

- `EXP-005_hierarchical_6mer_logistic.ipynb` — выполненный sparse
  `2/4/6-mer` L2 logistic regression; AP вырос на 0.347%, но формальный порог
  +1% не пройден, поэтому решение `reject`.
- `EXP-006_gc201_logistic.ipynb` — выполненный linear GC201 experiment;
  Spearman вырос, но AP снизился на 2.015% против EXP-005, решение `reject`.
- `EXP-007_gc201_bins_logistic.ipynb` — выполненный fixed categorical GC201
  experiment; AP **0.002053635**, Spearman **0.104302**, решение `adopt`.
- `EXP-008_k2_gc_interaction.ipynb` — выполненный interaction experiment;
  AP вырос на 0.519%, Spearman немного снизился, CV gate не пройден, решение
  `reject`. Полный отрицательный результат сохранён.

- `EXP-009_cpg_oe201_logistic.ipynb` — выполненный fixed CpG O/E201 experiment;
  AP вырос на 15.10%, Spearman — с 0.104302 до 0.115517, решение `adopt`.

## Новый ручной workbench

Генератор создаёт source-only notebook со STOP-точками перед feature pilot,
полным train count, paired CV, full fit и validation. Новый признак по умолчанию
реализуется в `src/ml_project/epic_experiments/`; мы заполняем функции вместе,
а пользователь по-прежнему вручную запускает каждую ячейку.
