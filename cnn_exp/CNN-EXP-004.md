---
id: CNN-EXP-004
type: experiment
experiment_type: cnn-candidate
status: completed
decision: adopt
date: 2026-09-21
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-002
tags:
  - ml/experiment
  - ml/cnn
---

# CNN-EXP-004 — RF++

← [[cnn_exp/_index.md|Реестр CNN]] · [[cnn_exp/CNN-EXP-002.md|Текущий champion]]

## Pre-registration

- **Гипотеза:** увеличиваю ещё сильнее RF до максимума
- **Единственное изменение:** TODO — заполнить до запуска.
- **Что заморожено:** split, generator, targets, loss, optimizer, validation protocol.
- **Reference:** [[cnn_exp/CNN-EXP-002.md|CNN-EXP-002]].
- **Notebook:** [[notebooks/experiments/CNN_experiments/CNN_EXP004_rf.ipynb]].
- **Критерий:** AP выше reference, EPIC Spearman не ниже reference; проверить 3/3 contigs.

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| CNN-EXP-002 | 0.014379453 | 0.174194808 |
| **CNN-EXP-004** | **0.028371535** | **0.187643022** |

- Δ AP: `+0.013992082` (+97.31%).
- Δ EPIC Spearman: `+0.013448214`.

### Candidate по contigs

| Segment | AP | EPIC Spearman |
|---|---:|---:|
| overall | 0.028371535 | 0.187643022 |
| NC_064034.1 | 0.026795568 | 0.201736289 |
| NC_064041.1 | 0.025171724 | 0.234365094 |
| NC_064042.1 | 0.033451802 | 0.219286576 |

### Графики

![[artifacts/experiments/CNN_EXP004/run_001/plots/01_training_losses.png]]

![[artifacts/experiments/CNN_EXP004/run_001/plots/02_overall_comparison.png]]

![[artifacts/experiments/CNN_EXP004/run_001/plots/03_per_contig_comparison.png]]

### Финальные заметки

RF++ охват очень сильно влияет нужно смотреть шире!

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

-
