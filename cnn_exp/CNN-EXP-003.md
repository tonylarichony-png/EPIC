---
id: CNN-EXP-003
type: experiment
experiment_type: cnn-candidate
status: completed
decision: review
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

# CNN-EXP-003 — RF+

← [[cnn_exp/_index.md|Реестр CNN]] · [[cnn_exp/CNN-EXP-002.md|Текущий champion]]

## Pre-registration

- **Гипотеза:** RF+ даст улучшение метрик больше зависимостей и логики
- **Единственное изменение:** TODO — заполнить до запуска.
- **Что заморожено:** split, generator, targets, loss, optimizer, validation protocol.
- **Reference:** [[cnn_exp/CNN-EXP-002.md|CNN-EXP-002]].
- **Notebook:** [[notebooks/experiments/CNN_experiments/CNN_EXP003_rf.ipynb]].
- **Критерий:** AP выше reference, EPIC Spearman не ниже reference; проверить 3/3 contigs.

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| CNN-EXP-002 | 0.014379453 | 0.174194808 |
| **CNN-EXP-003** | **0.020911441** | **0.187888409** |

- Δ AP: `+0.006531988` (+45.43%).
- Δ EPIC Spearman: `+0.013693601`.

### Candidate по contigs

| Segment | AP | EPIC Spearman |
|---|---:|---:|
| overall | 0.020911441 | 0.187888409 |
| NC_064034.1 | 0.019997224 | 0.202334419 |
| NC_064041.1 | 0.018589801 | 0.227961321 |
| NC_064042.1 | 0.024519418 | 0.220358383 |

### Графики

![[artifacts/experiments/CNN_EXP003/run_001/plots/01_training_losses.png]]

![[artifacts/experiments/CNN_EXP003/run_001/plots/02_overall_comparison.png]]

![[artifacts/experiments/CNN_EXP003/run_001/plots/03_per_contig_comparison.png]]

### Финальные заметки

—

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

-
