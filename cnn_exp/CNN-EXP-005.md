---
id: CNN-EXP-005
type: experiment
experiment_type: cnn-candidate
status: completed
decision: adopt
date: 2026-09-21
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-004
tags:
  - ml/experiment
  - ml/cnn
---

# CNN-EXP-005 — STEP++ exp

← [[cnn_exp/_index.md|Реестр CNN]] · [[cnn_exp/CNN-EXP-004.md|Текущий champion]]

## Pre-registration

- **Гипотеза:** увеличение шагов даст  явный прирост AP
- **Единственное изменение:** TODO — заполнить до запуска.
- **Что заморожено:** split, generator, targets, loss, optimizer, validation protocol.
- **Reference:** [[cnn_exp/CNN-EXP-004.md|CNN-EXP-004]].
- **Notebook:** [[notebooks/experiments/CNN_experiments/CNN_EXP005_step_exp.ipynb]].
- **Критерий:** AP выше reference, EPIC Spearman не ниже reference; проверить 3/3 contigs.

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель          |   Validation AP |   EPIC Spearman |
| --------------- | --------------: | --------------: |
| CNN-EXP-004     |     0.028371535 |     0.187643022 |
| **CNN-EXP-005** | **0.051213682** | **0.208714182** |

- Δ AP: `+0.022842147` (+80.51%).
- Δ EPIC Spearman: `+0.021071160`.

### Candidate по contigs

| Segment     |          AP | EPIC Spearman |
| ----------- | ----------: | ------------: |
| overall     | 0.051213682 |   0.208714182 |
| NC_064034.1 | 0.055117152 |   0.231524769 |
| NC_064041.1 | 0.045130924 |   0.251953770 |
| NC_064042.1 | 0.051911817 |   0.241364878 |

### Графики

![[artifacts/experiments/CNN_EXP005/run_001/plots/01_training_losses.png]]

![[artifacts/experiments/CNN_EXP005/run_001/plots/02_overall_comparison.png]]

![[artifacts/experiments/CNN_EXP005/run_001/plots/03_per_contig_comparison.png]]

### Финальные заметки

увеличенное число шагов сильно улучшает AP

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

-
