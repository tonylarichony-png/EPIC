---
id: CNN-EXP-041
type: experiment
experiment_type: cnn-candidate
status: completed
decision: review
date: 2026-09-25
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-039
tags:
  - ml/experiment
  - ml/cnn
  - ml/training-budget
---

# CNN-EXP-041 — WIDTH128 @25k RF1029

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** WIDTH128 @10k недообучен; продолжение до 25k улучшит natural-background ranking.
- **Единственное изменение:** training budget 10k → 25k.
- **Resume:** точный checkpoint EXP039 @10k вместе с optimizer и sampler RNG state.
- **Что заморожено:** WIDTH128, RF1029, LR=1e-3, batch=4, split, generator logic, targets, loss.
- **Reference:** [[cnn_exp/CNN-EXP-039.md|CNN-EXP-039]].

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| CNN-EXP-039 | 0.033810685 | 0.173279658 |
| **CNN-EXP-041** | **0.074412694** | **0.188400388** |

- Δ AP: `+0.040602010` (+120.09%).
- Δ EPIC Spearman: `+0.015120730`.

### Candidate по contigs

| Segment | AP | EPIC Spearman |
|---|---:|---:|
| NC_064034.1 | 0.072823387 | 0.201311216 |
| NC_064041.1 | 0.070200707 | 0.238796696 |
| NC_064042.1 | 0.080031185 | 0.222172707 |
| overall | 0.074412694 | 0.188400388 |

### Графики

![[artifacts/experiments/CNN_EXP041_WIDTH128_25K_RF1029/run_001/plots/01_training_losses.png]]

![[artifacts/experiments/CNN_EXP041_WIDTH128_25K_RF1029/run_001/plots/02_overall_comparison.png]]

![[artifacts/experiments/CNN_EXP041_WIDTH128_25K_RF1029/run_001/plots/03_per_contig_comparison.png]]

### Финальные заметки

WIDTH128 exact resume from EXP039 10k to 25k. LR, architecture, sampling and batch unchanged.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

- **Статус:** новый подтверждённый champion серии.
- Checkpoint: `artifacts/experiments/CNN_EXP041_WIDTH128_25K_RF1029/run_001/width128_25k_final.pt`.
- Продолжение обучения имеет больший приоритет, чем очередная frozen head:
  15k дополнительных шагов дали `+0.0406 AP`, тогда как EXP038 дал
  `+0.000076 AP`.
