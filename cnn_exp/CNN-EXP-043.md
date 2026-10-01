---
id: CNN-EXP-043
type: experiment
experiment_type: cnn-candidate
status: completed
decision: review
date: 2026-09-25
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-042
tags:
  - ml/experiment
  - ml/cnn
  - ml/training-budget
---

# CNN-EXP-043 — WIDTH256 @50k RF1029

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** WIDTH256 @25k всё ещё training-budget limited; continuation до 50k улучшит natural-background ranking.
- **Единственное изменение:** training budget 25k → 50k.
- **Resume:** exact checkpoint EXP042 @25k с optimizer и sampler RNG.
- **Что заморожено:** WIDTH256, RF1029, LR=1e-3, batch=4, split, generator logic, targets, loss.
- **Reference:** [[cnn_exp/CNN-EXP-042.md|CNN-EXP-042]].

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| CNN-EXP-042 | 0.075671327 | 0.187675476 |
| **CNN-EXP-043** | **0.071507846** | **0.187260523** |

- Δ AP: `-0.004163481` (-5.50%).
- Δ EPIC Spearman: `-0.000414953`.

### Candidate по contigs

| Segment | AP | EPIC Spearman |
|---|---:|---:|
| NC_064034.1 | 0.070022086 | 0.205101848 |
| NC_064041.1 | 0.064961226 | 0.227637053 |
| NC_064042.1 | 0.079443746 | 0.221846282 |
| overall | 0.071507846 | 0.187260523 |

### Графики

![[artifacts/experiments/CNN_EXP043_WIDTH256_50K_RF1029/run_001/plots/01_training_losses.png]]

![[artifacts/experiments/CNN_EXP043_WIDTH256_50K_RF1029/run_001/plots/02_overall_comparison.png]]

![[artifacts/experiments/CNN_EXP043_WIDTH256_50K_RF1029/run_001/plots/03_per_contig_comparison.png]]

### Финальные заметки

WIDTH256 exact resume from EXP042 25k to 50k. LR, architecture, sampling and batch unchanged.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

-
