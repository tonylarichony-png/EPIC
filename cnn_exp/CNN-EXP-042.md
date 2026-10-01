---
id: CNN-EXP-042
type: experiment
experiment_type: cnn-candidate
status: completed
decision: review
date: 2026-09-25
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-040
tags:
  - ml/experiment
  - ml/cnn
  - ml/training-budget
---

# CNN-EXP-042 — WIDTH256 @25k RF1029

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** WIDTH256 @10k недообучен; continuation до 25k улучшит natural-background ranking.
- **Единственное изменение:** training budget 10k → 25k.
- **Resume:** exact checkpoint EXP040 @10k с optimizer и sampler RNG.
- **Что заморожено:** WIDTH256, RF1029, LR=1e-3, batch=4, split, generator logic, targets, loss.
- **Reference:** [[cnn_exp/CNN-EXP-040.md|CNN-EXP-040]].

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| CNN-EXP-040 | 0.032517261 | 0.170113295 |
| **CNN-EXP-042** | **0.075671327** | **0.187675476** |

- Δ AP: `+0.043154066` (+132.71%).
- Δ EPIC Spearman: `+0.017562181`.

### Candidate по contigs

| Segment | AP | EPIC Spearman |
|---|---:|---:|
| NC_064034.1 | 0.075333735 | 0.200920284 |
| NC_064041.1 | 0.068388687 | 0.235345393 |
| NC_064042.1 | 0.082411454 | 0.225596234 |
| overall | 0.075671327 | 0.187675476 |

### Графики

![[artifacts/experiments/CNN_EXP042_WIDTH256_25K_RF1029/run_001/plots/01_training_losses.png]]

![[artifacts/experiments/CNN_EXP042_WIDTH256_25K_RF1029/run_001/plots/02_overall_comparison.png]]

![[artifacts/experiments/CNN_EXP042_WIDTH256_25K_RF1029/run_001/plots/03_per_contig_comparison.png]]

### Финальные заметки

WIDTH256 exact resume from EXP040 10k to 25k. LR, architecture, sampling and batch unchanged.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

-
