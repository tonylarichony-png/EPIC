---
id: CNN-EXP-044
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
  - ml/lr-schedule
---

# CNN-EXP-044 — WIDTH256 cosine 25k→50k

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** после 25k cosine LR 3e-4→3e-5 сохранит/улучшит generalization относительно constant 1e-3.
- **Branch point:** EXP042 WIDTH256 @25k.
- **Adam moments:** сохранены.
- **Sampler RNG:** сохранён.
- **Единственное изменение:** LR trajectory после 25k.
- **Checkpoints:** 30k / 35k / 40k / 45k / 50k.

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| CNN-EXP-042 | 0.075671327 | 0.187675476 |
| **CNN-EXP-044** | **0.093361726** | **0.195896953** |

- Δ AP: `+0.017690399` (+23.38%).
- Δ EPIC Spearman: `+0.008221477`.

### Candidate по contigs

| Segment | AP | EPIC Spearman |
|---|---:|---:|
| NC_064034.1 | 0.092442775 | 0.212594807 |
| NC_064041.1 | 0.088141615 | 0.241664827 |
| NC_064042.1 | 0.098942577 | 0.233617872 |
| overall | 0.093361726 | 0.195896953 |

### Графики

![[artifacts/experiments/CNN_EXP044_WIDTH256_25K_TO_50K_COSINE_LR/run_001/plots/01_training_losses.png]]

![[artifacts/experiments/CNN_EXP044_WIDTH256_25K_TO_50K_COSINE_LR/run_001/plots/02_overall_comparison.png]]

![[artifacts/experiments/CNN_EXP044_WIDTH256_25K_TO_50K_COSINE_LR/run_001/plots/03_per_contig_comparison.png]]

### Финальные заметки

WIDTH256 branch from EXP042 @25k. Adam moments and sampler trajectory restored; only LR changed to cosine 3e-4→3e-5.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

-
