---
id: CNN-EXP-045
type: experiment
experiment_type: cnn-candidate
status: completed
decision: review
date: 2026-09-25
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-044
tags:
  - ml/experiment
  - ml/cnn
  - ml/capacity
  - ml/overnight
---

# CNN-EXP-045 — WIDTH512 @50k RF1029

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** WIDTH512 улучшит ranking относительно зрелого WIDTH256.
- **Architecture change:** channels 256 → 512.
- **RF:** 1029, unchanged.
- **Training:** 50k steps.
- **LR:** 1e-3 through 25k; then cosine 3e-4→3e-5 through 50k.
- **Batch:** 4.
- **Validation:** one final full-holdout strong validation only.
- **Reference:** CNN-EXP-044.

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итог

| Segment | AP@5dp | Raw EPIC Spearman |
|---|---:|---:|
| Overall | **0.097630094** | **0.200124323** |
| NC_064034.1 | 0.095690521 | 0.216091096 |
| NC_064041.1 | 0.091407336 | 0.247023791 |
| NC_064042.1 | 0.105202063 | 0.238226041 |

- WIDTH512 превзошёл зрелый WIDTH256 reference и стал базой для EXP046–048.
- Результат подтвердил пользу capacity scaling, но оставил большой разрыв до AP `0.15`.
- Полное сравнение последующей ветки: [[docs/CNN_EXP045_049_RESULTS_REVIEW_2026-10-01]].

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

- Artifact: `artifacts/experiments/CNN_EXP045_WIDTH512_50K_TWO_PHASE_LR_RF1029/run_001/EXP045_WIDTH512_50K_RESULT.json`.
