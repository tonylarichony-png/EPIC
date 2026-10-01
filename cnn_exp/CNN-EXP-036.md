---
id: CNN-EXP-036
type: experiment
experiment_type: cnn-candidate
status: completed
decision: adopt
date: 2026-09-24
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-004
tags:
  - ml/experiment
  - ml/cnn
---

# CNN-EXP-036 — WIDTH64 RF1029

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** увеличение ширины backbone с 32 до 64 каналов улучшит качество при неизменном RF=1029.
- **Единственное изменение:** channels 32 → 64.
- **Что заморожено:** split, generator, targets, loss, optimizer, RF, validation protocol.
- **Reference:** [[cnn_exp/CNN-EXP-004.md|CNN-EXP-004]].
- **Критерий:** AP выше reference; проверить стабильность по contigs и структуру ошибок.

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| CNN-EXP-004 (`presence_probability`) | 0.027444381 | 0.169951687 |
| **CNN-EXP-036** | **0.033996892** | **0.177041367** |

- Δ AP: `+0.006552511` (**+23.88%**).
- Δ EPIC Spearman: `+0.007089679`.
- **Решение:** `adopt`. Сравнение выполнено по одинаковому выходу
  `presence_probability`; прежняя автоматическая строка смешивала candidate
  presence с reference `expected_signal`.

### Candidate по contigs

| Segment | AP | EPIC Spearman |
|---|---:|---:|
| NC_064034.1 | 0.033782513 | 0.191207334 |
| NC_064041.1 | 0.028973147 | 0.221014023 |
| NC_064042.1 | 0.038690374 | 0.206499785 |
| overall | 0.033996892 | 0.177041367 |

### Графики

![[artifacts/experiments/CNN_EXP036_WIDTH64_RF1029/run_001/plots/01_training_losses.png]]

![[artifacts/experiments/CNN_EXP036_WIDTH64_RF1029/run_001/plots/02_overall_comparison.png]]

![[artifacts/experiments/CNN_EXP036_WIDTH64_RF1029/run_001/plots/03_per_contig_comparison.png]]

### Финальные заметки

WIDTH64 capacity test. Review strong validation, error populations and cross-contig feature probe first.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

- WIDTH64 принят как подтверждённое улучшение WIDTH32 при 10k шагах.
- Позднее заменён новым champion CNN-EXP-041 WIDTH128 @25k.
