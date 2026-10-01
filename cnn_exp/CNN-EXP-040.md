---
id: CNN-EXP-040
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
  - ml/capacity
---

# CNN-EXP-040 — WIDTH256 RF1029

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** увеличение width 128 → 256 улучшит representation и/или natural-background ranking.
- **Единственное изменение:** backbone channels 128 → 256.
- **Что заморожено:** RF1029, split, generator, targets, loss, optimizer, batch, steps, validation protocol.
- **Reference:** [[cnn_exp/CNN-EXP-039.md|CNN-EXP-039]].

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

-
