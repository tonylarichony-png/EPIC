---
id: CNN-EXP-007
type: experiment
experiment_type: cnn-candidate
status: planned
decision: pending
date: 2026-09-22
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-005
tags:
  - ml/experiment
  - ml/cnn
---

# CNN-EXP-007 — BPE++ GPU

← [[cnn_exp/_index.md|Реестр CNN]] · [[cnn_exp/CNN-EXP-005.md|Текущий champion]]

## Pre-registration

- **Гипотеза:** Увеличить BPE мерджи и перенести вычисления на GPU
- **Единственное изменение:** TODO — заполнить до запуска.
- **Что заморожено:** split, generator, targets, loss, optimizer, validation protocol.
- **Reference:** [[cnn_exp/CNN-EXP-005.md|CNN-EXP-005]].
- **Notebook:** [[notebooks/experiments/CNN_experiments/CNN_EXP007_bpe_gpu.ipynb]].
- **Критерий:** AP выше reference, EPIC Spearman не ниже reference; проверить 3/3 contigs.

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

Эксперимент ещё не сохранён финальной ячейкой notebook.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

-
