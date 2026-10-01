---
id: CNN-EXP-039
type: experiment
experiment_type: cnn-candidate
status: completed
decision: iterate
date: 2026-09-25
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-036
tags:
  - ml/experiment
  - ml/cnn
  - ml/capacity
---

# CNN-EXP-039 — WIDTH128 RF1029

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** увеличение width 64 → 128 улучшит natural-background ranking.
- **Единственное изменение:** backbone channels 64 → 128.
- **Что заморожено:** RF1029, split, generator, targets, loss, optimizer,
  batch, steps, validation protocol.
- **Reference:** [[cnn_exp/CNN-EXP-036.md|CNN-EXP-036]].

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение @10k

| Модель | Validation AP | Presence Spearman | Intensity Spearman |
|---|---:|---:|---:|
| CNN-EXP-036 WIDTH64 | **0.033996892** | **0.177041367** | **0.188708663** |
| CNN-EXP-039 WIDTH128 | 0.033810685 | 0.173279658 | 0.182840765 |

- Δ AP: `-0.000186207` (−0.55%).
- AP улучшился на двух из трёх contigs, но overall остался ниже WIDTH64.
- К 10k EMA presence loss WIDTH128 стал на 2.87% ниже WIDTH64 и продолжал
  снижаться быстрее.

### Representation

Средний ROC-AUC LOW_TP против score-matched TN:

| Representation | Mean AUC |
|---|---:|
| WIDTH64 @10k | 0.5143 |
| **WIDTH128 @10k** | **0.5804** |

Улучшение присутствует на всех трёх validation-contigs. TP/HARD_FP linear
probe также вырос примерно с `0.852` до `0.866`.

### Решение

`iterate` — checkpoint @10k не принят как champion, но архитектура не
отклонена. Более богатое representation и продолжающееся снижение loss стали
основанием точно продолжить ту же optimizer/sampler trajectory до 25k в
CNN-EXP-041.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

- Не интерпретировать результат @10k как потолок WIDTH128.
- Продолжение без изменения архитектуры выполнено в CNN-EXP-041.
