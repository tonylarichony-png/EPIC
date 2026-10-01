---
id: CNN-EXP-048
type: experiment
experiment_type: cnn-candidate
status: completed
decision: reject
date: 2026-09-28
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-047-B
tags:
  - ml/experiment
  - ml/cnn
  - ml/loss
  - ml/hard-negatives
---

# CNN-EXP-048 — ultra-tail risk refinement

← [[cnn_exp/_index|Реестр CNN]] · [[docs/CNN_EXP045_049_RESULTS_REVIEW_2026-10-01|Сводный разбор EXP045–049]]

## Гипотеза

Более сильный вес top-0.1% опасных отрицательных исправит самые высокие false positives и улучшит natural genome-wide AP относительно EXP047-B.

## Результат

| Метрика | EXP047-B | EXP048 | Δ |
|---|---:|---:|---:|
| AP@5dp | **0.104211926** | 0.082120732 | −0.022091194 |
| EPIC Spearman | **0.202291612** | 0.197690110 | −0.004601502 |

EXP048 действительно снизил score ultra-tail `0–0.1%`, но поднял более массовые отрицательные диапазоны `0.5–15%`. Локальная победа на узком FP-срезе ухудшила глобальное ранжирование.

## Решение

`reject`. Не продолжать увеличение веса ultra-tail. Primary AP имеет приоритет над привлекательной диагностикой отдельной когорты.

Источники:

- `artifacts/experiments/CNN_EXP048_WIDTH512_ULTRA_TAIL_RISK_REFINEMENT/run_001/EXP048_RESULT.json`;
- `artifacts/experiments/CNN_EXP048_TAIL_DIAGNOSTIC/run_001/diagnostic_summary.json`.
