---
id: CNN-EXP-046B
type: experiment
experiment_type: cnn-candidate
status: completed
decision: reject
date: 2026-09-26
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
reference_experiment: CNN-EXP-045
tags:
  - ml/experiment
  - ml/cnn
  - ml/regional
  - ml/frozen-correction
---

# CNN-EXP-046B — frozen regional correction R32

← [[cnn_exp/_index|Реестр CNN]] · [[docs/CNN_EXP045_049_RESULTS_REVIEW_2026-10-01|Сводный разбор EXP045–049]]

## Гипотеза

Frozen regional head поверх признаков EXP045 сможет исправить ranking nucleotide score без изменения backbone и основного nucleotide head.

## Протокол

- backbone и nucleotide head заморожены;
- strand-specific region size `R32`;
- вариант и `alpha=0.25` выбраны по pooled train-contig OOF AP@5dp;
- validation не использовалась для обучения, mining или выбора конфигурации.

## Результат

| Метрика | EXP045 | EXP046B | Δ |
|---|---:|---:|---:|
| Overall AP@5dp | 0.097630094 | **0.098867560** | +0.001237466 |

Все три validation contig улучшились, LOW_TP guard не нарушен. Однако эффект слишком мал для нужного скачка и ветка отклонена.

Источник: `artifacts/experiments/CNN_EXP046B_FROZEN_REGIONAL_CORRECTION_R32/run_001/EXP046B_RESULT.json`.
