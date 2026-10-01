---
id: CNN-EXP-037
type: experiment
experiment_type: cnn-candidate
status: completed
decision: reject
date: 2026-09-24
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-036
tags:
  - ml/experiment
  - ml/cnn
  - ml/hard-negative
  - ml/ranking
---

# CNN-EXP-037 — Frozen WIDTH64 Hard-FP Reranker

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** hard-negative sampling + ranking objective улучшат
  natural-background AP при frozen WIDTH64 backbone.
- **Единственное изменение:** новая linear presence-head и objective/sampling.
- **Что заморожено:** WIDTH64 backbone, RF1029, split, sequence representation,
  old intensity-head.
- **Reference:** [[cnn_exp/CNN-EXP-036.md|CNN-EXP-036]].

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| CNN-EXP-036 WIDTH64 | **0.033996892** | **0.177041367** |
| CNN-EXP-037 reranker | 0.008222070 | 0.102930993 |

- Δ AP: `-0.025774822` (**−75.82%**).
- Δ EPIC Spearman: `-0.074110374` (**−41.86%**).
- Улучшенных validation-contigs: `0 / 3`.
- Baseline EXP036 внутри notebook точно воспроизведён: AP `0.033996892`.

### Результат по validation-contigs

| Contig | EXP036 AP | EXP037 AP | Δ AP |
|---|---:|---:|---:|
| NC_064034.1 | 0.033782513 | 0.007611207 | −0.026171305 |
| NC_064041.1 | 0.028973147 | 0.007120696 | −0.021852451 |
| NC_064042.1 | 0.038690374 | 0.010076506 | −0.028613867 |

### Cross-contig CV внутри train

| Конфигурация | Hard fraction | λ rank | Mean AP | Mean positive Spearman |
|---|---:|---:|---:|---:|
| **hard25_lambda0p0** | 0.25 | 0.0 | **0.009860** | **0.110448** |
| hard25_lambda0p1 | 0.25 | 0.1 | 0.006264 | 0.080463 |
| hard25_lambda0p3 | 0.25 | 0.3 | 0.003524 | 0.035786 |
| hard25_lambda1p0 | 0.25 | 1.0 | 0.001332 | −0.054872 |
| hard50_lambda0p0 | 0.50 | 0.0 | 0.002464 | 0.006119 |

Лучшей из проверенных конфигураций стала `hard25_lambda0p0`, то есть BCE без
pairwise-компоненты. Рост `λ` монотонно ухудшал AP на каждом held-out
train-contig; увеличение доли HARD_FP до 50% также резко ухудшало результат.

### Что произошло в верхней части ранжирования

| Бюджет | Baseline precision | Reranker precision | Baseline recall | Reranker recall | HARD_FP осталось |
|---:|---:|---:|---:|---:|---:|
| Top 1% | 0.02284 | 0.01277 | 0.34054 | 0.19041 | 20 000 → 9 079 |
| Top 5% | 0.00817 | 0.00611 | 0.60891 | 0.45517 | 20 000 → 16 359 |
| Top 10% | 0.00496 | 0.00417 | 0.74004 | 0.62220 | 20 000 → 18 764 |

В top-1% reranker убрал 54.6% заранее выделенной HARD_FP-когорты, но общее
число FP выросло с `818 936` до `827 446`. Освободившиеся места заняли другие
отрицательные позиции, а recall TP упал на 15.0 процентного пункта.

### Решение

`reject` — checkpoint/head EXP037 не использовать.

Эксперимент опровергает гипотезу, что глобальная замена presence-head,
обученная на статическом наборе HARD_FP, улучшит естественный genome-wide AP.
Диагностическая разделимость TP/HARD_FP внутри отобранной когорты не
перенеслась на полную популяцию отрицательных позиций.

EXP037 не доказывает отсутствие полезной информации в WIDTH64 representation.
Он показывает, что оптимизация против фиксированных HARD_FP переупорядочивает
ошибки: известные FP опускаются, но поднимаются ранее не выбранные TN и
демонтируются уже правильно найденные TP. Pairwise ranking усиливает этот
эффект.

Рабочей моделью остаётся CNN-EXP-036 WIDTH64. Следующая проверка должна быть
направлена на LOW_TP относительно score-matched естественных TN и использовать
старый EXP036 logit как якорь:

`score = old_logit + α × residual`, включая `α=0` в train-only CV.

Критерий — полный natural-background AP и число новых FP на один восстановленный
LOW_TP. Глобальную replacement-head, MLP поверх той же статической HARD_FP
выборки и дальнейший перебор `λ` не продолжать.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

- **Статус:** отрицательный, но методологически полезный эксперимент.
- Validation использовался один раз после выбора конфигурации по cross-contig
  CV внутри девяти train-contigs.
- LOW_TP отдельно не оптимизировались. Их recall немного вырос, но общий recall
  упал сильнее; это не считается решением LOW_TP.
- Артефакты: `artifacts/experiments/CNN_EXP037_FROZEN_WIDTH64_HARDFP_RERANKER/run_001/`.
- Основные файлы: `EXP037_RESULT.json`, `cross_contig_cv_summary.csv`,
  `final_validation_metrics.csv`, `top_budget_diagnostics.csv`.
