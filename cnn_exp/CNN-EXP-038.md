---
id: CNN-EXP-038
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
  - ml/low-tp
  - ml/residual
---

# CNN-EXP-038 — LOW_TP score-matched residual test

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** frozen WIDTH64 features содержат residual LOW_TP signal,
  который можно добавить к старому EXP036 logit, не разрушая baseline ranking.
- **Единственное изменение:** linear residual correction к old presence logit.
- **Что заморожено:** WIDTH64 backbone, RF1029, old presence-head,
  old intensity-head, split.
- **Reference:** [[cnn_exp/CNN-EXP-036.md|CNN-EXP-036]].

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| CNN-EXP-036 WIDTH64 | 0.033996892 | 0.177041367 |
| CNN-EXP-038 residual | **0.034072849** | **0.177058339** |

- Δ AP: `+0.000075957` (**+0.22%**).
- Улучшение AP формально присутствует на `3 / 3` validation-contigs.
- Train-only CV выбрал `α=0.1`, но выигрыш над `α=0` составил только
  `0.000059443` mean AP.

### LOW_TP recovery

| Бюджет | Восстановлено LOW_TP | Δ FP |
|---:|---:|---:|
| Top 1% | 0 | −34 |
| Top 5% | 0 | −145 |
| Top 10% | 4 из 14 052 | −63 |

### Решение

`reject` — положительный residual-сигнал слишком мал для принятия модели.
Эксперимент безопасно сохранил исходное ранжирование EXP036, но фактически не
восстановил LOW_TP. Дополнительная head и inference не оправдываются приростом
AP `0.000076`.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

- Диагностический вывод: anchored residual значительно безопаснее полной
  replacement-head EXP037, но WIDTH64 frozen features не дают материального
  запаса для LOW_TP.
- Рабочей моделью после эксперимента оставалась CNN-EXP-036.
