---
type: registry
entity: cnn-experiment
---

# Реестр CNN-экспериментов

Отдельная серия экспериментов с моделями, которые предсказывают профиль
инициации транскрипции непосредственно из последовательности ДНК.

Основной план: [[cnn_exp/CNN-ROADMAP.md|рабочая CNN-roadmap]]. Биологические
гипотезы и исходное проектирование: [[notebooks/experiments/CNN_experiments/CNN_plan.md|CNN plan]].

## Текущий champion

| ID | Модель | Validation AP | EPIC Spearman | Решение |
|---|---|---:|---:|---|
| [[cnn_exp/CNN-EXP-001.md\|CNN-EXP-001]] | компактная dilated residual CNN | 0.012360070 | 0.155747574 | `adopt` |
| [[cnn_exp/CNN-EXP-002.md\|CNN-EXP-002]] | presence + intensity CNN | **0.014379** | **0.174195** | `adopt` |
| [[cnn_exp/CNN-EXP-003.md\|CNN-EXP-003]] | RF+ | 0.020911441 | 0.187888409 | `review` |
| [[cnn_exp/CNN-EXP-004.md\|CNN-EXP-004]] | RF++ | 0.028371535 | 0.187643022 | `adopt` |
| [[cnn_exp/CNN-EXP-005.md\|CNN-EXP-005]] | STEP++ exp | 0.051213682 | 0.208714182 | `adopt` |
| [[cnn_exp/CNN-EXP-036.md\|CNN-EXP-036]] | WIDTH64 RF1029 | 0.033996892 | 0.177041367 | `adopt` |
| [[cnn_exp/CNN-EXP-037.md\|CNN-EXP-037]] | Frozen WIDTH64 Hard-FP Reranker | 0.008222070 | 0.102930993 | `reject` |
| [[cnn_exp/CNN-EXP-038.md\|CNN-EXP-038]] | LOW_TP score-matched residual | 0.034072849 | 0.177058339 | `reject` |
| [[cnn_exp/CNN-EXP-039.md\|CNN-EXP-039]] | WIDTH128 RF1029 @10k | 0.033810685 | 0.173279658 | `iterate` |
| [[cnn_exp/CNN-EXP-041.md\|CNN-EXP-041]] | WIDTH128 @25k RF1029 | 0.074412694 | 0.188400388 | `review` |
| [[cnn_exp/CNN-EXP-044.md\|CNN-EXP-044]] | WIDTH256 cosine 25k→50k | 0.093361726 | 0.195896953 | `review` |
| [[cnn_exp/CNN-EXP-043.md\|CNN-EXP-043]] | WIDTH256 @50k RF1029 | 0.071507846 | 0.187260523 | `review` |
| [[cnn_exp/CNN-EXP-042.md\|CNN-EXP-042]] | WIDTH256 @25k RF1029 | 0.075671327 | 0.187675476 | `review` |
| [[cnn_exp/CNN-EXP-047.md\|CNN-EXP-047]] | paired WIDTH512 risk-focused loss continuation | 0.104211926 | 0.202291612 | `reject` |
| [[cnn_exp/CNN-EXP-046B.md\|CNN-EXP-046B]] | frozen regional correction R32 | 0.098867560 | — | `reject` |
| [[cnn_exp/CNN-EXP-048.md\|CNN-EXP-048]] | ultra-tail risk refinement | 0.082120732 | 0.197690110 | `reject` |
| [[cnn_exp/CNN-EXP-049.md\|CNN-EXP-049 joint 50k]] | R16 A→FiLM→B joint cascade | 0.104208798 | 0.130878357 | `reject / no new champion` |

Последний сводный разбор с таблицами и графиками:
[[docs/CNN_EXP045_049_RESULTS_REVIEW_2026-10-01|CNN-EXP-045—049: итог последних экспериментов]].

## Правила серии

- Train, validation и test разделяются по contig согласно `contig_holdout_v1`.
- AP считается на полном whitelist, а не на сбалансированном батче.
- EPIC Spearman — Pearson correlation между dense ranks только среди позиций с `count > 0`.
- Test не используется для выбора архитектуры.
- Каждый следующий эксперимент изменяет одну заранее названную часть baseline.

## Все CNN-эксперименты

```query
path:"cnn_exp" -file:"_index"
```
