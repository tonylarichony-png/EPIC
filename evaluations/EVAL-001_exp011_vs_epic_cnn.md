---
id: EVAL-001
type: evaluation
status: completed
date: 2026-09-18
model: EXP-011/run_001
assembly: jaNemVect1.1
---

# EVAL-001 — EXP-011 против EPIC Solution

| Модель                  |      AP mean | EPIC Spearman mean |
| ----------------------- | -----------: | -----------------: |
| EXP-011 `GG+TA`         | **0.002848** |       **0.105124** |
| EPIC 8-mer baseline     |     0.001480 |           0.140200 |
| EPIC Solution CNN `big` | **0.131650** |       **0.391000** |

## Короткий вывод

- EXP-011 почти вдвое лучше EPIC 8-mer по AP, но слабее по Spearman.
- CNN превосходит EXP-011 примерно в 46 раз по AP и в 3.7 раза по Spearman.
- Validation правильно предсказала переносимость AP; логистический baseline завершён.
- Split обучения EXP-011 и CNN различается, поэтому это внешний benchmark, а не строгий ablation.

Артефакты: [[artifacts/evaluations/EVAL-001/exp011_official_test_metrics.csv|метрики]] · [[artifacts/evaluations/EVAL-001/comparison_with_epic_solution.csv|сравнение]] · [[artifacts/evaluations/EVAL-001/metadata.json|контракт]]
