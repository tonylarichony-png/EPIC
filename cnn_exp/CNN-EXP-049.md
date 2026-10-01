---
id: CNN-EXP-049
type: experiment
experiment_type: cnn-candidate
status: completed
decision: reject
date: 2026-10-01
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-047-B
tags:
  - ml/experiment
  - ml/cnn
  - ml/cascade
  - ml/film
  - ml/regional
---

# CNN-EXP-049 — R16 A→FiLM→B joint cascade

← [[cnn_exp/_index|Реестр CNN]] · [[docs/CNN_EXP045_049_RESULTS_REVIEW_2026-10-01|Полный отчёт с графиками]]

## Проверенная гипотеза

Network A с длинным контекстом выдаёт R16 log-rate и 16 latent channels. Network B получает их через FiLM после блоков `2,4,6,8` и обучается совместно с A. Проверяется, даёт ли эта факторизация новый уровень genome-wide nucleotide AP.

## Парный результат

| Milestone | Zero-condition control AP | Joint AP | Joint−control |
|---:|---:|---:|---:|
| 20k | 0.046371994 | 0.092451843 | +0.046079849 |
| 50k | 0.073778159 | **0.104208798** | +0.030430639 |

Joint улучшил matched control на всех validation contig. Следовательно, региональные признаки A информативны и FiLM ими реально пользуется.

## Почему experiment всё равно закрыт

- EXP049 joint 50k: `0.104208798 AP`.
- EXP047-B: `0.104211926 AP`.
- Разность: `−0.000003128 AP` — фактическое равенство, не новый champion.
- Joint-control gap сократился с `0.046080` до `0.030431`, поскольку control между 20k и 50k рос быстрее.
- До целевого `0.15` остаётся `0.045788 AP`.

## Решение

`reject` как основную архитектурную ветку для дальнейшего масштабирования. Сигнал A подтверждён, но дальнейшие шаги той же схемы не обоснованы как путь к `0.15`. Компоненты могут использоваться как evidence для будущей архитектуры, но не как повод добавлять ещё одну голову к тому же каскаду.

Подробности: [[docs/CNN_EXP049A_RATE_PILOT]], [[docs/CNN_EXP049B_JOINT_PILOT]], [[docs/CNN_EXP045_049_RESULTS_REVIEW_2026-10-01]].
