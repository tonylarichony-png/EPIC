---
type: experiment-review
status: completed
date: 2026-10-01
primary_metric: genome-wide nucleotide AP@5dp
split_version: contig_holdout_v1
tags:
  - ml/cnn
  - ml/experiment-review
  - ml/cascade
  - ml/error-analysis
---

# CNN-EXP-045—049: итог последних экспериментов

Связано: [[cnn_exp/CNN-EXP-045|EXP045]], [[cnn_exp/CNN-EXP-047|EXP047]], [[docs/CNN_EXP049A_RATE_PILOT|EXP049A]], [[docs/CNN_EXP049B_JOINT_PILOT|EXP049B]], [[notebooks/experiments/CNN_experiments/CNN_EXP049_R16_COARSE_TO_FINE_CASCADE_DESIGN|дизайн EXP049]].

## Короткий итог

> [!summary]
> Лучший измеренный genome-wide nucleotide AP в этой серии остаётся у **EXP047-B: 0.104211926**. EXP049 joint после 50k получил **0.104208798**, то есть совпал с champion с точностью до `0.0000031 AP`, но не превзошёл его. До цели `AP=0.15` остаётся `0.045788` абсолютного AP, или `+43.94%` относительно champion.

Региональное conditioning полезно: в честном парном сравнении EXP049 joint превосходит zero-condition control на `+0.046080 AP` при 20k и на `+0.030431 AP` при 50k, причём на всех трёх validation contig. Однако оно не создало новый уровень качества: к 50k каскад сошёлся к тому же AP, который уже дал EXP047-B.

![[assets/cnn/exp045_049_review_2026-10-01/01_overall_ap_comparison.svg]]

## Сводная таблица

| Эксперимент | Основное изменение | Шаги | Nucleotide AP@5dp | EPIC Spearman | Итог |
|---|---|---:|---:|---:|---|
| EXP045 | WIDTH512, RF1029, two-phase LR | 50k | 0.097630094 | 0.200124323 | сильный reference |
| EXP046B | frozen R32 regional correction | — | 0.098867560 | — | `+0.001237`, reject |
| EXP047-A | matched continuation EXP045 | +20k | 0.093970943 | 0.192573186 | control continuation ухудшилась |
| **EXP047-B** | paired risk-focused loss | +20k | **0.104211926** | **0.202291612** | текущий AP champion; preregistered gate не пройден |
| EXP048 | ultra-tail risk refinement | +20k | 0.082120732 | 0.197690110 | сильный regress, reject |
| EXP049 control | B с нулевыми regional channels | 20k | 0.046371994 | 0.120001431 | matched control |
| EXP049 joint | online A → FiLM → B | 20k | 0.092451843 | 0.128554933 | большой эффект A относительно control |
| EXP049 control | continuation | 50k | 0.073778159 | 0.133377696 | продолжает учиться |
| **EXP049 joint** | continuation | 50k | **0.104208798** | 0.130878357 | равен EXP047-B по AP, не новый champion |

> [!warning] Сопоставимость Spearman
> AP@5dp рассчитан на тех же `83,807,486` natural validation positions и сопоставим между строками. Spearman EXP045/047/048 и пилотный evaluator EXP049 имеют различающиеся score heads/links и историю квантования, поэтому прямой межсемейный вывод по Spearman слабее. Внутри пары control/joint одного milestone сравнение корректно.

Числовой источник для графика: [[assets/cnn/exp045_049_review_2026-10-01/overall_metrics.csv]].

## EXP045: увеличение ширины до 512

- `14,708,738` trainable parameters, RF `1029`, 50k шагов.
- AP вырос до `0.097630094`.
- Raw EPIC Spearman: `0.200124323`.
- Эксперимент показал, что ёмкость и зрелый schedule ещё давали прирост, но сам по себе WIDTH512 не приблизил модель к `0.15`.

Источник: `artifacts/experiments/CNN_EXP045_WIDTH512_50K_TWO_PHASE_LR_RF1029/run_001/EXP045_WIDTH512_50K_RESULT.json`.

## EXP046/046B: региональный oracle и обучаемая коррекция

EXP046B использовал frozen backbone/head и обучаемую R32-коррекцию. Конфигурация была выбрана только по train-contig OOF:

| Метрика | EXP045 | EXP046B | Δ |
|---|---:|---:|---:|
| Overall AP@5dp | 0.097630094 | 0.098867560 | +0.001237466 |
| NC_064034.1 | 0.095690521 | 0.096437130 | +0.000746609 |
| NC_064041.1 | 0.091407336 | 0.092722893 | +0.001315558 |
| NC_064042.1 | 0.105202063 | 0.106932924 | +0.001730860 |

Рост согласован по contig, но слишком мал для продолжения frozen-correction ветки.

### Что означает oracle по разрешениям

| Ширина региона | Oracle AP@5dp | Доля фиксированных HARD_FP в пустых регионах |
|---:|---:|---:|
| R1 | 1.000000 | 100.00% |
| R2 | 0.699112 | 97.33% |
| R4 | 0.497604 | 93.22% |
| R8 | 0.369152 | 88.22% |
| R16 | 0.289735 | 83.46% |
| R32 | 0.238318 | 79.21% |
| R64 | 0.204896 | 75.43% |
| R128 | 0.181955 | 71.65% |

![[assets/cnn/exp045_049_review_2026-10-01/05_oracle_resolution_curve.svg]]

> [!danger] Oracle — это target leakage
> Oracle зануляет score в регионах, которые объявляются пустыми **по validation labels**. Уменьшение R механически раскрывает всё более точное положение target; R1 фактически сообщает бинарную метку каждой позиции. Поэтому oracle нельзя использовать как вход, teacher, distillation target, calibration target или доказательство обучаемости R4/R8/R16. Это только диагностика того, где находятся ошибки сохранённого EXP045 score.

Полный аудит: `artifacts/experiments/CNN_EXP046_REGIONAL_ACTIVITY_GATE/run_001/oracle_resolution_audit/oracle_resolution_audit.md`.

## EXP047: risk-focused loss

Парный дизайн отделил эффект loss от эффекта дополнительных шагов:

| Ветка | AP@5dp | Δ к EXP045 | Δ к control A |
|---|---:|---:|---:|
| A — обычное продолжение | 0.093970943 | −0.003659152 | — |
| B — risk-focused | **0.104211926** | +0.006581832 | +0.010240984 |

На каждом validation contig B улучшил A примерно на `0.0090–0.0113 AP`. Но заранее заданный порог `B−A ≥ 0.02` не был достигнут, поэтому гипотеза о **крупном** скачке от risk-loss отклонена. Практически checkpoint B остаётся лучшим AP reference.

## EXP048: чрезмерный акцент на ultra-tail

EXP048 снизил score самых экстремальных отрицательных, но ухудшил более широкий dangerous tail:

| Negative band | Mean EXP047-B | Mean EXP048 | Δ048−047 |
|---|---:|---:|---:|
| 0–0.1% | 0.9276 | 0.8617 | −0.0659 |
| 0.1–0.2% | 0.8588 | 0.8243 | −0.0345 |
| 0.2–0.5% | 0.7916 | 0.7844 | −0.0073 |
| 0.5–1% | 0.7166 | 0.7338 | +0.0171 |
| 1–5% | 0.5325 | 0.5667 | +0.0342 |
| 5–15% | 0.2983 | 0.3226 | +0.0243 |

Итоговый AP упал с `0.104211926` до `0.082120732` (`−0.022091194`). Это важный отрицательный результат: оптимизация узкого top-0.1% хвоста не эквивалентна улучшению глобального AP; цена в широком отрицательном хвосте оказалась больше выигрыша.

## EXP049A: чему научилась Network A

| Checkpoint | Regional AP, R16 | Spearman count среди active R16 | Empty suppression при 99.5% recall |
|---|---:|---:|---:|
| A-v0 20k | **0.133937963** | 0.134444240 | **0.195664970** |
| A-v1 5k | 0.055724698 | 0.179522860 | 0.108324901 |
| A-v1 10k | 0.095152726 | 0.234085601 | 0.148124003 |
| A-v1 20k | 0.131841037 | **0.252141879** | 0.156131558 |

![[assets/cnn/exp045_049_review_2026-10-01/04_exp049a_regional_tradeoff.svg]]

A-v1 действительно научилась лучше упорядочивать интенсивность активных регионов: `+0.117698 Spearman` к A-v0. Но её regional AP сохранил только `98.43%` AP A-v0, ниже preregistered eligibility `99%`, а suppression пустых регионов ухудшился на `3.95 п.п.`. Следовательно, отдельно взятая A-v1 не прошла собственное правило выбора. Дальнейший joint pilot был оправдан только как exploratory end-to-end проверка, а не как подтверждённая победа региональной головы.

## EXP049B: joint против matched control

### Learning curve

![[assets/cnn/exp045_049_review_2026-10-01/02_exp049_learning_curve.svg]]

| Сравнение | 20k | 50k | Изменение 20k→50k |
|---|---:|---:|---:|
| Control AP | 0.046371994 | 0.073778159 | +0.027406165 |
| Joint AP | 0.092451843 | 0.104208798 | +0.011756955 |
| Joint − control | +0.046079849 | +0.030430639 | gap сократился на 0.015649211 |

Продолжение до 50k было полезно обеим веткам, но control рос быстрее. Это не обнуляет эффект A: при 50k conditioning всё ещё даёт `+0.03043 AP`. Однако уменьшение разрыва указывает, что часть огромного преимущества на 20k была ускорением оптимизации B, а не новым недостижимым для B решением.

### Стабильность по contig

![[assets/cnn/exp045_049_review_2026-10-01/03_exp049_per_contig_ap.svg]]

| Contig | Control 50k | Joint 50k | Joint−control | EXP047-B | Joint−EXP047-B |
|---|---:|---:|---:|---:|---:|
| NC_064034.1 | 0.072404916 | 0.108164010 | +0.035759093 | 0.103602500 | +0.004561510 |
| NC_064041.1 | 0.067657010 | 0.094356448 | +0.026699438 | 0.096008027 | −0.001651579 |
| NC_064042.1 | 0.080670010 | 0.107035054 | +0.026365044 | 0.111698221 | −0.004663167 |
| **Overall** | **0.073778159** | **0.104208798** | **+0.030430639** | **0.104211926** | **−0.000003128** |

Вывод устойчив: каскад лучше своего matched control на всех contig, но не лучше EXP047-B. Один contig улучшен относительно champion, два ухудшены.

## Сравнение с EPIC solution

| Модель | Protocol | AP/PRAUC | Spearman |
|---|---|---:|---:|
| EXP047-B | наш фиксированный validation holdout | 0.104212 | 0.202292 |
| EXP049 joint 50k | тот же набор validation positions | 0.104209 | 0.130878 |
| EPIC `wide`, 24k | demo-test halves, mean | 0.118940 | 0.374900 |
| EPIC `big`, 96k | demo-test halves, mean | **0.131650** | **0.391000** |

EPIC numbers нельзя вычитать напрямую из наших как paired delta: split, training volume и evaluation protocol различаются. Но направление доказательств однозначно: native multiscale tower + гораздо больший бюджет остаётся более сильным архитектурным reference, чем каскад R16→FiLM.

## Научное решение

1. **EXP047-B сохраняется как measured AP champion.** Формально его risk-loss гипотеза не прошла preregistered `+0.02`, но checkpoint является лучшим фактическим reference.
2. **EXP049 закрывается как полезный, но не прорывной результат.** Conditioning A несёт реальный сигнал, однако дальнейшее простое увеличение шагов или добавление ещё одной локальной головы не имеет достаточного обоснования для цели `0.15`.
3. **EXP048 не продолжать.** Он показал metric–objective mismatch: победа на ultra-tail FP ухудшила глобальное ранжирование.
4. **Oracle не использовать для выбора архитектуры.** R4/R8/R16 показывают ценность точной информации о местоположении target, но не показывают, что эту информацию можно извлечь из DNA.
5. Следующая архитектура должна одновременно сохранять нативное base-resolution представление, многоуровневый контекст и моделировать неоднородность/повторности target; новый каскад оправдан только если он меняет эту факторизацию, а не повторяет regional gate.

## Provenance и ограничения

- EXP045–048: локальные JSON/CSV в `artifacts/experiments/`.
- Oracle audit: полный локальный расчёт без нового inference/training.
- EXP049A/B: метрики перенесены из сохранённых Kaggle evaluation logs; checkpoints находились в экспортированных `exp049a_rate_pilot.zip` и `exp049b_joint_pilot.zip`.
- Validation split уже многократно просматривался, поэтому результаты являются exploratory architecture evidence, а не несмещённой оценкой leaderboard generalization.
- CSV для всех новых графиков лежат рядом в `assets/cnn/exp045_049_review_2026-10-01/`.
