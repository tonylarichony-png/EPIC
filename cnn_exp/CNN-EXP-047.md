---
id: CNN-EXP-047
type: experiment
experiment_type: cnn-candidate
status: completed
decision: reject
date: 2026-09-27
seed: 42
split_version: contig_holdout_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-047-A
tags:
  - ml/experiment
  - ml/cnn
  - ml/loss
  - ml/hard-negatives
---

# CNN-EXP-047 — paired WIDTH512 risk-focused loss continuation

← [[cnn_exp/_index.md|Реестр CNN]]

## Pre-registration

- **Гипотеза:** end-to-end акцент на опасном хвосте train negatives даст крупный прирост natural nucleotide AP относительно matched continuation.
- **Branch point:** EXP045 WIDTH512 @50k; одинаковые веса, Adam moments и RNG.
- **A:** исходный loss, ещё 20k шагов.
- **B:** тот же режим, но половина negative objective равномерно распределена между четырьмя фиксированными danger strata.
- **Единственное различие A/B:** веса отрицательной части presence-loss.
- **Validation:** один полный финальный проход на ветвь; validation не используется для выбора checkpoint или порогов.
- **Adopt:** B−A AP ≥ 0.02 и положительная ΔAP на каждом validation contig.

## Автоматический отчёт

<!-- auto:cnn-experiment-report:start -->

## Итоговое сравнение

| Segment     |   EXP045 AP | A control AP | B risk-focused AP |          B−A |
| ----------- | ----------: | -----------: | ----------------: | -----------: |
| NC_064034.1 | 0.095690521 |  0.092303176 |       0.103602500 | +0.011299324 |
| NC_064041.1 | 0.091407336 |  0.086973971 |       0.096008027 | +0.009034056 |
| NC_064042.1 | 0.105202063 |  0.101721784 |       0.111698221 | +0.009976437 |
| overall     | 0.097630094 |  0.093970943 |       0.104211926 | +0.010240984 |

- Pre-registered decision: **REJECT_RISK_FOCUSED_LOSS**.
- B−A overall AP: `+0.010240984`.
- B−EXP045 overall AP: `+0.006581832`.
- AP 0.20 reached: `False`.
- Risk-loss selection rule satisfied: `False`.

### Графики

![[artifacts/experiments/CNN_EXP047_WIDTH512_PAIRED_RISK_LOSS_CONTINUATION/run_001/plots/01_training_losses.png]]

![[artifacts/experiments/CNN_EXP047_WIDTH512_PAIRED_RISK_LOSS_CONTINUATION/run_001/plots/02_overall_comparison.png]]

![[artifacts/experiments/CNN_EXP047_WIDTH512_PAIRED_RISK_LOSS_CONTINUATION/run_001/plots/03_per_contig_comparison.png]]

![[artifacts/experiments/CNN_EXP047_WIDTH512_PAIRED_RISK_LOSS_CONTINUATION/run_001/plots/04_paired_training_losses.png]]

![[artifacts/experiments/CNN_EXP047_WIDTH512_PAIRED_RISK_LOSS_CONTINUATION/run_001/plots/05_exp045_vs_A_vs_B_ap.png]]

### Вывод

Ветка B сравнивается с matched continuation A, поэтому дополнительное обучение отделено от эффекта risk-focused loss. Validation остаётся exploratory: этот split просматривался в предыдущих экспериментах.

<!-- auto:cnn-experiment-report:end -->

## Ручные заметки

- EXP047 является exploratory: validation многократно просматривался ранее.
