---
id: CNN-EXP-002
type: experiment
experiment_type: multitask-cnn
status: completed
decision: adopt
date: 2026-09-21
seed: 42
split_version: contig_holdout_v1
run_name: presence_intensity_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: CNN-EXP-001
implementation: module-and-notebook
hardware: AMD Radeon RX 6750 GRE 12GB
backend: torch-directml
tags:
  - ml/experiment
  - ml/cnn
  - ml/multitask
  - biology/transcription-initiation
---

# CNN-EXP-002 — presence + intensity

← [[cnn_exp/_index.md|Реестр CNN]] · [[cnn_exp/CNN-EXP-001.md|CNN baseline]] · [[cnn_exp/CNN-ROADMAP.md|Roadmap]]

> [!success] Итог
> Добавление intensity-head улучшило AP и EPIC Spearman глобально и на каждом
> validation-contig. `expected_signal` принимается как новый единый основной
> score, а CNN-EXP-002 — как новый CNN champion.

## Гипотеза

Прямое обучение на величине положительного сигнала даст backbone дополнительный
биологически содержательный supervision: модель будет учить не только наличие
инициации, но и различия в её интенсивности. Это должно улучшить Spearman без
ухудшения AP.

## Единственное изменение

Backbone, receptive field, генератор, split, seed, optimizer, batch size и
бюджет 10 000 шагов оставлены такими же, как в CNN-EXP-001.

Изменены только выход и objective:

```text
DNA → общий CNN backbone → 32 признака на позицию
                              ├─ presence-head
                              └─ intensity-head
```

- Presence target: `count > 0`.
- Presence loss: прежний importance-weighted BCE на полном whitelist.
- Intensity target: `log1p(count)`.
- Intensity loss: importance-weighted Huber только на `count > 0`.
- Общий loss: `presence_loss + 0.1 × intensity_loss`.
- Обучаемых параметров: **46 466**, всего на 33 больше CNN-EXP-001.

## Единый score

До validation были зафиксированы три способа чтения модели:

1. `presence_probability = sigmoid(presence_logit)`;
2. `intensity_prediction` — монотонно преобразованный intensity-выход;
3. `expected_signal = p × expm1(max(intensity, 0))` с последующим монотонным
   сжатием в диапазон `[0, 1]` для потокового AP.

Главным единым score для обеих метрик заранее выбран `expected_signal`.
Остальные два выхода являются диагностическими и не подбирались по validation.

## Обучение

| Параметр | Значение |
|---|---:|
| Steps | 10 000 |
| Batch size | 4 genomic tiles |
| Seed | 42 |
| Optimizer | custom DirectML Adam |
| Learning rate | 0.001 |
| Intensity loss weight | 0.1 |
| Huber delta | 1.0 |
| Train time | 359.60 сек |
| Time per step | 0.0360 сек |
| Увидено position-strand | 76 146 258 |
| Финальный EMA presence | 0.416737 |
| Финальный EMA intensity | 0.238785 |
| Финальный EMA total | 0.440616 |
| CPU fallback | отсутствует |

## Глобальные результаты

| Модель / score | Validation AP | EPIC Spearman | AP lift |
|---|---:|---:|---:|
| CNN-EXP-001 | 0.012360070 | 0.155747574 | 18.43× |
| EXP-002 presence probability | 0.013621 | 0.159182 | 20.31× |
| EXP-002 intensity prediction | 0.011986 | **0.177420** | 17.87× |
| **EXP-002 expected signal** | **0.014379** | **0.174195** | **21.44×** |

Относительно CNN-EXP-001:

- presence-head сама по себе улучшила AP примерно на **10.2%**;
- `expected_signal` улучшил AP примерно на **16.3%**;
- `expected_signal` улучшил Spearman на **0.01845**;
- чистый intensity-score дал максимальный диагностический Spearman, но уступил
  единому score по AP.

## Результаты по validation-contig

| Contig | AP EXP-001 | AP expected signal | Δ AP | Spearman EXP-001 | Spearman expected signal | Δ Spearman |
|---|---:|---:|---:|---:|---:|---:|
| NC_064034.1 | 0.013691 | **0.016163** | **+18.1%** | 0.165281 | **0.183534** | +0.018253 |
| NC_064041.1 | 0.010646 | **0.012891** | **+21.1%** | 0.194761 | **0.214187** | +0.019426 |
| NC_064042.1 | 0.012570 | **0.013813** | **+9.9%** | 0.183528 | **0.208917** | +0.025389 |

Выигрыш по обеим метрикам наблюдается на всех `3/3` validation-contigs.

## Интерпретация

Улучшение AP происходит двумя путями:

1. Presence-head улучшилась даже без использования intensity при inference.
   Наиболее вероятное объяснение — auxiliary intensity-task заставила общий
   backbone выучить более содержательные признаки активных стартов и выступила
   как multitask-регуляризация.
2. `expected_signal` дополнительно использует предсказанную силу и лучше
   ранжирует кандидатов, поэтому AP вырос сильнее, чем у одной presence-head.

Это сильное свидетельство полезности intensity-сигнала при фиксированном seed и
бюджете, но не окончательное доказательство причинности: для него нужны повторы
на нескольких seeds и, при необходимости, абляция коэффициента loss.

## Наблюдения и ограничения

- Глобальный Spearman ниже per-contig Spearman: межконтинговый сдвиг шкалы
  score/count остаётся.
- Округление scores создаёт ties, поэтому основные выводы делаются по исходным
  float scores.
- Выполнен один seed.
- 10 000 шагов — фиксированный сравнительный бюджет, а не найденный optimum.
- Test не использовался.

## Артефакты

- Notebook: [[notebooks/experiments/CNN_experiments/CNN_EXP002.ipynb]]
- Model: [[src/ml_project/epic_cnn/model.py]]
- DirectML losses: [[src/ml_project/epic_cnn/directml.py]]
- Multitask forward: [[src/ml_project/epic_cnn/profile.py]]
- Evaluator: [[src/ml_project/epic_cnn/evaluation.py]]
- Checkpoint: [[artifacts/experiments/CNN_EXP002/presence_intensity_v1/exp002_final.pt]]
- Training history: [[artifacts/experiments/CNN_EXP002/presence_intensity_v1/training_history.csv]]
- Validation metrics: [[artifacts/experiments/CNN_EXP002/presence_intensity_v1/validation_metrics.csv]]

## Решение

`adopt` — принять `expected_signal` CNN-EXP-002 как нового CNN champion.
Intensity-head сохраняется в дальнейших экспериментах.

## Следующие действия

- [ ] Оценить checkpoints 2 000 / 5 000 / 10 000 одним validation-протоколом,
      чтобы увидеть зависимость качества от бюджета обучения.
- [ ] Следующее архитектурное изменение проверять относительно CNN-EXP-002 и
      менять только одну составляющую.
- [ ] Повторить champion на дополнительных seeds перед финальным выводом.
