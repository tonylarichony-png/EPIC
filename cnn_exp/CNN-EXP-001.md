---
id: CNN-EXP-001
type: experiment
experiment_type: cnn-baseline
status: completed
decision: adopt
date: 2026-09-21
seed: 42
split_version: contig_holdout_v1
run_name: baseline_v1
primary_metric: average_precision
secondary_metric: epic_spearman
reference_experiment: EXP-011
implementation: module-and-notebook
hardware: AMD Radeon RX 6750 GRE 12GB
backend: torch-directml
tags:
  - ml/experiment
  - ml/cnn
  - biology/transcription-initiation
---

# CNN-EXP-001 — компактная residual CNN baseline

← [[cnn_exp/_index.md|Реестр CNN]] · [[notebooks/experiments/CNN_experiments/CNN_plan.md|План]] · [[experiments/EXP-011.md|Logistic reference]]

> [!success] Итог
> Первая CNN успешно обучена и полностью провалидирована на AMD GPU без CPU
> fallback. Она существенно превзошла logistic champion одновременно по AP и
> EPIC Spearman и принимается как baseline для всей следующей CNN-серии.

## Гипотеза

Компактная свёрточная модель с полем восприятия 261 нт сможет выучить
позиционную грамматику вокруг TSS лучше моделей на вручную заданных k-mer,
GC/CpG и региональных динуклеотидных признаках.

Биологическая мотивация: точный выбор `+1` зависит не от одного мотива, а от
совместного расположения локального Inr-контекста, upstream-элементов,
downstream-элементов и состава ДНК. Окно ориентируется по предполагаемому
направлению транскрипции; для minus-цепи используется reverse complement.

## Данные и validation contract

| Поле | Значение |
|---|---:|
| Split | `contig_holdout_v1` |
| Train contigs | 9 |
| Validation contigs | 3 |
| Общие train/validation contigs | 0 |
| Train position-strand | 228 064 160 |
| Train positives | 170 443 |
| Validation position-strand | 83 807 486 |
| Validation positives | 56 205 |
| Test | не открывался |

Validation выполнена последовательным проходом по всем 47 743 tiles. Каждая
разрешённая position-strand учтена ровно один раз. AP не оценивался на
искусственно сбалансированной выборке.

## Представление входа

- Алфавит: пять каналов `A/C/G/T/N`, one-hot.
- Target tile: 1024 нт.
- Контекст: по 130 нт с каждой стороны.
- Полная длина входа: 1284 нт.
- Plus и reverse-complement minus проходят через одну модель с общими весами.
- Raw FASTA codes и маска валидной последовательности сохраняются генератором.

## Архитектура

| Слой                          | Каналы вход → выход | Ядро `k` | Dilation `d` | Padding с каждой стороны | RF после слоя, нт |
| ----------------------------- | ------------------: | -------: | -----------: | -----------------------: | ----------------: |
| Stem                          |              5 → 32 |        1 |            1 |                        0 |                 1 |
| Блок 1 — свёртка 1            |             32 → 32 |        7 |            1 |                        3 |                 7 |
| Блок 1 — свёртка 2 + residual |             32 → 32 |        7 |            1 |                        3 |                13 |
| Блок 2 — свёртка 1            |             32 → 32 |        3 |            2 |                        2 |                17 |
| Блок 2 — свёртка 2 + residual |             32 → 32 |        3 |            2 |                        2 |                21 |
| Блок 3 — свёртка 1            |             32 → 32 |        3 |            4 |                        4 |                29 |
| Блок 3 — свёртка 2 + residual |             32 → 32 |        3 |            4 |                        4 |                37 |
| Блок 4 — свёртка 1            |             32 → 32 |        3 |            8 |                        8 |                53 |
| Блок 4 — свёртка 2 + residual |             32 → 32 |        3 |            8 |                        8 |                69 |
| Блок 5 — свёртка 1            |             32 → 32 |        3 |           16 |                       16 |               101 |
| Блок 5 — свёртка 2 + residual |             32 → 32 |        3 |           16 |                       16 |               133 |
| Блок 6 — свёртка 1            |             32 → 32 |        3 |           32 |                       32 |               197 |
| Блок 6 — свёртка 2 + residual |             32 → 32 |        3 |           32 |                       32 |               261 |
| Голова профиля                |              32 → 1 |        1 |            1 |                        0 |               261 |
| Обрезка краёв выхода          |          2 → 2 цепи |        — |            — |   убираем по 130 позиций |               261 |

- Residual block: `BN → ReLU → Conv → BN → ReLU → Conv + skip`.
- Receptive field: **261 нт**.
- Обучаемых параметров: **46 433**.
- Padding сохраняет длину на каждом свёрточном слое. Края отбрасываются уже
  после модели: центральные 1024 позиции имеют полный контекст 130 нт слева и
  справа.

## Sampling и objective

- Неперекрывающиеся genomic tiles длиной 1024 нт.
- Половина train-батча выбирается из positive-ветви, половина из background-ветви.
- Tiles positive-ветви выбираются пропорционально числу positive positions.
- Tiles negative-ветви выбираются пропорционально числу negative positions.
- Importance weights возвращают objective к полной train population.
- Loss: class-balanced population BCE по разрешённым position-strand.
- Batch size: 4 genomic tiles, объединённый вход обеих цепей имеет размер 8.

## Обучение

| Параметр | Значение |
|---|---:|
| Optimizer | custom DirectML Adam |
| Learning rate | 0.001 |
| Betas | (0.9, 0.999) |
| Steps | 10 000 |
| Seed | 42 |
| Train time | 326.71 сек |
| Time per step | 0.0327 сек |
| Увидено position-strand | 76 146 258 |
| Финальный EMA loss | 0.415531 |
| CPU fallback | отсутствует |

Custom BCE и Adam использованы потому, что стандартные операции PyTorch 2.4.1
вызывали CPU fallback в `torch-directml`. Их численная эквивалентность
стандартным CPU-реализациям была проверена до запуска.

## Результаты

| Модель | Validation AP | EPIC Spearman |
|---|---:|---:|
| Случайный / constant reference | 0.000670644 | 0.000000000 |
| [[experiments/EXP-011.md\|EXP-011 logistic champion]] | 0.002558 | 0.112190 |
| **CNN-EXP-001** | **0.012360070** | **0.155747574** |

- AP вырос примерно в **4.83 раза** относительно EXP-011.
- AP превышает validation prevalence в **18.43 раза**.
- EPIC Spearman вырос на **0.043558** относительно EXP-011.
- Полный validation inference: **29.83 сек**, около **2.81 млн position-strand/с**.
- Validation inference также прошёл без CPU fallback.

### Результаты по validation-contig

| Segment | Position-strand | Positives | Prevalence | AP | AP lift | EPIC Spearman | Spearman после округления |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 83 807 486 | 56 205 | 0.000671 | **0.012360** | 18.43× | **0.155748** | 0.146725 |
| NC_064034.1 | 33 949 114 | 23 529 | 0.000693 | 0.013691 | 19.75× | 0.165281 | 0.161368 |
| NC_064041.1 | 24 146 754 | 13 984 | 0.000579 | 0.010646 | 18.38× | 0.194761 | 0.191160 |
| NC_064042.1 | 25 711 618 | 18 692 | 0.000727 | 0.012570 | 17.29× | 0.183528 | 0.179814 |

На всех трёх contigs AP существенно выше prevalence. Spearman внутри каждого
contig выше глобального Spearman. Это указывает на межконтинговый сдвиг шкалы
score/count: локальное ранжирование сильнее, чем единая глобальная калибровка.
Это наблюдение фиксируется как будущая гипотеза, но не исправляется постфактум
на validation.

### Округление score

EPIC Spearman по исходным float scores: **0.155747574**. Искусственное
округление вероятностей до пяти знаков снижает его до **0.146724983**, создавая
много ties среди малых вероятностей. Поэтому исходные prediction scores нельзя
округлять до сохранения submission. Значение с округлением сохраняется только
как диагностическое.

## Артефакты

- Notebook: [[notebooks/experiments/CNN_experiments/CNN_baseline.ipynb]]
- Model code: [[src/ml_project/epic_cnn/model.py]]
- Data generator: [[src/ml_project/epic_cnn/profile.py]]
- DirectML operations: [[src/ml_project/epic_cnn/directml.py]]
- Validation evaluator: [[src/ml_project/epic_cnn/evaluation.py]]
- Checkpoint: [[artifacts/experiments/CNN_baseline/baseline_v1/baseline_final.pt]]
- Training history: [[artifacts/experiments/CNN_baseline/baseline_v1/training_history.csv]]
- Validation metrics: [[artifacts/experiments/CNN_baseline/baseline_v1/validation_metrics.csv]]

## Ограничения

- Выполнен один seed; пока нет оценки разброса результатов.
- Loss обучает наличие сигнала, но не моделирует `count` отдельной intensity-head.
- Receptive field ограничен 261 нт и не видит дальний регуляторный контекст.
- Модель оценивает intrinsic promoter potential из последовательности, а не
  клеточно-специфическую активность, хроматин или 3D-контакты.

## Решение

`adopt` — принять CNN-EXP-001 как воспроизводимый CNN baseline. Не увеличивать
одновременно глубину, контекст и число голов: следующий эксперимент должен
проверять одну гипотезу относительно этой точки отсчёта.

## Следующие действия

- [x] Сохранить validation evaluator в модуле, чтобы следующие модели
      оценивались по одному и тому же протоколу.
- [x] Добавить per-contig AP и EPIC Spearman без использования test.
- [ ] Выбрать первый изолированный эксперимент: intensity/count head или
      увеличение поля восприятия.
- [ ] Повторять baseline с дополнительными seeds только перед окончательным
      сравнением близких кандидатов.
