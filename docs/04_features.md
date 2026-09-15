---
type: stage
stage: features
status: draft
owner:
last_reviewed:
tags:
  - ml/stage
  - ml/features
---

# 04 — Признаки

← [[docs/03_validation.md|Validation]] · [[README.md|Dashboard]] · Далее → [[docs/05_experiments.md|Experiments]]

> [!abstract] Результат этапа
> Понятно, какие признаки используются, откуда они берутся, в какой момент доступны, как преобразуются и одинаково ли вычисляются при train и inference.

## Model-ready выборка

Здесь принимаются окончательные решения о составе данных после [[docs/02_eda.md|EDA]]. Каждое удаление строки, фильтр или join должны иметь причину и выполняться кодом внутри воспроизводимого pipeline.

```text
DATA snapshot → фильтры / join → split по [[docs/03_validation.md]] → preprocessing внутри folds → model-ready matrices
```

<!-- auto:model-ready-contract:start -->

Исполняемый состав model-ready выборки появится после синхронизации baseline.

<!-- auto:model-ready-contract:end -->

### Ручные решения о составе строк

- **Правила join:**
- **Фильтры строк:**
- **Почему эти фильтры допустимы:**

| Решение | Основание из EDA | Реализация | Влияние на строки | Статус |
|---|---|---|---:|---|
|  |  |  |  | candidate / active / rejected |

## Feature strategy

- Какие аспекты поведения / объекта должны отражать признаки?
- Какие ограничения задают latency и доступность?
- Какие представления являются baseline?
- Какие группы признаков проверяются отдельно?

## Реестр признаков

<!-- auto:feature-registry:start -->

Реестр используемых и исключённых признаков появится после синхронизации baseline.

<!-- auto:feature-registry:end -->

## Preprocessing

Первый исполняемый вариант задаётся в `src/ml_project/baseline_config.py` и
строится [[notebooks/03_baseline.ipynb|baseline notebook]] как единый sklearn
Pipeline. Здесь хранится не копия Python-конфига, а **причины решений**,
ограничения и статус признаков.

<!-- auto:preprocessing-contract:start -->

Исполняемый preprocessing появится после синхронизации baseline.

<!-- auto:preprocessing-contract:end -->

### Обоснование числового preprocessing

- Почему выбран текущий imputer:
- Почему выбран текущий scaler:
- Какие выбросы требуют отдельного решения:

### Обоснование категориального preprocessing

- Почему выбран текущий encoder:
- Риски редких и новых категорий:
- Признаки высокой кардинальности:

### Время

- Timezone:
- Cyclic features:
- Rolling windows:
- Cutoff semantics:

### Текст / изображения / последовательности

- Pretrained representation:
- Tokenization / normalization:
- Версия модели / vocabulary:
- Ограничения inference:

## Генерация признаков

| Идея | Механизм | Связанная гипотеза | Эксперимент | Итог |
|---|---|---|---|---|
|  |  |  |  |  |

## Point-in-time correctness

Для каждого временного или агрегированного признака:

- [ ] Определён event time и processing time.
- [ ] Окно заканчивается не позже момента предсказания.
- [ ] Учтена фактическая задержка появления данных.
- [ ] Join не подтягивает будущие записи.
- [ ] Offline и online implementations эквивалентны.

## Feature selection и ablation

| Набор | Что добавлено / удалено | Эксперимент | Метрика | Δ | Решение |
|---|---|---|---:|---:|---|
|  |  |  |  |  |  |

## Отклонённые признаки

| Признак | Причина | Доказательство / эксперимент | Можно пересмотреть, если… |
|---|---|---|---|
|  | leakage / cost / no gain / unstable |  |  |

Отрицательные решения сохраняются, чтобы не повторять работу.

## Train-serving parity

| Проверка | Offline | Online | Допуск | Статус |
|---|---|---|---:|---|
| Schema |  |  |  |  |
| Null rate |  |  |  |  |
| Feature values |  |  |  |  |
| Category mapping |  |  |  |  |

## Стоимость

| Набор признаков | Время расчёта | Память | Latency | Зависимости | Комментарий |
|---|---:|---:|---:|---|---|
|  |  |  |  |  |  |

## Stage Gate: Features

- [ ] Состав model-ready выборки воспроизводим.
- [ ] Все фильтры и исключения обоснованы выводами EDA или требованиями качества.
- [ ] У активных признаков определены источник и формула.
- [ ] Проверена доступность каждого признака при inference.
- [ ] Preprocessing находится внутри воспроизводимого pipeline.
- [ ] Временные и агрегированные признаки point-in-time correct.
- [ ] Обработаны unknown categories и пропуски.
- [ ] Зафиксированы версии внешних представлений.
- [ ] Ключевые наборы прошли ablation.
- [ ] Отклонённые признаки имеют записанную причину.
- [ ] Проверяется train-serving parity.

> [!success] Следующий этап
> После выполнения Stage Gate можно переходить к [[docs/05_experiments.md|05 — Эксперименты]].
