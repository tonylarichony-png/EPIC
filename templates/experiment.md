---
id: EXP-xxx
type: experiment
experiment_type: hypothesis-test
status: planned
owner:
created: "{{date}}"
started:
completed:
hypothesis:
dataset_version:
code_version:
model:
seed:
primary_metric:
result:
baseline_delta:
decision: pending
eda_findings: []
tags:
  - ml/experiment
---

# {{title}}

← [[experiments/_index.md|Реестр экспериментов]] · [[docs/05_experiments.md|Сводка]] · [[README.md|Dashboard]]

> [!abstract] Цель
> Одним предложением: какую гипотезу проверяет запуск и что считается успехом.

## Pre-registration — заполнить до запуска

### Связи

- **Гипотеза:**
- **Validation:** [[docs/03_validation.md]]
- **Связанные решения:**
- **Предыдущий сравнимый эксперимент:**

### Изменение

- **Что меняем:**
- **Что остаётся фиксированным:**
- **Почему ожидаем эффект:**
- **Критерий успеха:**
- **Guardrails:**
- **Бюджет / stop condition:**

## Setup

### Данные

| Поле | Значение |
|---|---|
| Dataset version |  |
| Период |  |
| Train / validation / test |  |
| Sampling / filters |  |
| Target version |  |

### Features

- **Feature set / version:**
- **Preprocessing:**
- **Изменения к baseline:**

### Model

- **Алгоритм / архитектура:**
- **Ключевые параметры:**
- **Initialization / pretrained version:**
- **Seed / seeds:**

### Воспроизводимость

| Поле | Значение |
|---|---|
| Repository |  |
| Commit / branch |  |
| Config |  |
| Command |  |
| Environment / image |  |
| Hardware |  |
| Tracking run |  |

```shell
# Команда воспроизведения
```

## Результаты

### Основные метрики

| Split / fold | Metric | Baseline | Result | Δ | Notes |
|---|---|---:|---:|---:|---|
|  |  |  |  |  |  |

### Guardrails и сегменты

| Сегмент / guardrail | Baseline | Result | Δ | Pass? |
|---|---:|---:|---:|---|
|  |  |  |  |  |

### Стабильность

- **Mean ± std:**
- **Confidence interval:**
- **Разброс по seeds / folds:**
- **Статистическая / практическая значимость:**

### Ресурсы

| Ресурс | Baseline | Result | Δ |
|---|---:|---:|---:|
| Train time |  |  |  |
| Inference latency |  |  |  |
| Memory |  |  |  |
| Cost |  |  |  |

## Артефакты

- **Model artifact:**
- **Predictions / OOF:**
- **Logs:**
- **Графики:**
- **Notebook / report:**

Файлы можно хранить в [[assets/_index.md|assets]].

## Анализ результата

- **Что произошло:**
- **Подтвердилась ли гипотеза:**
- **Почему мог получиться такой результат:**
- **Неожиданные наблюдения:**
- **Ошибки / ограничения эксперимента:**
- **Сравним ли результат с leaderboard:** yes / no, почему.

## Решение

- **Машинное решение:** измените `decision:` во frontmatter и запустите `sync-experiment-state.cmd`.
- **Outcome:** adopt / reject / iterate / inconclusive.
- **Что меняем в проекте:**
- **Нужно ли отдельное DEC:**
- **Что обновить:** README / features / validation / production / другое.

## Следующие действия

- [ ]
- [ ]

## Completion checklist

- [ ] Заполнены dataset и code versions.
- [ ] Зафиксированы config, command и seed.
- [ ] Результат сравнивается по согласованному протоколу.
- [ ] Проверены guardrails и стоимость.
- [ ] Сохранены необходимые артефакты.
- [ ] Написан вывод, а не только таблица метрик.
- [ ] Принято решение и обновлена [[docs/05_experiments.md]].
