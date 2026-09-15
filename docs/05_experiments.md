---
type: stage
stage: experiments
status: draft
owner:
last_reviewed:
tags:
  - ml/stage
  - ml/experiments
---

# 05 — Эксперименты

← [[docs/04_features.md|Features]] · [[README.md|Dashboard]] · Далее → [[docs/06_error_analysis.md|Error analysis]]

> [!abstract] Результат этапа
> Сравнимый журнал проверок: что изменили, почему ожидали эффект, при каких условиях запускали, что получили и какое решение приняли.

> [!tip] Следующий эксперимент
> Начните с [[README.md#Как начать новый эксперимент|короткой инструкции в карточке проекта]]. Подробный процесс описан в [[GUIDE.md#Новый эксперимент|руководстве]].

> [!tip] Выбор семейства модели
> После стабилизации feature set используйте [[notebooks/06_model_screening.ipynb|групповой screening]]. Его карточки и общий список находятся в [[model-screening/_index.md]].

## Правила экспериментов

1. Один эксперимент проверяет одну основную гипотезу.
2. Validation следует [[docs/03_validation.md]].
3. До запуска фиксируются setup и критерий успеха.
4. После запуска обязательно записываются вывод и решение.
5. Отрицательные и неудачные результаты не удаляются.
6. При несравнимом протоколе это явно указывается.

## Текущий baseline

<!-- auto:current-baseline:start -->

| Поле | Значение |
|---|---|
| Эксперимент / версия |  |
| Данные |  |
| Модель |  |
| Основная метрика |  |
| Значение |  |
| Стоимость / latency |  |

<!-- auto:current-baseline:end -->

Блок может быть синхронизирован из [[notebooks/03_baseline.ipynb]] после
проверки результата. Полный контекст запуска всё равно фиксируется отдельной
заметкой в [[experiments/_index.md]].

## Последний зафиксированный эксперимент

<!-- auto:latest-experiment:start -->

| Поле | Значение |
|---|---|
| Эксперимент |  |
| Гипотеза |  |
| Изменение |  |
| Метрика |  |
| Reference |  |
| Кандидат |  |
| Δ к reference |  |
| Решение |  |

<!-- auto:latest-experiment:end -->

Блок обновляется из [[notebooks/04_experiment.ipynb]]. Полный автоматический отчёт и ручной вывод хранятся в карточке эксперимента.

## Лучший измеренный результат

<!-- auto:best-measured-result:start -->

Блок появится после синхронизации baseline или контролируемого эксперимента.

<!-- auto:best-measured-result:end -->

## Leaderboard

<!-- auto:experiment-leaderboard:start -->

| Experiment | Hypothesis | Change | Metric | Reference | Result | Δ | Decision |
|---|---|---|---|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |

<!-- auto:experiment-leaderboard:end -->

> [!tip]
> Leaderboard содержит только сопоставимые результаты. Полный протокол хранится в отдельных заметках [[experiments/_index.md]].

## Screening семейств моделей

Feature-эксперименты отвечают на вопрос «какой сигнал добавить», а групповой
screening — «какое семейство лучше использует уже выбранный сигнал». Поэтому
MS-запуски ведутся в отдельном [[model-screening/_index.md|реестре]] и не
смешиваются с контролируемыми EXP-проверками одного изменения.

Одинаковые folds, primary/secondary metrics и точный feature champion позволяют
смотреть не только на mean, но и на std, paired wins/losses, OOF-переходы ошибок
и стоимость fit. Лучшие стартовые конфигурации переходят в shortlist, затем в
coarse tuning; новый champion на этапе screening автоматически не назначается.

### Последний сохранённый screening

<!-- auto:latest-model-screening:start -->

Блок появится после сохранения первого группового screening.

<!-- auto:latest-model-screening:end -->

### Сводка screening

<!-- auto:model-screening-summary:start -->

| Screening | Feature set | Группа | Лидер | Метрика | Reference | Лучший | Δ | Shortlist |
|---|---|---|---|---|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |  |

<!-- auto:model-screening-summary:end -->

> [!tip]
> Здесь одна строка соответствует одному MS-запуску. Подробные результаты всех
> моделей, paired folds и параметры остаются в [[model-screening/_index.md|реестре screening]].

## Очередь

| Priority | Hypothesis | Expected impact | Effort | Status |
|---|---|---|---|---|
|  |  |  |  |  |

## Все эксперименты

```query
path:"experiments" -file:"_index"
```

Создание: новая заметка `EXP-xxx Короткое название` → [[templates/experiment.md|шаблон experiment]].

## Правила повышения baseline

Новый baseline принимается, если:

- улучшение превышает минимально значимый эффект;
- результат стабилен по folds / seeds;
- guardrail-метрики не ухудшились недопустимо;
- нет нового leakage;
- стоимость соответствует ограничениям;
- оформлено решение, если изменение существенно.

## Несравнимые изменения

Если изменилась версия данных, target, split или метрика:

1. создайте [[templates/decision.md|DEC]];
2. начните новую секцию leaderboard или новую baseline-линейку;
3. не сравнивайте значения напрямую без пересчёта.

## Stage Gate: Experiments

- [ ] Есть воспроизводимый baseline.
- [ ] Каждый значимый эксперимент имеет ID и заметку.
- [ ] Эксперименты связаны с гипотезами или помечены exploration.
- [ ] Зафиксированы data version, code version, seed и environment.
- [ ] Leaderboard содержит только сопоставимые результаты.
- [ ] У каждого завершённого эксперимента есть вывод.
- [ ] Лучший кандидат проверен на стабильность и guardrails.
- [ ] Отрицательные результаты сохранены.
- [ ] Выбран кандидат для [[docs/06_error_analysis.md]].

> [!success] Следующий этап
> После выполнения Stage Gate можно переходить к [[docs/06_error_analysis.md|06 — Анализ ошибок]].
