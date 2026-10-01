---
type: stage
stage: experiments
status: draft
owner:
last_reviewed: 2026-09-18
tags:
  - ml/stage
  - ml/experiments
---

# 05 — Эксперименты

← [[docs/04_features.md|Features]] · [[README.md|Dashboard]] · Далее → [[docs/06_error_analysis.md|Error analysis]]

> [!abstract] Результат этапа
> Сравнимый журнал проверок: что изменили, почему ожидали эффект, при каких условиях запускали, что получили и какое решение приняли.

> [!tip] Следующий эксперимент
> Начните с [[README.md#Как начать новый эксперимент|короткой инструкции в карточке проекта]]. Геномный notebook-first процесс и общий data contract описаны в [[docs/epic_experiment_workflow.md]].

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
| Эксперимент / версия | [[experiments/EXP-001.md|EXP-001]], `run_001` |
| Данные | `jaNemVect1.1`, `contig_holdout_v1`, полный train/validation |
| Модель | L2 logistic regression; ориентированный динуклеотид `[-1,0]`; 17 категорий |
| Обязательные метрики | Average Precision; EPIC Spearman |
| Значение | AP **0.001330258**; EPIC Spearman **0.090720** |
| Стоимость / latency | 34 агрегированные solver-строки; fit ≈ 0.09 s после полного подсчёта категорий |

<!-- auto:current-baseline:end -->

Полный контекст запуска зафиксирован в [[experiments/EXP-001.md]] и выполненном
[[notebooks/experiments/EXP-001_dinucleotide_baseline.ipynb|notebook]].

## Последний завершённый validation-эксперимент

<!-- auto:latest-experiment:start -->

| Поле | Значение |
|---|---|
| Эксперимент | [[experiments/EXP-009.md|EXP-009 — fixed CpG O/E201 bins]] |
| Гипотеза | Региональный CpG O/E содержит сигнал сверх GC201-bin |
| Изменение | Fixed CpG O/E201 bins поверх EXP-007 |
| Average Precision: reference / candidate / relative Δ | 0.002053635 / **0.002363735** / +15.10% |
| EPIC Spearman: reference / candidate / Δ | 0.104302 / **0.115517** / +0.011215 |
| Решение | `adopt`: текущий подтверждённый champion |

<!-- auto:latest-experiment:end -->

Для новых геномных экспериментов машинная запись сохраняется последней ячейкой
отдельного `notebooks/experiments/EXP-xxx_*.ipynb`, а краткий человеческий вывод
хранится в карточке с тем же `EXP-ID`.

## Лучший измеренный результат

<!-- auto:best-measured-result:start -->

[[experiments/EXP-009.md|EXP-009]]: hierarchical `2/4/6-mer` LR с fixed
GC201-bins и CpG O/E201-bins, validation AP **0.002363735**, EPIC Spearman **0.115517**.
[[experiments/EXP-003.md|EXP-003]] остаётся историческим champion lookup-серии,
а [[experiments/EXP-001.md|EXP-001]] — исходным baseline.

<!-- auto:best-measured-result:end -->

## Leaderboard

<!-- auto:experiment-leaderboard:start -->

| Experiment | Hypothesis | Change | Metric | Reference | Result | Δ | Decision |
|---|---|---|---|---:|---:|---:|---|
| [[experiments/EXP-001.md|EXP-001]] | Пара `[-1,0]` содержит сигнал | Constant → dinucleotide LR | Average Precision | 0.000670644 | **0.001330258** | **+0.000659614** | `adopt` |
| [[experiments/EXP-001.md|EXP-001]] | Пара `[-1,0]` содержит сигнал | Constant → dinucleotide LR | EPIC Spearman | 0.000000 | **0.090720** | **+0.090720** | `adopt` |
| [[experiments/EXP-002.md|EXP-002]] | 4-mer содержит сигнал сверх пары | 2-mer → 4-mer lookup | Average Precision | 0.001330258 | **0.001486787** | **+11.77%** | `adopt` |
| [[experiments/EXP-002.md|EXP-002]] | 4-mer содержит сигнал сверх пары | 2-mer → 4-mer lookup | EPIC Spearman | 0.090720 | **0.094232** | **+0.003512** | `adopt` |
| [[experiments/EXP-003.md|EXP-003]] | 6-mer содержит сигнал сверх 4-mer | 4-mer → 6-mer lookup | Average Precision | 0.001486787 | **0.001600537** | **+7.65%** | `adopt` |
| [[experiments/EXP-003.md|EXP-003]] | 6-mer содержит сигнал сверх 4-mer | 4-mer → 6-mer lookup | EPIC Spearman | 0.094232 | **0.097186** | **+0.002954** | `adopt` |
| [[experiments/EXP-004.md|EXP-004]] | 8-mer содержит сигнал сверх 6-mer | 6-mer → 8-mer lookup | Average Precision | 0.001600537 | 0.001603481 | +0.18% | `reject` |
| [[experiments/EXP-004.md|EXP-004]] | 8-mer содержит сигнал сверх 6-mer | 6-mer → 8-mer lookup | EPIC Spearman | **0.097186** | 0.092203 | −0.004983 | `reject` |
| [[experiments/EXP-005.md|EXP-005]] | LR обучит hierarchical backoff лучше lookup | 6-mer lookup → 2/4/6-mer LR | Average Precision | 0.001600537 | 0.001606098 | +0.35% | `reject` |
| [[experiments/EXP-006.md|EXP-006]] | Linear GC201 добавит региональный сигнал | EXP-005 + linear GC201 | Average Precision | **0.001606098** | 0.001573742 | −2.02% | `reject` |
| [[experiments/EXP-007.md|EXP-007]] | GC201 имеет немонотонную форму | Linear GC201 → fixed bins | Average Precision | 0.001600537 | **0.002053635** | **+28.31%** | `adopt` |
| [[experiments/EXP-007.md|EXP-007]] | GC201 имеет немонотонную форму | Linear GC201 → fixed bins | EPIC Spearman | 0.097186 | **0.104302** | **+0.007117** | `adopt` |
| [[experiments/EXP-008.md|EXP-008]] | Эффект k2 зависит от GC201-bin | EXP-007 + 128 interactions | Average Precision | 0.002053635 | 0.002064290 | +0.519% | `reject` |
| [[experiments/EXP-008.md|EXP-008]] | Эффект k2 зависит от GC201-bin | EXP-007 + 128 interactions | EPIC Spearman | **0.104302** | 0.104167 | −0.000135 | `reject` |
| [[experiments/EXP-009.md|EXP-009]] | CpG O/E201 несёт сигнал сверх GC | EXP-007 + fixed CpG bins | Average Precision | 0.002053635 | **0.002363735** | +15.10% | `adopt` |
| [[experiments/EXP-009.md|EXP-009]] | CpG O/E201 несёт сигнал сверх GC | EXP-007 + fixed CpG bins | EPIC Spearman | 0.104302 | **0.115517** | +0.011215 | `adopt` |

<!-- auto:experiment-leaderboard:end -->

> [!tip]
> Leaderboard содержит только сопоставимые результаты. Полный протокол хранится в отдельных заметках [[experiments/_index.md]].

## Текущий feature-screening: EXP-010

[[experiments/EXP-010.md|EXP-010]] завершил AP-screening 15 региональных пар: 45 fit сошлись. Mean train-CV AP reference EXP-009 — 0.002672437559; лучшие отдельные добавки: GG 0.002791222 (+4.445%), TA 0.002777185 (+3.920%), TG 0.002748877 (+2.860%). Полная таблица и интерпретация: [[eda/findings/EDA-005.md|EDA-005]].

Это **train-CV**, не validation-метрики из leaderboard. Полного Spearman-screening нет; gate и решение остаются pending. Предложение следующей проверки: [[experiment-ideas/IDEA-006_gg_ta_complementarity.md|EXP-011 — совместная польза GG и TA]].

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
| 1 | Другие региональные пары добавляют сигнал сверх EXP-009 | Завершить Spearman-gate, выбор и validation одного кандидата | medium | EXP-010: AP-screening завершён, полный gate pending |
| 2 | GG и TA дополняют друг друга | Сравнить +GG+TA с обоими одиночными кандидатами | medium | IDEA-006: предложение EXP-011, не запускался |

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
