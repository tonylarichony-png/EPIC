---
type: guide
tags:
  - ml/project-system
---

# Руководство по ML Project Template

## 1. Что это за система

Это не набор отчётов, а **ML Project OS** — связная система ведения ML-проекта в Obsidian. Она объединяет:

- текущее состояние проекта;
- документацию по этапам;
- проверяемые гипотезы;
- воспроизводимые эксперименты;
- принятые решения;
- проблемы, риски и production-контекст.

Главный UX-принцип: открыв [[README.md]], за 30 секунд должно быть понятно, **где находится проект и что делать дальше**.

Перед заполнением документов откройте [[PROJECT_CONFIG.md]] и включите только те условные разделы, которые действительно относятся к проекту.

## 2. Модель информации

```text
Project Dashboard ([[README.md]])
        │
        ├── Stage docs ([[docs/00_problem.md]] … [[docs/07_production.md]])
        │
        ├── Hypothesis H-xxx
        │       └── Experiment EXP-xxx
        │               ├── Result
        │               ├── Error analysis
        │               └── Decision DEC-xxx
        │
        └── Issue ISSUE-xxx ──→ действие / решение / новый эксперимент
```

### Единые источники истины

| Информация | Где хранить |
|---|---|
| Текущее состояние и следующий шаг | [[README.md]] |
| Видимость необязательных разделов | [[PROJECT_CONFIG.md]] |
| Формулировка задачи, основная метрика и критерий успеха | [[docs/00_problem.md]] |
| Исходные файлы данных | `data/raw/` |
| Общая конфигурация данных | `src/ml_project/config.py` |
| Переиспользуемый Python-код | `src/ml_project/` |
| Паспорт файлов и схемы | [[notebooks/01_data.ipynb]] |
| Источники, схема и версии raw-данных | [[docs/01_data.md]] |
| Первичный профиль и EDA | [[notebooks/02_eda.ipynb]] |
| Статистический обзор выбросов без изменения данных | [[notebooks/02_eda_anomalies.ipynb]] |
| Проверка EDA-гипотез и связей с target | [[notebooks/02_eda_hypotheses.ipynb]] |
| Выводы EDA и рекомендации по обработке | [[docs/02_eda.md]] |
| Реализация метрики, split и протокол проверки | [[docs/03_validation.md]] |
| Model-ready выборка, preprocessing и признаки | [[docs/04_features.md]] |
| Параметры первого baseline | `src/ml_project/baseline_config.py` |
| Общее ML-ядро baseline и экспериментов | `src/ml_project/modeling/` |
| Расчёт первого baseline | [[notebooks/03_baseline.ipynb]] |
| Контракт и реализация одного эксперимента | `src/ml_project/experiments/exp_xxx_*.py` |
| Выбранный для запуска эксперимент | `src/ml_project/experiment_config.py` |
| Создание следующего эксперимента | `.\new-experiment.cmd` |
| Контролируемое сравнение, графики и фиксация | [[notebooks/04_experiment.ipynb]] |
| Разбор пути признака, importance и OOF-ошибок | [[notebooks/05_diagnostics.ipynb]] |
| Групповой выбор семейства на лучшем feature set | [[notebooks/06_model_screening.ipynb]] |
| Группы и стартовые гиперпараметры | `src/ml_project/model_screening_config.py` |
| Full-train fit и Kaggle CSV выбранного EXP/MS-кандидата | [[notebooks/07_submission.ipynb]] |
| История отправленных файлов и Public score | [[submissions/_index.md]] |
| Синхронизация решений и Experiment ↔ EDA без переобучения | `sync-experiment-state.cmd` |
| Сводка результатов | [[docs/05_experiments.md]] |
| Системные ошибки модели | [[docs/06_error_analysis.md]] |
| Inference, мониторинг и retraining | [[docs/07_production.md]] |
| Проверяемая идея | отдельная заметка в [[hypotheses/_index.md|hypotheses]] |
| Один воспроизводимый запуск | отдельная заметка в [[experiments/_index.md|experiments]] |
| Важный выбор и его обоснование | отдельная заметка в [[decisions/_index.md|decisions]] |
| Блокер, риск или дефект | отдельная заметка в [[issues/_index.md|issues]] |
| Локальные модели, CV-таблицы и metadata | [[artifacts/_index.md]] |

Не дублируйте подробности: в dashboard и сводках оставляйте короткий итог и ссылку на источник.

> [!important] Граница этапов Data → EDA → Features
> В [[docs/01_data.md]] хранится общая информация о файлах, источниках, структуре и версиях. В [[docs/02_eda.md]] исследуются значения, target, пропуски, дубликаты, распределения и причины проблем. Окончательные фильтры, исключения и preprocessing реализуются и документируются в [[docs/04_features.md]] после фиксации [[docs/03_validation.md|validation-протокола]].

### Архитектура моделирования

`src/ml_project/modeling/` — общее исполняемое ядро для baseline и всех
контролируемых экспериментов:

| Модуль | Ответственность |
|---|---|
| `contracts.py` | Настройки и типизированные контракты данных, метрик и результатов |
| `settings.py` | Проверка и компактное отображение готового объекта `BASELINE` |
| `features.py` | Feature plan, model-ready данные и preprocessing |
| `validation.py` | Scorers, CV protocol и одинаковая оценка reference/candidate |
| `diagnostics.py` | Путь признака по pipeline, fitted-fold importance и OOF-ошибки candidate |
| `estimators.py` | Dummy/simple estimators и сборка sklearn Pipeline |
| `model_groups.py` | Реестр sklearn/external estimators и preprocessing-профили групп |
| `screening.py` | Frozen feature context, сборка одной группы и одинаковый CV-run |
| `screening_diagnostics.py` | OOF-сравнение и агрегированная native/coefficient importance |
| `screening_reporting.py` | Графики, локальные таблицы, MS-карточка и отдельный registry |
| `submission.py` | Каталог EXP/MS-кандидатов, inference audit, full-train fit, CSV и SUB-карточки |
| `artifacts.py` | CSV, metadata, snapshot окружения, модели и Git-отслеживаемые графики |
| `report_blocks.py` | Чистые генераторы Markdown-блоков для Validation, Features, README и реестров |
| `reporting.py` | Карточка baseline и оркестрация синхронизации всех отчётов |
| `report_audit.py` | Проверка auto-маркеров, локальных ссылок и согласованности решений |

`src/ml_project/baseline.py` остаётся тонким совместимым фасадом. Старые
импорты продолжают работать, но notebooks используют публичный API
`ml_project.modeling`.

Контролируемые эксперименты масштабируются отдельно от общего ядра:

| Модуль | Ответственность |
|---|---|
| `experiments/exp_xxx_*.py` | Неизменяемая pre-registration, явный `parent_experiment_module`, критерии успеха и ровно одно кандидатное изменение |
| `experiment_config.py` | Одна строка-селектор активного модуля; экспериментальная логика здесь не хранится |
| `experiment.py` | Загрузка контракта, одинаковая оценка, provenance, артефакты и синхронизация отчётов |
| `experiment_scaffold.py` | Создание следующего модуля и локального workbench |
| `experiment_workbench.py` | Генерация безопасной notebook-first лаборатории без записи официальных результатов |
| `notebooks/workbench/` | Игнорируемые Git черновики для разработки функции, smoke-fit и dry-run CV |
| `model_screening_config.py` | Один читаемый конфиг групп, starter-параметров и frozen feature reference |
| `model_screening.py` | Тонкий стабильный фасад screening API для notebook |

### Синхронизация notebook → docs

- [[notebooks/01_data.ipynb]] обновляет автоматические блоки в [[docs/01_data.md]].
- Перед синхронизацией `01_data` печатает готовую заготовку `FIELD_DESCRIPTIONS` для `src/ml_project/config.py`: найденные столбцы добавляются автоматически, но файл конфигурации не перезаписывается.
- [[notebooks/02_eda.ipynb]] обновляет фактический профиль в [[docs/02_eda.md]].
- [[notebooks/02_eda_anomalies.ipynb]] рассчитывает границы IQR и MAD только по train, применяет их к train/inference, показывает отмеченные признаки и строки и обновляет только автоматическую сводку раздела «Пропуски, выбросы и аномалии». Он ничего не удаляет и не исправляет.
- В `02_eda_hypotheses` сохранение выполняется только при явном `SAVE_FINDING = True`: карточка может содержать график, одну или несколько Markdown-таблиц либо оба типа артефактов. Большие таблицы полностью сохраняются как CSV в `assets/eda/`, а ссылка на карточку автоматически появляется в [[docs/02_eda.md#Сохранённые EDA-наблюдения]]. Идеи для модельной проверки переносите в [[hypotheses/_index.md]].
- [[notebooks/03_baseline.ipynb]] напрямую использует готовый объект `baseline_config.BASELINE`: скрытой сборки настроек через `modeling_tools` нет. Metric берётся из [[docs/00_problem.md]], а data contract — из `config.py`. При `BASELINE.sync_docs=True` notebook обновляет исполняемый validation contract, secondary metrics, baseline и воспроизводимость в [[docs/03_validation.md]], model-ready состав и preprocessing в [[docs/04_features.md]], current/best measured result в [[docs/05_experiments.md]], реестр карточек и ключевые результаты в [[README.md]]. CSV, metadata и final model сохраняются только отдельными явными полями `BASELINE`.
- [[notebooks/04_experiment.ipynb]] читает из `experiment_config.py` только имя активного модуля, а из самого модуля — типизированный объект `EXPERIMENT` и две функции: подготовку кандидатных данных и сборку моделей. `parent_experiment_module` явно связывает эксперимент с последним принятым champion; notebook не копируется между EXP-002, EXP-003 и следующими запусками.
- Локальный `notebooks/workbench/EXP-xxx_*.ipynb` используется до строгого runner: он загружает raw train без импорта незавершённого experiment-модуля, восстанавливает принятую родительскую pipeline, позволяет написать draft-функцию, проверяет contract, feature groups, smoke-fit reference/candidate и необязательный короткий CV. Fold-safe imputation остаётся внутри pipeline, поэтому пропуски могут быть видны в DataFrame до `fit`. Workbench ничего не синхронизирует.
- [[notebooks/04_experiment.ipynb]] автоматически применяет data-hooks всех принятых предков, пересчитывает champion reference и candidate на одинаковых folds и не использует старый CSV как модель. Metadata и карточка фиксируют цепочку модулей с SHA-256. Гипотеза, критерии и guardrails хранятся в модуле; изменяемое после интерпретации решение хранится во frontmatter карточки. Графики, таблицы, registry, leaderboard и ключевые результаты формируются автоматически.
- Тот же запуск сохраняет диагностику одного primary candidate: fitted-модели и validation-индексы тех же folds, transformed lineage, paired Δ, permutation/native importance и OOF-переходы ошибок. [[notebooks/05_diagnostics.ipynb]] только читает эти артефакты и ничего не переобучает.
- [[notebooks/06_model_screening.ipynb]] восстанавливает принятый feature champion из versioned experiment-модуля, оставляет validation contract неизменным и запускает только выбранную группу. Параметры заранее видны в `model_screening_config.py`; fitted folds сразу дают ranking, paired wins/losses, OOF-сравнение и importance. Финальная ячейка пишет локальные таблицы в `artifacts/model-screening/`, Git-tracked PNG в `assets/model-screening/<MS-ID>/`, отдельную карточку и [[model-screening/_index.md|реестр]].
- [[notebooks/07_submission.ipynb]] показывает единый каталог baseline/EXP/MS-кандидатов и выбирает один устойчивым Candidate ID прямо в notebook. Перед full-train fit он восстанавливает точный pipeline, сверяет source hash и проверяет, что raw feature hook не зависит от состава окружающего DataFrame. CSV, fitted pipeline и metadata остаются локально в `artifacts/submissions/<SUB-ID>/`; tracked-карточка и Public score ведутся в [[submissions/_index.md]].
- EDA-основания добавляются после запуска во frontmatter experiment-карточки: `eda_findings: ["EDA-003"]`. Команда `sync-experiment-state.cmd` проверяет существование ID, обновляет таблицы оснований и одновременно синхронизирует решение. Старое имя `sync-experiment-links.cmd` оставлено как совместимый alias. Связи и решение не меняют hash исполняемого Python-модуля.
- Metadata и реестр фиксируют полный hash данных, путь и hash Python-модуля эксперимента, а metadata дополнительно фиксирует hash baseline-конфигурации. Это связывает измеренный результат не только с параметрами, но и с реально исполненным кодом.
- Решение (`pending`, `adopt`, `reject`, `iterate`, `inconclusive`) меняется только в поле `decision:` frontmatter experiment-карточки. После `sync-experiment-state.cmd` оно попадает в generated-отчёт, все CSV registry и сводные документы без повторного обучения. Python-модуль после официального запуска для этого не редактируется.
- Финальная ячейка modeling notebook запускает проверку целостности: она сообщает об отсутствующих `assets/artifacts`, повреждённых auto-маркерах и несовпадающих experiment decisions. Проверка ничего не исправляет сама.
- Перед EDA распределите каждый столбец train, кроме target, ровно в одну группу `FEATURE_GROUPS`: `numeric`, `count`, `categorical`, `ordinal`, `text`, `datetime`, `identifier` или `ignored`. Числовой отчёт использует `numeric + count`, категориальный — `categorical + ordinal`.
- После редактирования `src/ml_project/config.py` финальная ячейка `01_data` сама перечитывает конфиг через `importlib.reload`, пересоздаёт каталог и обновляет Markdown — перезапуск kernel не требуется.
- Запись выполняется только явной финальной ячейкой notebook.
- Код изменяет только содержимое между маркерами `<!-- auto:...:start -->` и `<!-- auto:...:end -->`.
- Ручные выводы храните вне автоматических блоков: при повторном запуске они не перезаписываются.

## 3. Два режима использования

### Конфигурация разделов

В [[PROJECT_CONFIG.md]] находятся обычные Markdown-галочки. Они управляют необязательными секциями:

- снятая галочка убирает секцию из рабочего документа и сохраняет её текст в `sections`;
- установленная галочка возвращает секцию как обычный редактируемый Markdown;
- изменение применяется, когда целевой документ открыт или открывается в Live Preview / режиме чтения.

Механизм использует DataviewJS и общий скрипт `system/conditional-section/view.js`. Не редактируйте технические HTML-маркеры условной секции вручную.

### Минимальный режим

Подходит для pet-проекта или быстрого прототипа:

1. Заполнить Problem, Data и Validation.
2. Настроить `baseline_config.py`, запустить [[notebooks/03_baseline.ipynb]] и зафиксировать baseline.
3. Создавать заметку только для значимых экспериментов.
4. Обновлять README и таблицу лучших результатов.
5. Перед завершением выполнить Error analysis.

### Полный режим

Подходит для долгого, командного или production-проекта:

- каждая экспериментальная ветка начинается с гипотезы;
- версии данных и кода фиксируются;
- важные выборы оформляются как решения;
- проблемы получают владельца и срок;
- переходы между этапами проходят через Stage Gate;
- production включает rollout, мониторинг и rollback.

## 4. Жизненный цикл

```text
Problem → Data → EDA → Validation → Baseline
                            ↓
                     Hypotheses → Experiments
                            ↓             │
                      Error Analysis ←────┘
                            ↓
                       Production
                            ↓
                 Monitoring / новые гипотезы
```

Нумерация документов задаёт навигацию, но процесс не обязан быть линейным. Главное правило: **Validation фиксируется до массового сравнения моделей**.

## 5. Stage Gates

В конце каждого документа есть контрольный список. Этап можно отметить завершённым в [[README.md]], когда:

- заполнены обязательные разделы;
- открытые вопросы либо закрыты, либо оформлены как ISSUE;
- важные решения зафиксированы;
- понятен следующий этап и его владелец.

Stage Gate — не бюрократия, а защита от дорогих ошибок: leakage, неверной метрики, несогласованного target и невоспроизводимых результатов.

## 6. Сущности и статусы

### Проект

`idea → active → paused → completed → archived`

Этапы: `problem`, `data`, `eda`, `validation`, `features`, `experiments`, `error-analysis`, `production`.

### Гипотеза

`backlog → planned → testing → validated / rejected / parked`

Гипотеза описывает **ожидаемое причинное изменение**, а не действие. Хорошо: «агрегации за 30 дней повысят PR-AUC, потому что отражают стабильное поведение». Плохо: «попробовать CatBoost».

### Эксперимент

`planned → running → completed / failed / cancelled`

Одна заметка — один сравнимый запуск или одна строго определённая серия запусков.

### Решение

`proposed → accepted / rejected → superseded`

Решение хранит не только выбор, но контекст, альтернативы и последствия.

### Проблема

`open → investigating → blocked → resolved / wont-fix`

## 7. Имена и идентификаторы

Рекомендуемый формат:

```text
hypotheses/H-001 Короткое название.md
experiments/EXP-001 Короткое название.md
decisions/DEC-001 Короткое название.md
issues/ISSUE-001 Короткое название.md
```

Правила:

- ID никогда не переиспользуется;
- название описывает смысл, а не дату;
- ID можно использовать в коде, commit-сообщениях и MLflow tags;
- переименование безопасно делать средствами Obsidian, чтобы обновились Wiki-ссылки.

## 8. Рабочие сценарии

### Первый baseline

1. Завершить EDA-рекомендации и зафиксировать [[docs/03_validation.md|validation-протокол]].
2. В `src/ml_project/config.py` заполнить target, key и `FEATURE_GROUPS`.
3. В объекте `BASELINE` файла `src/ml_project/baseline_config.py` выбрать тип задачи, CV, preprocessing и действия записи.
4. Запустить [[notebooks/03_baseline.ipynb]] сверху вниз и сравнить `dummy` с простой моделью.
5. После проверки результата включить нужные write-флаги, создать карточку `EXP-xxx Baseline` и обновить [[README.md]].

Не заполняйте пропуски и не кодируйте категории заранее на полном train: эти операции должны находиться внутри model pipeline и обучаться отдельно в каждом fold.

### Новая гипотеза

1. Создать заметку в `hypotheses`.
2. Выполнить **Templates: Insert template** → `hypothesis`.
3. Назначить следующий свободный `H-xxx`.
4. Указать механизм, ожидаемый эффект и критерий успеха.
5. Добавить ссылку в активную работу [[README.md]].

### Новый эксперимент

1. Для первого выбранного примера выполнить `.\new-experiment.cmd --workbench-only`. Для следующих запустить `.\new-experiment.cmd`: launcher создаст модуль и workbench, а модуль последней карточки с `decision: adopt` запишет как явный parent. Для независимой проверки используйте `--from-baseline`.
2. В workbench написать draft-преобразование, обновить draft feature groups, выполнить contract и smoke-fit. Reference уже содержит все принятые родительские изменения; при необходимости включите несохраняемый `RUN_DRY_CV`.
3. Перенести проверенную реализацию из workbench в две функции модуля: `prepare_candidate_data(...)` и `build_candidate_models(...)`. Все обучаемые преобразования должны оставаться внутри sklearn Pipeline, чтобы обучаться отдельно на каждом fold.
4. В модуле заполнить `EXPERIMENT`: гипотезу, единственное изменение, ожидаемый эффект, `primary_improvement_min` и при необходимости `metric_guardrails`; затем включить `RUN_MODULE_SMOKE` в workbench.
5. Только после успешного module smoke перезапустить kernel и выполнить строгий [[notebooks/04_experiment.ipynb]] сверху вниз. Он создаст карточку, графики, таблицы, metadata и обновит сводные документы.
6. Интерпретировать результат и изменить только поле `decision:` во frontmatter Markdown-карточки на `adopt`, `reject`, `iterate` или `inconclusive`; затем запустить `.\sync-experiment-state.cmd`. Только `adopt` делает модуль возможным родителем следующего experiment.
7. Следующий launcher автоматически выберет последний adopted-модуль и запишет его точное имя в новый контракт. Versioned Python-модуль после запуска не переписывать; workbench можно оставить локально или удалить после переноса кода.

В коде эксперимента используются три явные роли настроек:

- `initial_settings` — исходный `ModelingSettings` из EXP-001; он фиксирует общий schema/metrics/CV contract;
- `reference_settings` — настройки baseline или принятого чемпиона после применения всей parent-цепочки;
- `candidate_settings` — копия reference после единственного контролируемого изменения текущего эксперимента.

`ExperimentSettings` отвечает за другую часть контракта: гипотезу, критерии,
parent, решение и provenance. Старое имя типа `BaselineSettings` сохранено
только как совместимый alias; новый код должен использовать `ModelingSettings`.

Если эксперимент меняет только гиперпараметр или estimator, функция подготовки
может вернуть входные данные без изменений. Если меняются признаки, обучаемое
заполнение пропусков, encoding или selection, их реализация должна быть частью
candidate Pipeline, а не заранее преобразованного полного train.

### Групповой screening моделей

Начинайте этот этап, когда основные EDA/feature-гипотезы проверены и дальнейшие
улучшения на исходной модели вышли на плато.

1. В `model_screening_config.py` зафиксировать `feature_reference_module`
   последнего принятого feature champion. Это замораживает всю parent-цепочку.
2. Выбрать одну `active_group`: `classic_scaled`, `tree_bagging`,
   `sklearn_boosting`, `external_boosting` или `native_categorical`.
3. До запуска прочитать и при необходимости изменить `params` каждой включённой
   модели. Внутри одного `MS-ID` параметры больше не менять.
4. Выполнить [[notebooks/06_model_screening.ipynb]] сверху вниз. Reference и
   кандидаты оцениваются на одинаковых folds; положительный paired Δ всегда
   означает улучшение независимо от направления метрики.
5. В карточке объяснить устойчивость по folds, OOF-исправления и новые ошибки,
   затем отметить модели, которые переходят в shortlist.
6. Для следующей группы создать новый `MS-ID`, note и `run_name`. Feature
   reference, метрики и CV оставить прежними.
7. После покрытия групп выбрать 2–3 разных семейства и только для них провести
   coarse tuning. Финальный tuning выполняется позже, после точечных feature
   ablation/retest на shortlist-моделях.

Технический повтор того же запуска идемпотентен. Если изменились данные,
feature reference, группа или параметры, сохранение под прежними `MS-ID` и
`run_name` остановится и потребует новую идентичность запуска.

`Shortlist` означает «стоит исследовать дальше», а не «новый champion». Среднее
CV без параметров, std и paired-fold поведения недостаточно для окончательного
выбора модели. PyTorch/DNN остаётся отдельной будущей веткой и в этот runner не
включён.

### Kaggle submission

1. Выполнить каталог в [[notebooks/07_submission.ipynb]].
2. В единственной редактируемой ячейке указать новый `SUB-ID` и Candidate ID.
3. Прочитать восстановленный контракт и inference audit до fit.
4. Обучить pipeline на полном train и проверить sample-aligned output.
5. Явной финальной ячейкой сохранить CSV, model artifact, metadata и SUB-карточку.
6. После ручной загрузки заполнить Public score в карточке.

Для нового кандидата нужен новый `SUB-ID`. Повтор того же контракта идемпотентен,
но смена source, данных, feature module или estimator под существующим ID
блокируется. Reject-кандидат разрешён: решение эксперимента показывается как
контекст, а техническая возможность inference определяется отдельным audit.

### Важное решение

Создавайте DEC, если выбор:

- влияет на несколько экспериментов или этапов;
- меняет target, метрику, split или архитектуру;
- трудно или дорого отменить;
- требует объяснения через несколько недель.

### Работа с блокером

Оформляйте ISSUE, если проблема требует исследования, владельца или отслеживания. Мелкую задачу достаточно оставить чекбоксом рядом с контекстом.

## 9. Ритм обслуживания

### Перед рабочей сессией — 1 минута

- открыть [[README.md]];
- проверить текущую цель;
- выбрать одно следующее действие.

### После эксперимента — 5 минут

- записать результат и артефакты;
- сформулировать вывод;
- принять или запланировать решение;
- обновить текущий лучший результат.

### Раз в неделю — 15 минут

- закрыть устаревшие задачи;
- проверить активные гипотезы и ISSUE;
- обновить stage и риски;
- убедиться, что README отражает реальность.

### Перед переходом этапа

- выполнить Stage Gate;
- зафиксировать спорные решения;
- записать следующий этап и владельца.

## 10. Obsidian-native возможности

Система работает на стандартных функциях Obsidian:

- Properties/YAML — машинно-читаемый статус;
- Wiki-ссылки — связи между сущностями;
- Backlinks — история влияния решения или гипотезы;
- Templates — быстрое создание однотипных записей;
- Search query blocks — автоматический список файлов в реестрах;
- Graph и Local Graph — исследование связей.
- DataviewJS — управление условными секциями из [[PROJECT_CONFIG.md]].

Папка `templates` настроена для встроенного плагина Templates. Новые вложения автоматически сохраняются в `assets`, а Obsidian обновляет Wiki-ссылки при переименовании файлов.

Для условных секций требуется включённый плагин **Dataview** с настройкой **Enable JavaScript queries**. Он уже включён в конфигурацию этого vault-шаблона. Templater и Tasks можно добавить позже, но базовая система от них не зависит.

## 11. Правила качества

1. У каждого эксперимента есть гипотеза или явно указан тип `exploration`.
2. Validation не меняется незаметно между экспериментами.
3. Результат без версии данных и кода считается невоспроизводимым.
4. Отрицательный результат сохраняется — он предотвращает повтор работы.
5. Графики без вывода не считаются EDA.
6. Решение без альтернатив и причины быстро теряет ценность.
7. README показывает настоящее состояние, а не желаемое.
8. Документы остаются краткими: детали выносятся в атомарные заметки и артефакты.

## 12. Проверка готовности нового проекта

- [ ] Выбраны нужные разделы в [[PROJECT_CONFIG.md]].
- [ ] Переименован заголовок в [[README.md]].
- [ ] Заполнены owner, status, stage и primary metric.
- [ ] Определены target, объект и момент предсказания.
- [ ] Зафиксирована версия исходных данных.
- [ ] Выбрана стратегия split и holdout.
- [ ] Заполнен объект `BASELINE` в `src/ml_project/baseline_config.py`.
- [ ] Посчитан простой baseline.
- [ ] Создана первая гипотеза.
- [ ] Создан первый воспроизводимый эксперимент.
- [ ] Определено место для кода и артефактов.
