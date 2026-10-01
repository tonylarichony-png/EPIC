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

Общая подготовка интервалов и target зафиксирована в
`src/ml_project/epic_data.py`. Единая ориентация локальных окон по цепи и чтение
FASTA находятся в `src/ml_project/sequence_context.py`: `strand` сохраняется в
ключе и metadata, но не поступает в матрицу признаков. Детерминированное sparse
кодирование ориентированных оснований находится в
`src/ml_project/sequence_features.py`; конкретный эксперимент только выбирает
encoder и offsets. Sampling и выбор семейства признаков относятся к гипотезе и
остаются в отдельном experiment-notebook; граница слоёв описана в
[[docs/epic_experiment_workflow.md]].

> [!abstract] Результат этапа
> Понятно, какие признаки используются, откуда они берутся, в какой момент доступны, как преобразуются и одинаково ли вычисляются при train и inference.

## Model-ready выборка

Здесь принимаются окончательные решения о составе данных после [[docs/02_eda.md|EDA]]. Каждое удаление строки, фильтр или join должны иметь причину и выполняться кодом внутри воспроизводимого pipeline.

```text
DATA snapshot → фильтры / join → split по [[docs/03_validation.md]] → preprocessing внутри folds → model-ready matrices
```

### Контракт EDA model-probe для логистической регрессии

Это исследовательский контракт [[notebooks/02_eda_logistic_context.ipynb]], а не окончательно принятый production feature set.

**Единица строки:** `(assembly=jaNemVect1.1, contig, coordinate_0based, strand)` внутри train-whitelist.

**Формирование target:** BED каждой csRNA-повторности раскрывается до отдельных position-strand; `count = r1 + r2`. Для ключа whitelist, отсутствующего в разреженном BED, `count=0`. Бинарный target логрегрессии равен `int(count > 0)`. Исходный count сохраняется только для диагностики интенсивности и EPIC dense-rank correlation.

**Pilot-строки:** seed=42, без замещения по 1 500 положительных и 6 000 отрицательных на каждом из 12 train-контигов, всего 90 000 строк. Split — три фиксированные группы целых контигов; обе цепи всегда вместе. Evaluation использует inverse-inclusion `population_weight` внутри `contig × class`; training дополнительно балансирует суммарный вес двух классов. Score не считается калиброванной вероятностью.

**Формат хранения:**

| Артефакт | Формат | Содержание |
|---|---|---|
| `sample_rows.csv.gz` | gzip CSV, 90 000 строк | ключи, fold, target, count, population weight |
| Матрица `X` | `scipy.sparse.csr_matrix`, строится в памяти | позиционные one-hot и взаимодействия; агрегаты добавляются отдельными столбцами |
| `oof_predictions.csv.gz` | gzip CSV | ключи выборки и OOF-score каждого кандидата |
| `scores.csv`, `coefficients.csv`, `per_contig_scores.csv` | CSV | интерпретируемые результаты и диагностика |

Полный train на 311 871 646 position-strand не выгружается в feature CSV. Уже `pos401` содержит 1 604 разреженных позиционных столбца; плоский файл был бы избыточным и неудобным. Матрица детерминированно пересобирается из проверенной FASTA, реестра строк и конфигурации признаков. Для будущего полного inference применяется обработка блоками.

**Контракт ориентации:** общий extractor получает таблицу с `contig`,
`coordinate_0based`, `strand` и список offset. Для `+` используется
`coordinate + offset`; для `-` — `coordinate - offset` с комплементированием.
Поэтому offset `-1` имеет один биологический смысл на обеих цепях. Контекст не
переходит через границу contig; отсутствующая буква кодируется `N/edge`.

**Контракт кодирования:** общий feature encoder использует фиксированный
словарь, одинаковый на train/validation/test. Динуклеотид `[-1,0]` кодируется
15 контрастами канонических пар относительно `TT` и отдельным столбцом
`N/edge`. Позиционные окна кодируются независимо на каждом offset относительно
`T`. Порядок и имена sparse-столбцов не зависят от категорий, встретившихся в
конкретной выборке или fold.

**Колонки `sample_rows.csv.gz`:** `row_id`, `template_index`, `contig`, `coordinate_0based`, `strand`, `fold`, `target`, `count`, `population_weight`. Идентификаторы, координата, contig, strand, counts и сигнальные треки не входят в `X`.

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

- **Доступная информация:** только исходная ДНК и производные от неё признаки; все они доступны при inference.
- **Baseline:** ориентированный по цепи динуклеотид `[-1,0]`, reference=`TT`.
- **Измеренная ширина цельного слова:** lookup-серия `2 → 4 → 6 → 8` выбрала окно `[-5,0]` (`k=6`); это ориентир для следующего feature-эксперимента, а не уже готовая матрица логистической регрессии.
- **Позиционное представление:** A/C/G/N-or-edge на каждом offset, reference=`T`; окна 11/21/51/101/201/401 bp.
- **Агрегаты состава:** GC среди ACGT, доля N, доля окна за границей контига, `gc_missing`; масштабы 101/201/401/801 bp при фиксированном pos51.
- **Отдельные абляции:** соседние динуклеотидные взаимодействия `[-5,+5]` и lowercase fraction 201 bp.
- **Ограничение:** компактное разреженное представление и потоковый inference; полный dense feature table не создаётся.

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
| Цельный 4-mer `[-3,0]` | Train-only lookup со сглаживанием к 2-mer | Локальный мотив длиннее пары переносит дополнительный сигнал | [[experiments/EXP-002.md|EXP-002]] | `adopt`: AP +11.77%, Spearman вырос |
| Цельный 6-mer `[-5,0]` | Train-only lookup со сглаживанием к 4-mer | Ещё две upstream-буквы улучшают ранжирование | [[experiments/EXP-003.md|EXP-003]] | `adopt`: AP +7.65%, текущий champion ширины |
| Цельный 8-mer `[-7,0]` | Train-only lookup со сглаживанием к 6-mer | Более длинный мотив окупит редкость категорий | [[experiments/EXP-004.md|EXP-004]] | `reject`: AP +0.18%, Spearman снизился |
| Иерархические one-hot `2 + 4 + 6-mer` | Три активных suffix-признака, L2 logistic regression | Обучить силу backoff вместо фиксированной формулы | [[experiments/EXP-005.md|EXP-005]] | `reject`: AP +0.347% ниже порога +1%; Spearman вырос |
| Линейный GC окна 201 bp | Train-standardized доля G/C среди валидных A/C/G/T в `[-100,+100]` | Добавить к локальному мотиву сигнал состава промоторного окружения | [[experiments/EXP-006.md|EXP-006]] | `reject`: AP −2.015% против EXP-005, Spearman вырос |
| Категориальный GC201 | Восемь fixed bins + `missing`, reference=`[0.35,0.40)` | Проверить немонотонный GC-сигнал без линейного ограничения | [[experiments/EXP-007.md|EXP-007]] | `adopt`: AP +28.309% и Spearman +0.007117 против EXP-003 |
| `k2[-1,0] × GC201-bin` | 128 reference-coded interaction-столбцов поверх EXP-007 | Проверить, зависит ли локальный start-мотив от GC-окружения | [[experiments/EXP-008.md|EXP-008]] | `reject`: AP +0.519%, Spearman немного снизился, CV gate не пройден |
| Fixed CpG O/E201 bins | `CpG × valid / (C × G)` в `[-100,+100]`, семь contrasts относительно `[0.75,1.00)` | Проверить расположение C/G сверх их общей доли | [[experiments/EXP-009.md|EXP-009]] | `prepared`: ожидает ручного запуска |

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
| 4-mer lookup | `[-3,-2]` к паре `[-1,0]` | [[experiments/EXP-002.md|EXP-002]] | AP / Spearman | +11.77% / +0.003512 | принять |
| 6-mer lookup | `[-5,-4]` к 4-mer | [[experiments/EXP-003.md|EXP-003]] | AP / Spearman | +7.65% / +0.002954 | принять |
| 8-mer lookup | `[-7,-6]` к 6-mer | [[experiments/EXP-004.md|EXP-004]] | AP / Spearman | +0.18% / −0.004983 | отклонить |
| Hierarchical LR | one-hot `2/4/6-mer` вместо фиксированного lookup | [[experiments/EXP-005.md|EXP-005]] | AP / Spearman | +0.347% / +0.000384 | отклонить по порогу AP |
| Linear GC201 | один стандартизованный GC-признак к hierarchical LR | [[experiments/EXP-006.md|EXP-006]] | AP / Spearman vs EXP-005 | −2.015% / +0.003636 | отклонить линейную форму |
| Fixed GC201 bins | категориальный GC201 вместо linear GC | [[experiments/EXP-007.md|EXP-007]] | AP / Spearman vs EXP-003 | +28.309% / +0.007117 | принять |
| k2 × GC201-bin | 128 interactions поверх main effects EXP-007 | [[experiments/EXP-008.md|EXP-008]] | AP / Spearman vs EXP-007 | +0.519% / −0.000135 | отклонить |
| CpG O/E201 bins | семь fixed contrasts поверх EXP-007 | [[experiments/EXP-009.md|EXP-009]] | AP / Spearman vs EXP-007 | ожидает запуска | pending |

## Отклонённые признаки

| Признак | Причина | Доказательство / эксперимент | Можно пересмотреть, если… |
|---|---|---|---|
| Цельный 8-mer как следующий основной контекст | Практически нулевой AP gain и ухудшение secondary metric из-за разреженности | [[experiments/EXP-004.md|EXP-004]] | другая параметризация или новый train-only способ регуляризации проверяется отдельной гипотезой |
| Один линейный коэффициент абсолютного GC201 | Принудительно монотонно повышает score, хотя train enrichment имеет максимум и спад; AP снизился на всех validation-contig | [[experiments/EXP-006.md|EXP-006]] | немонотонная форма проверяется fixed bins в EXP-007 |
| Взаимодействия `k2 × GC201-bin` | Прирост AP +0.519% ниже порога, Spearman снизился, train-CV gate не пройден | [[experiments/EXP-008.md|EXP-008]] | другая биологическая гипотеза проверяется отдельным экспериментом |

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
