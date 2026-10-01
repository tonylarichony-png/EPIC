---
id: IDEA-004
type: experiment-idea
status: idea
owner: Команда EPIC
created: 2026-09-15
hypothesis: H-004
dataset_version: "jaNemVect1.1; MD5 733dccb6393a3c07b6e0d7ffaed3686f"
code_version: "working-tree; SHA256 исходного runner будет записан до fit"
model: "LogisticRegression L2 C=1 lbfgs max_iter=350 tol=1e-4"
seed: 42
primary_metric: weighted_sampled_average_precision
result: not_run
eda_findings: ["EDA-004"]
tags:
  - ml/experiment-idea
  - ml/eda
---

# IDEA-004 — Вклад lowercase после учёта состава

← [[experiment-ideas/_index.md|Банк идей]] · [[experiments/_index.md|Реестр экспериментов]] · [[README.md|Dashboard]]

> [!warning] Это не проведённый и не зарегистрированный эксперимент
> Карточка хранит только подсказку для будущей проверки. Когда идея будет
> выбрана, создайте новый `EXP-xxx` с отдельным notebook и актуальным протоколом.

> [!abstract] Цель
> Регистр FASTA добавляет информацию сверх pos51 и composition201. Для shortlist требуются относительный прирост среднего по 3 folds weighted sampled AP ≥5% к pos51 + composition201, нижняя граница 95% парного bootstrap-интервала среднего по контигам ΔAP >0 и сходимость. Иначе outcome iterate.

## Черновой план будущего эксперимента

Расчёты не запускались. Параметры ниже — материал для проектирования будущего
эксперимента, а не pre-registration и не результат.

### Связи

- **Гипотеза:** [[hypotheses/H-004.md|H-004]]; **EDA:** EDA-004.
- **Validation:** [[docs/03_validation.md]]; исполняемый протокол указан ниже.
- **Предшествующая идея:** [[experiment-ideas/IDEA-002_sequence_composition.md|IDEA-002]], возможный pos51 + composition201 reference.

### Изменение

- **Что меняем:** Добавить lowercase-признаки к фиксированному pos51 + composition201 и сравнить с тем же набором без lowercase.
- **Что остаётся фиксированным:** позиции, folds, label, sampling, seed, базовое кодирование, алгоритм, регуляризация и веса.
- **Почему ожидаем эффект:** В прежнем train-аудите частота сигнала на lowercase 0.00027820 против 0.00083429 на uppercase; независимый от GC и pos51 вклад не установлен.
- **Критерий успеха:** Для shortlist требуются относительный прирост среднего по 3 folds weighted sampled AP ≥5% к pos51 + composition201, нижняя граница 95% парного bootstrap-интервала среднего по контигам ΔAP >0 и сходимость. Иначе outcome iterate.
- **Guardrails:** Сходимость, AP по контигам и регистру центра, dense-rank корреляция, одинаковые positions/folds. Lowercase не использовать как основание удалять позиции.
- **Бюджет / stop condition:** 1 новый кандидат × 3 folds; reference можно взять из будущей проверки IDEA-002. Максимум 350 итераций каждого fit. Полная серия: 13 кандидатов × 3 folds; full-train refit не выполняется.

## Setup

### Данные

| Поле | Значение |
|---|---|
| Dataset version | jaNemVect1.1; MD5 733dccb6393a3c07b6e0d7ffaed3686f |
| Период | Пространственная геномная задача; временное разбиение неприменимо |
| Train / validation / test | 12 официальных train-контигов, 3 grouped folds с группировкой по длинам; test labels не читаются |
| Sampling / filters | 1 500 positives + 6 000 negatives на контиг без замещения; одинаковые позиции для всех кандидатов |
| Target version | Событие pooled csRNA count > 0; исходный count сохранён для диагностики интенсивности |
| Training weights | IPW с последующим балансированием сумм весов классов и нормированием среднего веса к 1; score не калиброван |
| Evaluation weights | IPW по вероятностям отбора positives и negatives отдельно внутри каждого контига |

### Features

- **Feature set / version:** pos51 + composition201; кандидат добавляет только lower_fraction_201 (без отдельного признака lowercase центра). Composition: GC/valid ACGT (0 при отсутствии ACGT), gc_missing, N_fraction/full width, edge_fraction/full width.
- **Preprocessing:** все fit-преобразования только на обучающих контигах fold; StandardScaler для composition с training weights. DNA ориентирована по цепи. Коэффициенты долей переводятся в исходную шкалу на +0.10 (10 п.п.).
- **Изменения к baseline:** Добавить lowercase-признаки к фиксированному pos51 + composition201 и сравнить с тем же набором без lowercase.

### Model

- **Алгоритм / параметры:** LogisticRegression, L2, C=1, solver=lbfgs, max_iter=350, tol=1e-4.
- **Initialization / pretrained version:** обучение с нуля; **seed:** 42.
- **Score:** исследовательский score без утверждения калибровки; не готовый конкурсный submission.

### Воспроизводимость

| Поле | Значение |
|---|---|
| Repository | D:/Projects/EPIC/EPIC |
| Code / config | scripts/eda_logistic_context.py; preregistration.json и SHA256 runner фиксируются до fit |
| Notebook / command | notebooks/02_eda_logistic_context.ipynb → Run All в kernel epic-eda |
| Environment | Miniforge conda epic-eda |
| Hardware | Локальный CPU; фактические параметры и время запишет runner |
| Tracking run | не создан |

## Ожидаемый формат результата будущего EXP

### Основные метрики

| Split / fold | Metric | Baseline | Result | Δ | Notes |
|---|---|---|---|---|---|
| Grouped 3-fold | Weighted sampled AP | возможный pos51 + composition201 reference из IDEA-002 | не вычислялось | не вычислялось | черновой план |
| Sampled positives | Dense-rank Pearson | тот же reference | не вычислялось | не вычислялось | черновой план |

### Guardrails и сегменты

Планируется проверить сходимость, обе метрики по contig/регистру и одинаковые positions/folds. Lowercase не использовать как основание удалять позиции.

### Стабильность

Парные сравнения на одинаковых OOF-позициях; 200 bootstrap-повторов по 12 контигам, 95% percentile-интервал среднего парного ΔAP по контигам (macro). Интервал условен на fitted OOF и выбранных строках: без refit и повторного sampling. Это иной estimand, чем mean fold AP; интервалы остаются исследовательскими.

### Ресурсы

Время fit/inference не измерялось. Черновой бюджет: 1 новый кандидат × 3 folds; возможный reference из IDEA-002; максимум 350 итераций каждого fit.

## Артефакты

- **Notebook / report:** [[notebooks/02_eda_logistic_context.ipynb]].
- **Code / logs / predictions / plots:** отсутствуют; будущий EXP должен сохранить их отдельно.
- **Model artifact:** full-train модель в этой серии не создаётся.

## Анализ результата

Результатов нет. При будущем запуске нужно явно отделить exploratory validation
от подтверждающей проверки и не использовать demo-test для выбора.

## Что решать после будущего запуска

- Принять lowercase-признаки только при независимой добавочной пользе.
- Решение записывать только в будущую карточку `EXP-xxx`, не в IDEA.

## Следующие действия

- [ ] При выборе идеи создать новый `EXP-xxx` и заново проверить протокол.
- [ ] До fit зафиксировать критерии по Average Precision и EPIC Spearman.

## Что перенести в будущую карточку EXP

- [ ] Проверить актуальность dataset, seed, модели, sampling и бюджета.
- [ ] Создать настоящую pre-registration до fit.
- [ ] Записаны source SHA256, выполненный config и версии окружения.
- [ ] Проверены guardrails, стоимость, метрики и сохранены артефакты.
- [ ] Заполнены вывод и решение; обновлена сводка экспериментов.

## EDA-основания

<!-- auto:experiment-eda-links:start -->

| EDA-наблюдение                                                                 | Признаки                        | Ключевой вывод                                                                                                                                                            |
| ------------------------------------------------------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[eda/findings/EDA-004.md\|EDA-004 — Регистр FASTA связан с частотой сигнала]] | lowercase fraction, N indicator | Частота положительного target на uppercase ACGT равна 0.00083429, на lowercase acgt — 0.00027820. В центре N/other положительных не найдено: 0 из 52 800 position-strand. |

<!-- auto:experiment-eda-links:end -->

