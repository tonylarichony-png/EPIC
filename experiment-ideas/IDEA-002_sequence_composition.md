---
id: IDEA-002
type: experiment-idea
status: idea
owner: Команда EPIC
created: 2026-09-15
hypothesis: H-002
dataset_version: "jaNemVect1.1; MD5 733dccb6393a3c07b6e0d7ffaed3686f"
code_version: "working-tree; SHA256 исходного runner будет записан до fit"
model: "LogisticRegression L2 C=1 lbfgs max_iter=350 tol=1e-4"
seed: 42
primary_metric: weighted_sampled_average_precision
result: not_run
eda_findings: ["EDA-002"]
tags:
  - ml/experiment-idea
  - ml/eda
---

# IDEA-002 — Состав последовательности на разных масштабах

← [[experiment-ideas/_index.md|Банк идей]] · [[experiments/_index.md|Реестр экспериментов]] · [[README.md|Dashboard]]

> [!warning] Это не проведённый и не зарегистрированный эксперимент
> Карточка хранит только подсказку для будущей проверки. Когда идея будет
> выбрана, создайте новый `EXP-xxx` с отдельным notebook и актуальным протоколом.

> [!abstract] Цель
> Агрегированный состав более широкого окна добавляет информацию к pos51. Для shortlist требуются относительный прирост среднего по 3 folds weighted sampled AP ≥5% к pos51, нижняя граница 95% парного bootstrap-интервала среднего по контигам ΔAP >0 и сходимость. Иначе outcome iterate.

## Черновой план будущего эксперимента

Расчёты не запускались. Параметры ниже — материал для проектирования будущего
эксперимента, а не pre-registration и не результат.

### Связи

- **Гипотеза:** [[hypotheses/H-002.md|H-002]]; **EDA:** EDA-002.
- **Validation:** [[docs/03_validation.md]]; исполняемый протокол указан ниже.
- **Предшествующая идея:** [[experiment-ideas/IDEA-001_context_width.md|IDEA-001]], возможный pos51 reference.

### Изменение

- **Что меняем:** Добавить composition101, composition201, composition401 и composition801 по одному к фиксированному pos51; сравнить с pos51.
- **Что остаётся фиксированным:** позиции, folds, label, sampling, seed, базовое кодирование, алгоритм, регуляризация и веса.
- **Почему ожидаем эффект:** Прежний EDA показал различие среднего GC в окне 201 bp у positive и равномерного train background; независимый от pos51 эффект не измерен.
- **Критерий успеха:** Для shortlist требуются относительный прирост среднего по 3 folds weighted sampled AP ≥5% к pos51, нижняя граница 95% парного bootstrap-интервала среднего по контигам ΔAP >0 и сходимость. Иначе outcome iterate.
- **Guardrails:** Сходимость всех folds кандидата, парное сравнение на одинаковых позициях, AP по folds/контигам и dense-rank корреляция; возможная связь composition с контигом.
- **Бюджет / stop condition:** 4 новых кандидата × 3 folds; pos51 можно взять из будущей проверки IDEA-001. Максимум 350 итераций каждого fit. Полная серия: 13 кандидатов × 3 folds; full-train refit не выполняется.

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

- **Feature set / version:** Позиционные основания 51 bp + composition101/201/401/801: GC/valid ACGT (0 при отсутствии ACGT), gc_missing, доля N внутри контига / полная ширина, edge_fraction за границей контига / полная ширина. Lowercase вынесен в IDEA-004.
- **Preprocessing:** все fit-преобразования только на обучающих контигах fold; StandardScaler для composition с training weights. DNA ориентирована по цепи. Коэффициенты долей переводятся в исходную шкалу на +0.10 (10 п.п.).
- **Изменения к baseline:** Добавить composition101, composition201, composition401 и composition801 по одному к фиксированному pos51; сравнить с pos51.

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
| Grouped 3-fold | Weighted sampled AP | возможный pos51 reference из IDEA-001 | не вычислялось | не вычислялось | черновой план |
| Sampled positives | Dense-rank Pearson | тот же reference | не вычислялось | не вычислялось | черновой план |

### Guardrails и сегменты

Планируется проверить сходимость, обе метрики по folds/contig и возможную связь composition с contig.

### Стабильность

Парные сравнения на одинаковых OOF-позициях; 200 bootstrap-повторов по 12 контигам, 95% percentile-интервал среднего парного ΔAP по контигам (macro). Интервал условен на fitted OOF и выбранных строках: без refit и повторного sampling. Это иной estimand, чем mean fold AP; интервалы остаются исследовательскими.

### Ресурсы

Время fit/inference не измерялось. Черновой бюджет: 4 новых кандидата × 3 folds; возможный pos51 reference из IDEA-001; максимум 350 итераций каждого fit.

## Артефакты

- **Notebook / report:** [[notebooks/02_eda_logistic_context.ipynb]].
- **Code / logs / predictions / plots:** отсутствуют; будущий EXP должен сохранить их отдельно.
- **Model artifact:** full-train модель в этой серии не создаётся.

## Анализ результата

Результатов нет. При будущем запуске нужно явно отделить exploratory validation
от подтверждающей проверки и не использовать demo-test для выбора.

## Что решать после будущего запуска

- Выбрать самый простой полезный масштаб либо признать идею неподтверждённой.
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

| EDA-наблюдение                                                                    | Признаки                             | Ключевой вывод                                                                                                                                                                  |
| --------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[eda/findings/EDA-002.md\|EDA-002 — GC и состав широкого окна связаны с target]] | GC fraction, N fraction, contig edge | Средний GC в окне 201 bp равен 0.43334 около положительных и 0.40330 на равномерном train-фоне. Это описательная ассоциация; добавочная польза и нужный масштаб не установлены. |

<!-- auto:experiment-eda-links:end -->

