---
type: experiment-note
project: EPIC
status: research
created: 2026-09-24
tags:
  - epic
  - false-positive
  - tss
  - sequence-model
  - error-analysis
  - cnn
  - genomics
related:
  - CNN-EXP-033
---

# EPIC — исследование false positives EXP033

## Контекст

Исследование выполнено для:

```text
CNN-EXP-033
NO BPE
base-level CNN
RF = 2053 bp
regional head = 32 bp
checkpoint = 10k steps
```

Holdout:

```text
regions = 2,560,380
positive regions = 27,982
regional prevalence = 1.0929%
```

Метрики EXP033 @10k:

```text
Regional AP32       = 0.119389
AP lift             = 10.924
Top-10% TSS recall  = 71.70%
95% TSS recall при  = 43.47% выбранных регионов
```

Цель исследования — понять природу наиболее уверенных false positives.

Главный вопрос:

> Ошибается ли CNN, потому что ещё недостаточно хорошо понимает DNA sequence,  
> или часть ошибок возникает потому, что по DNA регион выглядит как потенциальный TSS/promoter,
> но в конкретном экспериментальном состоянии транскрипция там не наблюдается?

---

# 1. Определение исследуемых групп

На полном holdout были сформированы группы по 10 000 регионов.

```text
HARD_FP
target = 0
model score = максимально высокий

EASY_NEG
target = 0
model score = максимально низкий

HIGH_TP
target = 1
model score = максимально высокий

RANDOM_TP
случайные target = 1
```

Для каждой группы анализировался strand-oriented DNA-контекст длиной 256 bp вокруг 32-bp региона.

---

# 2. Распределение model score

Для negative regions:

| quantile | score |
|---:|---:|
| 0.50 | 0.188 |
| 0.75 | 0.481 |
| 0.90 | 0.745 |
| 0.99 | 0.948 |
| 0.999 | 0.988 |

Для positive regions:

| quantile | score |
|---:|---:|
| 0.10 | 0.310 |
| 0.25 | 0.585 |
| 0.50 | 0.816 |
| 0.75 | 0.931 |
| 0.90 | 0.976 |
| 0.99 | 0.997 |

### Наблюдение

Распределения заметно перекрываются.

Верхний хвост negatives получает score, сопоставимый с сильными positives.

Это и есть область HARD_FP.

---

# 3. Простая sequence composition

Средние характеристики:

| group | score median | GC core32 | GC context256 | entropy core32 | entropy context256 |
|---|---:|---:|---:|---:|---:|
| EASY_NEG | 0.00279 | 0.3003 | 0.3656 | 1.6873 | 1.8958 |
| HARD_FP | 0.98193 | 0.4611 | 0.4425 | 1.9038 | 1.9705 |
| HIGH_TP | 0.95441 | 0.4597 | 0.4356 | 1.9044 | 1.9681 |
| RANDOM_TP | 0.67539 | 0.4358 | 0.4214 | 1.8913 | 1.9585 |

### Наблюдение

`HARD_FP` и `HIGH_TP` практически совпадают по:

- GC внутри 32 bp;
- GC в 256 bp;
- sequence entropy;
- доле N.

При этом `EASY_NEG` сильно отличается.

Упрощённо:

```text
EASY_NEG  ≠  HARD_FP ≈ HIGH_TP
```

---

# 4. 6-mer enrichment

Сравнивались частоты 6-mer относительно EASY_NEG.

Типичные примеры:

```text
GACTCG
HARD_FP vs EASY_NEG ≈ +3.36 log2
HIGH_TP vs EASY_NEG ≈ +3.07 log2

CGGACG
HARD_FP vs EASY_NEG ≈ +2.93
HIGH_TP vs EASY_NEG ≈ +2.92
```

`log2 enrichment ≈ 3` означает примерно 8-кратное обогащение.

Многие наиболее enriched 6-mer совпадают между HARD_FP и HIGH_TP.

Позже при score-matched сравнении:

```text
6-mer enrichment correlation = 0.9814
Top-50 enriched 6-mer Jaccard = 0.4286
Median |log2(HARD_FP / MATCHED_TP)| = 0.1164
```

`0.1164 log2` соответствует примерно 8% типичной разнице частоты.

### Вывод

У HARD_FP и сильных TP почти одинаковый общий словарь коротких sequence motifs.

---

# 5. 4-mer similarity

Использовался bag-of-4-mer representation.

Средняя cosine similarity:

| group | similarity to HIGH_TP centroid | similarity to EASY_NEG centroid | positive margin |
|---|---:|---:|---:|
| EASY_NEG | 0.6247 | 0.7131 | -0.0885 |
| HARD_FP | 0.6941 | 0.5972 | +0.0969 |
| HIGH_TP | 0.7042 | 0.6110 | +0.0932 |
| RANDOM_TP | 0.6966 | 0.6268 | +0.0698 |

### Наблюдение

По 4-mer representation HARD_FP настолько же "positive-like", как и сильные настоящие positives.

---

# 6. Простая классификация HARD_FP vs другие группы

Logistic Regression на bag-of-4-mer:

```text
HARD_FP vs EASY_NEG   AUC = 0.987
HARD_FP vs HIGH_TP    AUC = 0.610
HARD_FP vs RANDOM_TP  AUC = 0.697
```

### Интерпретация

- HARD_FP почти идеально отделяются от обычных negatives.
- HARD_FP слабо отделяются от HIGH_TP.
- HARD_FP ближе именно к сильным TP, чем к случайным TP.

Это означает, что HARD_FP — не случайный набор negatives.

---

# 7. Расстояние HARD_FP до настоящих TSS

Было проверено расстояние от каждого negative region до ближайшего exact TSS.

Медианные расстояния:

| group | same-strand TSS | any-strand TSS | opposite-strand TSS |
|---|---:|---:|---:|
| EASY_NEG | 4695 bp | 1528.5 bp | 2416.5 bp |
| HARD_FP | 1856 bp | 971.5 bp | 4791 bp |
| RANDOM_NEG | 3194 bp | 1289 bp | 3128 bp |

HARD_FP заметно ближе именно к **same-strand TSS**.

---

# 8. Доля negatives около TSS

## Same-strand

| radius | EASY_NEG | HARD_FP | RANDOM_NEG |
|---:|---:|---:|---:|
| ±32 bp | 0.01% | 9.17% | 0.83% |
| ±64 bp | 0.16% | 16.69% | 2.24% |
| ±128 bp | 0.73% | 22.29% | 4.60% |
| ±256 bp | 2.81% | 27.40% | 8.68% |
| ±512 bp | 7.05% | 33.01% | 15.87% |
| ±1024 bp | 14.99% | 41.06% | 26.62% |
| ±2048 bp | 28.41% | 51.76% | 39.67% |

### Ключевой результат

В радиусе ±32 bp HARD_FP встречаются рядом с настоящим same-strand TSS примерно в:

```text
9.17 / 0.83 ≈ 11×
```

чаще, чем RANDOM_NEG.

Это показывает существование реального класса:

```text
локализационные FP около настоящих TSS
```

Но:

```text
48.24% HARD_FP не имеют same-strand TSS даже в ±2 kb.
```

Следовательно, этим нельзя объяснить большую часть всех HARD_FP.

---

# 9. Opposite-strand анализ

Для opposite strand HARD_FP не показывают такого обогащения.

Например:

```text
within ±1024 bp:
HARD_FP    = 22.57%
RANDOM_NEG = 26.69%

within ±2048 bp:
HARD_FP    = 32.45%
RANDOM_NEG = 40.22%
```

### Интерпретация

Высокие FP не выглядят просто как участки с общей локальной транскрипционной активностью.

Сигнал гораздо более strand-specific.

---

# 10. Направление относительно same-strand TSS

Среди HARD_FP, имеющих same-strand TSS в пределах ±1 kb:

```text
n = 4106

upstream   = 49.63%
downstream = 50.37%

median signed distance = +16 bp
```

### Интерпретация

Нет выраженного upstream/downstream bias.

Для близких FP настоящий TSS часто находится почти в центре соседнего 32-bp региона.

Это согласуется с существованием локализационных ошибок.

---

# 11. FAR_FP и ISOLATED_FP

Чтобы убрать эффект соседних TSS, были определены более строгие группы.

```text
FAR_FP
нет same-strand TSS в ±2 kb

ISOLATED_FP
нет TSS ни одного strand в ±2 kb
```

Размеры:

| cohort | n | fraction of HARD_FP | score median |
|---|---:|---:|---:|
| ALL_HARD_FP | 10000 | 100% | 0.981931 |
| FAR_FP | 4824 | 48.24% | 0.981921 |
| ISOLATED_FP | 3586 | 35.86% | 0.982041 |

### Очень важное наблюдение

После удаления всех nearby TSS score практически не изменился.

```text
ALL HARD_FP   ≈ 0.98193
ISOLATED_FP   ≈ 0.98204
```

То есть isolated false positives модель считает TSS-подобными с той же уверенностью.

---

# 12. Строгий score-control

Первоначальный one-to-one matching был недостаточно строгим.

Поэтому был сделан новый контроль:

```text
score -> logit
bin width = 0.10
equal FP / TP count inside each bin
rank pairing inside bin
```

Для `ISOLATED_FP`:

```text
pairs = 2882
retained = 80.37%

FP median score = 0.983979
TP median score = 0.983968

median |score difference| = 0.000042
95%    |score difference| = 0.000279

median |logit difference| = 0.00328
95%    |logit difference| = 0.01901
```

### Вывод

Это практически идеальный score-control.

---

# 13. Composition после строгого score-control

Для ISOLATED_FP:

| feature | ISOLATED_FP | MATCHED_TP |
|---|---:|---:|
| score median | 0.983979 | 0.983968 |
| GC core32 | 0.4633 | 0.4623 |
| GC context256 | 0.4447 | 0.4350 |
| entropy core32 | 1.9027 | 1.9078 |
| entropy context256 | 1.9700 | 1.9708 |

### Наблюдение

Простая composition почти совпадает.

Небольшое отличие остаётся в GC окружающего 256-bp контекста.

---

# 14. Bag-of-4-mer vs position-aware sequence

После строгого score-control:

| cohort | pairs | bag-of-4mer AUC | position-aware 256bp AUC |
|---|---:|---:|---:|
| FAR_FP | 3132 | 0.6330 | 0.5274 |
| ISOLATED_FP | 2882 | 0.6373 | 0.4975 |

### Ключевой результат

Для `ISOLATED_FP`:

```text
position-aware AUC = 0.4975
```

То есть линейный классификатор, который знает:

```text
какой nucleotide находится на каждой конкретной позиции
```

практически не отличает FP от TP.

При этом bag-of-4-mer:

```text
AUC = 0.637
```

содержит некоторый различающий сигнал.

### Интерпретация

Различие между FP и TP:

- не выглядит как простая nucleotide-position signature;
- частично содержится в общей частоте коротких patterns;
- может находиться в более сложной nonlinear motif grammar.

---

# 15. Positional profiles ключевых 6-mer

Для общих enriched motifs анализировались позиции появления внутри 256-bp окна.

Примеры ISOLATED_FP:

```text
motif     max positional frequency diff

GACTCG    0.001388
CGGACG    0.001735
CGTCGA    0.001388
TCGCGG    0.001041
ACGGCG    0.001388
TGGCGG    0.003123
```

Средняя absolute positional difference обычно порядка:

```text
0.0002–0.0005
```

то есть примерно:

```text
0.02–0.05 процентного пункта
```

### Вывод

Для исследованных motifs абсолютное расположение внутри окна у FP и TP почти совпадает.

---

# 16. Что уже удалось исключить

Исследование ослабляет следующие объяснения.

## 16.1 «Это просто обычные negatives»

Нет.

```text
HARD_FP vs EASY_NEG
4-mer AUC = 0.987
```

HARD_FP — совершенно другой sequence-class.

## 16.2 «Это только соседние bins вокруг настоящих TSS»

Нет.

Да, такой эффект существует:

```text
9.17% HARD_FP находятся в ±32 bp same-strand TSS
27.40% — в ±256 bp
```

Но:

```text
48.24% не имеют same-strand TSS ±2 kb
35.86% не имеют вообще никакого TSS ±2 kb
```

## 16.3 «FP имеют просто другой GC / entropy»

Нет.

После score-control composition очень близка.

## 16.4 «FP и TP отличаются очевидной position-specific DNA signature»

Не обнаружено.

```text
ISOLATED_FP vs matched TP
position-aware linear AUC ≈ 0.498
```

## 16.5 «Ключевые motifs находятся в совершенно разных местах»

Для исследованных enriched 6-mer этого не видно.

Positional frequency differences очень малы.

---

# 17. Что пока НЕ исключено

Исследование не доказывает, что CNN достигла sequence ceiling.

Остаются как минимум две крупные гипотезы.

## A. Более сложная motif grammar

Например:

```text
motif A
   ↓ 43 bp
motif B
   ↓ 77 bp
motif C
```

Различие может находиться в:

- pairwise motif spacing;
- combinations;
- motif orientation;
- nonlinear interactions;
- более длинных motifs;
- более дальнем sequence context.

Линейный position-aware classifier этого не моделирует.

## B. Biological / experimental state

DNA может быть sequence-compatible с initiation, но конкретно в данном эксперименте site не активен из-за:

- chromatin accessibility;
- TF availability;
- cell state;
- enhancer/promoter state;
- stochastic initiation;
- ограниченной глубины csRNA-seq;
- experimental noise.

Текущий анализ не позволяет выбрать между A и B окончательно.

---

# 18. Главный промежуточный вывод

Наиболее уверенные false positives EXP033 формируют отдельный, очень специфический sequence-class.

Для `ISOLATED_FP`:

```text
target = 0
нет TSS любого strand в ±2 kb
model score ≈ 0.984
score практически идеально matched с TP
GC / entropy ≈ TP
short motif vocabulary ≈ TP
position-aware linear sequence classifier ≈ random
positions of selected enriched 6-mers ≈ TP
```

При этом bag-of-4-mer classifier всё ещё даёт:

```text
AUC ≈ 0.637
```

то есть FP и TP не идентичны по DNA.

Корректная формулировка:

> Существует большая популяция высокоуверенных false positives, которые по локальной DNA-последовательности очень похожи на настоящие активные TSS-регионы и не объясняются соседством с наблюдаемыми TSS. Простые композиционные и позиционные признаки плохо различают эти группы. Остающийся различающий сигнал может находиться либо в более сложной nonlinear motif grammar, либо в биологическом/экспериментальном состоянии, отсутствующем в sequence-only input.

---

# 19. Практический вывод для дальнейшей ML-работы

До этого момента основной цикл был:

```text
RF
channels
heads
context
architecture
→ небольшой прирост AP
```

FP-анализ показывает, что следующий полезный вопрос — уже не только:

> «Как сделать CNN мощнее?»

а:

> «Есть ли вообще дополнительный sequence signal, который позволяет отличить isolated FP от active TP?»

Следующий логичный эксперимент:

```text
ISOLATED_FP vs score-controlled TP
↓
nonlinear motif-grammar classifier
```

Например:

- небольшой CNN отдельно на 256 bp;
- small transformer;
- explicit motif-pair / spacing features;
- k-mer interactions;
- longer sequence context.

Если мощный специально обученный classifier всё равно не сможет хорошо отделять `ISOLATED_FP` от score-controlled TP, гипотеза о sequence-only ceiling станет значительно сильнее.

---

# 20. Текущее состояние гипотез

| Гипотеза | Статус |
|---|---|
| FP — обычные negatives | сильно ослаблена |
| FP — в основном соседние bins около TSS | частично объясняет ошибки, но недостаточно |
| FP отличаются простым GC/composition | сильно ослаблена |
| FP отличаются простой nucleotide-position signature | не поддержано |
| FP отличаются позициями нескольких ключевых 6-mer | почти не поддержано |
| FP отличаются более сложной motif grammar | остаётся открытой |
| часть FP обусловлена biological / experimental state | правдоподобна, но пока не доказана |

---

## Связанные артефакты

```text
CNN_EXP033_BASELEVEL_RF2053_REGIONAL_32_ONLY_10K.ipynb

EPIC_EXP033_FALSE_POSITIVE_STUDY.ipynb
EPIC_EXP033_FALSE_POSITIVE_STUDY_V2.ipynb
EPIC_EXP033_FALSE_POSITIVE_STUDY_V3_FAR_FP.ipynb
```
