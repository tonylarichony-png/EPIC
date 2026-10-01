---
type: architecture-hypothesis
status: concept
created: 2026-10-01
tags:
  - EPIC
  - TSS
  - CNN
  - stacking
  - signed-distance
  - ensemble
  - multiscale
---

# Геометрический ансамбль CNN для локализации TSS

## Основная идея

Текущие эксперименты показали, что простое расширение receptive field, focal loss, regional heads и joint coarse→fine архитектуры сами по себе не дали качественного скачка.

Новая гипотеза: вместо того чтобы учить сеть только отвечать **«есть TSS / нет TSS»** или **«есть ли TSS где-то в области»**, дать вспомогательным CNN более информативную геометрическую задачу:

> **определить signed distance — точное смещение до ближайшего TSS относительно текущего нуклеотида.**

Таким образом каждый нуклеотид не просто классифицируется, а «голосует», где находится центр TSS.

## 1. Signed-distance supervision

Для модели с радиусом `R=32` target:

```text
-32, -31, ..., -2, -1, 0, +1, +2, ..., +31, +32, NONE
```

Где:

- `0` — TSS находится прямо в текущем нуклеотиде;
- `+k` — ближайший TSS находится на `k` bp downstream;
- `-k` — ближайший TSS находится на `k` bp upstream;
- `NONE` — в заданном радиусе TSS нет.

Это отличается от regional target вида:

```text
есть ли TSS в ±32 bp?
```

Потому что сеть получает не только наличие TSS рядом, но и **направление + точное расстояние до него**.

Интуитивно вокруг настоящего TSS:

```text
... +4  +3  +2  +1   0   -1  -2  -3  -4 ...
                     TSS
```

Центр становится точкой смены знака дистанции.

## 2. Несколько геометрических экспертов разных масштабов

Базовый набор независимых CNN:

```text
CNN R=4
CNN R=8
CNN R=16
CNN R=32
```

Число выходных классов:

```text
R=4   → 10 классов
R=8   → 18 классов
R=16  → 34 класса
R=32  → 66 классов
```

Итого для каждого нуклеотида:

```text
10 + 18 + 34 + 66 = 128 геометрических признаков
```

Важно сохранять **полные logits/probability distributions**, а не только `argmax distance`.

Например:

```text
P(+5)=0.35
P(+6)=0.40
P(+7)=0.20
```

содержит больше информации, чем просто:

```text
predicted_distance = +6
```

## 3. Target radius и receptive field — разные параметры

Радиус геометрической задачи не обязан совпадать с receptive field CNN.

Пример:

```text
target radius R = 8 bp
receptive field = 512 bp
```

Сеть должна определить точное положение TSS в ближайших 8 bp, но для этого может использовать широкий promoter context.

Поэтому банк экспертов можно разнообразить по нескольким осям:

```text
target radius:
4 / 8 / 16 / 32 / 64

receptive field:
64 / 128 / 256 / 512 / 1024+

kernel:
3 / 5 / 7 / 11

dilation:
разные dilation schedules

depth:
разная глубина сети
```

Цель — получить **структурированно разные геометрические эксперты**, а не десятки почти одинаковых CNN.

## 4. Ensemble / committee of spatial experts

```text
                        DNA
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     CNN R4           CNN R8           CNN R16 ...
        │                │                │
  distance logits   distance logits   distance logits
        └────────────────┼────────────────┘
                         │
              multiscale geometry tracks
                         │
                         ▼
                     META-CNN
                         │
              ┌──────────┴──────────┐
              │                     │
        presence score         intensity score
              │                     │
              ▼                     ▼
             AP                  Spearman
```

Это можно рассматривать как **stacking / mixture of spatial experts**.

META-CNN учится интерпретировать:

- согласие экспертов;
- конфликт экспертов;
- изменение predicted distance вокруг позиции;
- uncertainty каждого эксперта;
- локальную пространственную структуру сигналов.

## 5. Random Forest как meta-baseline

Random Forest полезен как **дешёвый meta-baseline**.

Он может проверить, содержат ли геометрические признаки полезный сигнал вообще.

Но основная meta-модель предпочтительно небольшая 1D CNN, потому что важен не только вектор одного nucleotide, но и **изменение геометрических прогнозов вокруг него**.

Пример:

```text
position -2 → эксперты голосуют: TSS справа
position -1 → эксперты голосуют: TSS справа
position  0 → convergence / distance≈0
position +1 → эксперты голосуют: TSS слева
position +2 → эксперты голосуют: TSS слева
```

## 6. Финальная META-CNN решает исходную задачу

Геометрические CNN создают представление.

Финальная сеть получает:

```text
128+ geometry features
+
опционально исходную DNA
```

и предсказывает:

```text
presence score    → используется для AP
intensity/count   → используется для Spearman
```

AP и Spearman не являются target одного nucleotide — это метрики итогового набора предсказаний.

## 7. Loss геометрических сетей

Для каждого эксперта:

```text
CrossEntropy(distance_class)
```

Например для `R=32`:

```text
66 классов:
-32 ... 0 ... +32 + NONE
```

Проблема: `NONE` будет доминировать.

Поэтому проверить:

- weighted cross-entropy;
- controlled negative sampling;
- balanced batches по distance classes;
- позднее — ordinal/distance-aware loss.

На первом эксперименте loss лучше не усложнять сильнее необходимого.

## 8. Критически важный OOF stacking

META-CNN нельзя обучать на prediction tracks экспертов, если эти эксперты видели те же contigs при обучении.

Нужны out-of-fold predictions по contig-level split.

Пример для 5 folds:

```text
Experts train F2+F3+F4+F5 → predict F1
Experts train F1+F3+F4+F5 → predict F2
Experts train F1+F2+F4+F5 → predict F3
Experts train F1+F2+F3+F5 → predict F4
Experts train F1+F2+F3+F4 → predict F5
```

После этого:

```text
OOF geometry features всех train contigs
+
exact TSS target
↓
META-CNN training
```

Для test:

```text
геометрические CNN переобучаются на всём train
↓
создают geometry tracks test
↓
META-CNN
↓
final score
```

Split должен быть именно **по contig'ам**, а не по случайным нуклеотидам.

## 9. Анализ полезности экспертов до META-CNN

До meta-модели надо проверить, действительно ли эксперты дают разные сведения.

Для каждого эксперта:

```text
exact-offset accuracy
top-k offset accuracy
cross-entropy
MAE signed distance
```

Отдельно анализировать:

- рядом с TSS;
- hard negatives;
- далёкие negatives;
- разные contigs.

Для каждой пары CNN:

```text
corr(predicted distance)
corr(P(offset=0))
corr(errors)
```

Особенно важна **корреляция ошибок**.

Плохой дополнительный эксперт:

```text
accuracy A = 70%
accuracy B = 69%
corr(errors A,B) = 0.94
```

Потенциально полезный эксперт:

```text
accuracy A = 70%
accuracy B = 66%
corr(errors A,B) = 0.35
```

Хотя B слабее сам по себе, он может быть ценным в ансамбле, потому что ошибается иначе.

## 10. PCA / effective rank ансамбля

Если обучено много экспертов:

```text
20 CNN
↓
PCA
↓
95% variance объясняется 4 компонентами
```

Тогда фактически имеется только около четырёх типов поведения.

Если для 95% информации нужны 12–15 компонент, ансамбль действительно разнообразен.

Это позволит:

- удалить избыточные CNN;
- оставить экспертов с уникальными ошибками;
- уменьшить вычислительную цену META-CNN;
- понять, какие RF / kernel / radius дают реально новую информацию.

## 11. Как развивать ансамбль

Не стоит сразу обучать десятки дорогих CNN.

Первый пилот:

```text
R8   RF128
R8   RF512

R16  RF128
R16  RF512

R32  RF256
R32  RF1024
```

После OOF inference:

1. измерить качество каждого эксперта;
2. построить error-correlation matrix;
3. проверить PCA/effective rank;
4. обучить простой RF baseline;
5. обучить небольшую META-CNN;
6. оценить прирост AP.

Только если эксперты дают действительно независимый сигнал — расширять банк до 20–30 моделей.

## 12. Главная гипотеза

> **TSS detection можно представить не только как sparse binary classification, а как задачу пространственной локализации, где окружающие нуклеотиды учатся указывать направление и расстояние до TSS.**

Один TSS вместо единственного positive nucleotide создаёт множество геометрически осмысленных targets вокруг себя.

Главная ставка — на изменение формы supervision:

```text
было:
DNA → TSS / not TSS

стало:
DNA → signed distance до TSS
     ↓
multiscale geometric representation
     ↓
exact TSS prediction
```

## 13. Связь с предыдущими экспериментами

Идея родственна coarse→fine / regional-head подходу:

```text
region → exact TSS
```

Но ключевое отличие:

```text
regional target:
"есть ли TSS где-то рядом?"

signed-distance target:
"где именно относительно меня расположен TSS?"
```

Новый supervisory signal содержит:

- направление;
- расстояние;
- центр;
- uncertainty распределения;
- согласие разных пространственных масштабов.

## 14. Что считать успехом

Не обещать конкретный AP заранее.

Проверять поэтапно:

```text
1. Геометрические эксперты вообще учат distance?
2. Их OOF ошибки различаются?
3. RF/meta-linear baseline использует эти признаки?
4. META-CNN улучшает exact TSS AP?
5. Прирост стабилен по contigs?
```

Рабочая шкала интерпретации:

```text
< +0.005 AP   → слабый эффект
+0.01–0.02    → полезный сигнал
+0.03–0.05    → очень серьёзный результат
> +0.05       → потенциально фундаментально удачная ветка
```

Это не прогноз, а критерий интерпретации.

# Коротко

```text
DNA
↓
банк независимых signed-distance CNN
(R / RF / kernels / dilation различаются)
↓
OOF multiscale probability tracks
↓
анализ diversity и ошибок
↓
отбор полезных экспертов
↓
META-CNN
↓
exact presence + intensity
↓
AP + Spearman
```

Главная идея: **не ещё одна CNN, а более информативная геометрическая supervised-задача + stacking пространственно разных экспертов**.
