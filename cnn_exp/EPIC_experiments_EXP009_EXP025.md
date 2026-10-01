# EPIC — журнал экспериментов EXP009 → EXP025

> Период: серия экспериментов по развитию BPE/CNN-архитектуры для задачи EPIC.  
> Основной текущий фокус: построить сильный **Global detector**, который умеет находить небольшие регионы, содержащие TSS, после чего Local CNN сможет уточнять точную координату на уровне нуклеотида.

---

## 0. Важное замечание по метрикам

Официальный EPIC scorer получает **один score на каждую position-strand** и считает две метрики:

1. **Average Precision (AP)** против бинарной цели `count > 0`.
2. **EPIC Spearman** — Pearson-корреляция dense ranks на позициях, где `target > 0`.

В старых двухголовых моделях мы диагностически считали три разных candidate-score:

- `presence_probability`
- `intensity_prediction`
- `expected_signal`

Это **не три официальных EPIC-метрики**.  
Это три способа сформировать submission score и прогнать его через официальные метрики.

Практически:

- Presence-head в первую очередь интересен по **AP**.
- Intensity-head — по **Spearman**.
- `expected_signal` — только дополнительная fusion-гипотеза.

Начиная с Regional Classifier основной критерий для Global:

- **regional AP**
- **AP lift относительно prevalence**
- **TSS recall при top X% регионов**
- **доля регионов, которую нужно оставить для 90/95/99% TSS recall**

Главная практическая цель Global:

> оставить небольшую долю генома/регионов, но сохранить почти все TSS.

Идеальный ориентир:

```text
top 10% regions → 90–95% TSS recall
```

---

# 1. EXP009 — BPE400, hard split heads

Файл:

```text
CNN_EXP009_BPE400_SPLIT_HEADS_PRETOKENIZED.ipynb
```

## Гипотеза

Разделить задачи Presence и Intensity жёстко:

- Presence получает Local + Global.
- Intensity получает только Local.

Идея:

> Global-контекст может помогать определять наличие TSS, но мешать точной оценке интенсивности.

Одновременно использован более агрессивный `BPE400`.

## Архитектура

```text
Local + Global → Presence
Local          → Intensity
```

## Результат @10k

| score | AP | Spearman |
|---|---:|---:|
| Presence | **0.031791** | 0.170838 |
| Intensity | 0.019145 | **0.182697** |
| Expected | 0.022348 | 0.181991 |

## Вывод

Результат оказался хуже BPE200 shared baseline.

Но эксперимент был **конфундирован**:

- изменили BPE `200 → 400`;
- одновременно изменили архитектуру fusion.

Поэтому нельзя было точно сказать, что именно ухудшило результат.

---

# 2. EXP010 — BPE400, soft-gated intensity

Файл:

```text
CNN_EXP010_BPE400_SOFT_GATED_INTENSITY.ipynb
```

## Гипотеза

Вместо жёсткого отключения Global от Intensity дать модели самой решать, насколько Global полезен:

```text
Intensity = Local + gate × Global
```

Начальный `gate ≈ 0.05`.

## Результат @10k

| score | AP | Spearman |
|---|---:|---:|
| Presence | **0.033854** | 0.173113 |
| Intensity | 0.020181 | **0.189454** |
| Expected | 0.025470 | 0.187093 |

## Вывод

Soft gate оказался лучше hard split по Intensity Spearman, но Presence AP всё ещё хуже BPE200 shared.

Главный вывод:

> идея learnable-gate разумна, но BPE400 сам по себе, вероятно, уже ухудшает representation.

---

# 3. EXP011 — BPE400 shared

Файл:

```text
CNN_EXP011_BPE400_SHARED_PRETOKENIZED.ipynb
```

## Гипотеза

Вернуться к обычному shared fusion и проверить **чистое влияние BPE400**.

## Результат @10k

| score | AP | Spearman |
|---|---:|---:|
| Presence | **0.031657** | 0.170865 |
| Intensity | 0.002530 | **0.174834** |
| Expected | 0.030214 | 0.180681 |

## Вывод

BPE400 shared оказался хуже BPE200.

Это уже сильный сигнал:

> проблема EXP009/010 была не только в heads/fusion — BPE400 сам по себе оказался слишком агрессивным.

---

# 4. EXP012 — BPE300 shared

Файл:

```text
CNN_EXP012_BPE300_SHARED_PRETOKENIZED.ipynb
```

## Гипотеза

Найти компромисс между BPE200 и BPE400.

## Результат @10k

| score | AP | Spearman |
|---|---:|---:|
| Presence | **0.034933** | 0.176928 |
| Intensity | 0.003858 | **0.189627** |
| Expected | 0.034688 | 0.190247 |

## Вывод

BPE300 оказался очень сильным для Intensity:

```text
Intensity Spearman = 0.189627
```

Это был лучший Intensity Spearman в merge-sweep.

Presence AP при этом немного уступал BPE200.

---

# 5. EXP013 — BPE150 shared

Файл:

```text
CNN_EXP013_BPE150_SHARED_PRETOKENIZED.ipynb
```

## Гипотеза

Проверить, не является ли более слабое сжатие выгодным.

## Результат @10k

| score | AP | Spearman |
|---|---:|---:|
| Presence | **0.031931** | 0.166172 |
| Intensity | 0.001100 | **0.176206** |
| Expected | 0.031613 | 0.180357 |

## Вывод

BPE150 оказался хуже BPE200 и BPE300.

Дополнительно в EXP013 был технический checkpoint-assert bug; validation пришлось запускать после ручного отключения ошибочного assert.

---

# 6. EXP014 — BPE250 shared

Файл:

```text
CNN_EXP014_BPE250_SHARED_PRETOKENIZED.ipynb
```

## Гипотеза

Проверить промежуточную точку между BPE200 и BPE300.

## Результат @10k

| score | AP | Spearman |
|---|---:|---:|
| Presence | **0.033198** | 0.175975 |
| Intensity | 0.001121 | **0.182397** |
| Expected | 0.031747 | 0.186464 |

## Итог merge-sweep

| BPE | Presence AP | Intensity Spearman |
|---:|---:|---:|
| 100 | 0.032755 | 0.184274 |
| 150 | 0.031931 | 0.176206 |
| **200** | **0.035511** | 0.180405 |
| 250 | 0.033198 | 0.182397 |
| **300** | 0.034933 | **0.189627** |
| 400 | 0.031657 | 0.174834 |

## Вывод

Для Presence лучшим оказался **BPE200**.  
Для Intensity — **BPE300**.

Для дальнейших Global-экспериментов был выбран BPE200 как наиболее сильный вариант для detection/ranking positives.

---

# 7. EXP015 — BPE200 Local Motif Multi-Scale, RF≈59 bp

Файл:

```text
CNN_EXP015_BPE200_LOCAL_MOTIF_MULTISCALE.ipynb
```

## Гипотеза

Сильно сузить Local CNN и заставить её отвечать почти только за локальные promoter motifs.

Архитектура Local:

```text
stem k7
↓
residual k7
↓
residual k5
↓
multi-scale:
k7,d1
k5,d2
k3,d4
k3,d8
↓
fusion
```

Максимальный RF Local:

```text
≈59 bp
```

Global BPE200 и shared fusion оставались.

## Результат @10k

| score | AP | Spearman |
|---|---:|---:|
| Presence | **0.006859** | 0.139442 |
| Intensity | 0.001379 | **0.128867** |
| Expected | 0.007146 | 0.141256 |

## Сравнение с BPE200 shared baseline

Presence AP:

```text
0.035511 → 0.006859
≈ -80.7%
```

Intensity Spearman:

```text
0.180405 → 0.128867
≈ -28.6%
```

## Вывод

Очень важный отрицательный эксперимент.

> RF≈59 bp для Local слишком мал.

Local CNN должна видеть **сотни bp**, а не только непосредственный core motif.

Это подтвердило, что старый Local backbone действительно несёт существенную contextual information.

---

# 8. EXP016 — Global-only localization

Файл:

```text
CNN_EXP016_BPE200_GLOBAL_ONLY_LOCALIZATION.ipynb
```

## Гипотеза

Полностью убрать Local и отдельно измерить:

> способен ли один Global BPE encoder локализовать TSS?

## Результат nucleotide-level

```text
AP = 0.005700
AP lift ≈ 8.50×
Spearman ≈ 0.10798
```

## Spatial AP

| tolerance/window | prevalence | AP | AP lift |
|---:|---:|---:|---:|
| 1 | 0.000671 | 0.005700 | 8.50× |
| 8 | 0.003660 | 0.020023 | 5.47× |
| 16 | 0.006201 | 0.028707 | 4.63× |
| 32 | 0.010578 | 0.040746 | 3.85× |
| 64 | 0.018189 | 0.057756 | 3.18× |
| 128 | 0.031453 | 0.082340 | 2.62× |
| 256 | 0.054255 | 0.118682 | 2.19× |

## Peak alignment

Для истинных TSS искался strongest peak в ±384 bp.

```text
median abs offset = 119 bp
p75               = 243 bp
p90               = 335 bp
```

Hit-rate:

| tolerance | hit |
|---:|---:|
| ±4 | 5.17% |
| ±8 | 9.67% |
| ±16 | 16.69% |
| ±32 | 25.70% |
| ±64 | 36.42% |
| ±128 | 51.71% |
| ±256 | 77.21% |

## Важная интерпретация

`77% within ±256` **не означает**, что Global обнаруживает 77% TSS.

Это conditioned metric:

> уже известен истинный TSS, а затем ищется strongest peak внутри ±384.

Случайный максимум уже попадал бы в ±256 примерно в 66.7% случаев.

## Вывод

Global действительно имеет spatial signal, но он слабый.

Главная проблема:

> Global умеет грубо чувствовать область, но недостаточно хорошо ранжирует истинные TSS-регионы.

---

# 9. EXP017 — Coarse-to-Fine Dilated Targets

Файл:

```text
CNN_EXP017_BPE200_GLOBAL_COARSE_TO_FINE.ipynb
```

## Гипотеза

Обучать несколько Global heads на расширенных бинарных targets:

```text
±256 bp
±128 bp
±64 bp
±32 bp
```

Идея:

> крупная head сначала учится видеть область, более мелкая — уточнять.

## Результат

Exact nucleotide AP всех heads остался почти одинаковым:

```text
~0.0054–0.0055
```

Coarse targets:

| radius | prevalence | AP | lift |
|---:|---:|---:|---:|
| 256 | 0.099647 | 0.147920 | 1.48× |
| 128 | 0.057711 | 0.104849 | 1.82× |
| 64 | 0.033085 | 0.074240 | 2.24× |
| 32 | 0.019011 | 0.052925 | 2.78× |

Peak alignment практически не изменился:

```text
median ≈ 116–118 bp
p75    ≈ 235–236 bp
p90    ≈ 329–331 bp
hit±256 ≈ 78.4–78.6%
```

## Вывод

Просто дать разным 1×1 heads targets разной ширины оказалось недостаточно.

> Heads выучили почти один и тот же spatial signal.

Нужна была архитектура, которая **физически предсказывает регионы**, а не просто размытые per-bp targets.

---

# 10. EXP018 — True Regional Classifier, target=1024 bp

Файлы:

```text
CNN_EXP018_BPE200_GLOBAL_REGIONAL_CLASSIFIER.ipynb
CNN_EXP018_BPE200_GLOBAL_REGIONAL_CLASSIFIER_DML_FIXED.ipynb
```

## Гипотеза

Перейти от per-bp heads к настоящему regional classifier.

Target 1024 bp делится на неперекрывающиеся регионы:

```text
4 × 256 bp
8 × 128 bp
16 × 64 bp
32 × 32 bp
```

Для каждого региона:

```text
target = 1, если внутри есть хотя бы один TSS
```

## Regional pooling

Изначально использовался max-pooling, но DirectML backward падал:

```text
RuntimeError: DirectML scatter doesn't allow partially modified dimensions...
```

Поэтому сделали DirectML-safe pooling:

```text
mean(features)
+
RMS(features)
```

и затем classifier:

```text
64 → 32 → 1
```

## Regional AP

| region | prevalence | AP | AP lift |
|---:|---:|---:|---:|
| 256 | 6.18% | 0.200576 | 3.25× |
| 128 | 3.43% | 0.143506 | 4.18× |
| 64 | 1.92% | 0.104440 | 5.43× |
| 32 | 1.09% | **0.074943** | **6.86×** |

## TSS recall at top region fractions

| selected | 256 | 128 | 64 | 32 |
|---:|---:|---:|---:|---:|
| 1% | 18.93% | 20.94% | 22.30% | **23.04%** |
| 2% | 26.41% | 28.67% | 30.46% | **31.23%** |
| 5% | 39.53% | 42.38% | 44.82% | **45.28%** |
| 10% | 53.42% | 56.34% | 58.02% | **59.31%** |
| 20% | 69.21% | 71.78% | 73.64% | **74.70%** |
| 50% | 89.64% | 91.40% | 92.62% | **93.38%** |

Для 95% TSS recall:

| region | fraction needed |
|---:|---:|
| 256 | 64.77% |
| 128 | 60.48% |
| 64 | 58.19% |
| 32 | **55.49%** |

## Вывод

32-bp head оказался лучшим coarse detector по:

- AP lift;
- TSS recall при одинаковом selection budget;
- доле пространства для заданного recall.

Но Global всё ещё слаб:

```text
95% TSS recall → надо оставить ~55.5% регионов
```

---

# 11. EXP019 — True Regional Classifier, target 1024 → 2048 bp

Файл:

```text
CNN_EXP019_BPE200_GLOBAL_REGIONAL_2048.ipynb
```

## Гипотеза

Увеличить **центральный supervised target**:

```text
1024 → 2048 bp
```

При этом flank оставить:

```text
±4096 bp
```

Важно:

```text
EXP018 input = 4096 + 1024 + 4096 = 9216 bp
EXP019 input = 4096 + 2048 + 4096 = 10240 bp
```

То есть total context увеличился всего ~11%; основное изменение — вдвое больше supervised positions на sample.

## Результат

| region | EXP018 AP | EXP019 AP | growth |
|---:|---:|---:|---:|
| 256 | 0.200576 | 0.246493 | +22.9% |
| 128 | 0.143506 | 0.182936 | +27.5% |
| 64 | 0.104440 | 0.135343 | +29.6% |
| **32** | **0.074943** | **0.098862** | **+31.9%** |

32-bp AP lift:

```text
6.86× → 9.05×
```

## TSS recall, 32 bp

```text
top 10%: 59.31% → 64.60%
top 20%: 74.70% → 78.61%
top 50%: 93.38% → 94.74%
```

Для 95% TSS recall:

```text
55.49% → 50.99% regions
```

## Вывод

Очень сильный положительный эксперимент.

> Увеличение target 1024 → 2048 сильно помогло всем regional heads.

Это показало, что модели явно выгодно получать больше supervised positions за один sampled window.

---

# 12. EXP020 — target=2048, flank ±6144

Файл:

```text
CNN_EXP020_BPE200_GLOBAL_REGIONAL_2048_FLANK6144.ipynb
```

## Гипотеза

После успеха target=2048 увеличить уже **внешний genomic context**:

```text
flank ±4096 → ±6144
```

Итого:

```text
2048 target + 2×6144 flank = 14336 bp input
```

Архитектура осталась прежней.

## BPE/RF диагностика

На 14336 bp:

```text
median BPE tokens ≈ 4273.5
p95               ≈ 4381
max               ≈ 4408
```

Текущий Global RF:

```text
4093 BPE tokens
```

Получилось:

```text
fraction <= RF = 0
fraction >  RF = 1
```

То есть все sampled windows были немного длиннее RF.

## Результат

| region | EXP019 AP | EXP020 AP |
|---:|---:|---:|
| 256 | 0.246493 | 0.244332 |
| 128 | 0.182936 | 0.182128 |
| 64 | 0.135343 | 0.136020 |
| **32** | **0.098862** | **0.099950** |

32 bp:

```text
AP 0.09886 → 0.09995
≈ +1.1%
```

TSS recall:

```text
top 10%: 64.60% → 65.06%
top 20%: 78.61% → 79.12%
```

Но для 95% recall:

```text
50.99% → 51.26%
```

то есть практически без улучшения.

## Вывод

Внешний context вышел на **плато**.

EXP020 стал нашим текущим лучшим baseline:

```text
BPE200
target=2048
flank=±6144
RF=4093
32-bp AP=0.099950
```

---

# 13. EXP021 — RF 4093 → 8189

Файл:

```text
CNN_EXP021_BPE200_GLOBAL_REGIONAL_RF8189.ipynb
```

## Гипотеза

Раз все 14.3-kb windows немного длиннее RF, увеличить receptive field.

Добавили один residual block:

```text
dilation=1024
```

Получилось:

```text
RF:
4093 → 8189 BPE tokens
```

Контекст при этом не менялся.

## Результат

| region | RF4093 | RF8189 | delta |
|---:|---:|---:|---:|
| 256 | 0.244332 | 0.229980 | -5.9% |
| 128 | 0.182128 | 0.168576 | -7.4% |
| 64 | 0.136020 | 0.123798 | -9.0% |
| **32** | **0.099950** | **0.090216** | **-9.7%** |

32-bp TSS recall:

```text
top10%: 65.06% → 62.86%
top20%: 79.12% → 77.43%
```

95% recall:

```text
51.26% → 52.91% regions
```

Training EMA тоже ухудшился:

```text
EXP020 0.8229
EXP021 0.8400
```

## Вывод

Текущий RF=4093 не выглядел главным bottleneck.

Простой большой jump до dilation=1024 ухудшил и train, и validation.

---

# 14. EXP022 — Global channels 32 → 64

Файл:

```text
CNN_EXP022_BPE200_GLOBAL_REGIONAL_CH64.ipynb
```

## Гипотеза

Возможно, контекста достаточно, но 32 feature channels недостаточно для кодирования информации.

Меняем:

```text
Global channels:
32 → 64
```

RF и context остаются как в EXP020.

Regional pooling:

```text
mean 64 + RMS 64 → 128 features
```

Head hidden оставлен 32, чтобы проверять именно capacity encoder.

## Результат

| region | 32 ch | 64 ch | delta |
|---:|---:|---:|---:|
| 256 | 0.244332 | 0.237568 | -2.8% |
| 128 | 0.182128 | 0.178143 | -2.2% |
| 64 | 0.136020 | 0.133175 | -2.1% |
| **32** | **0.099950** | **0.098753** | **-1.2%** |

TSS recall top10%:

```text
65.06% → 63.70%
```

95% recall:

```text
51.26% → 52.00%
```

Но training EMA улучшился:

```text
0.8229 → 0.8099
```

## Вывод

Более широкая сеть лучше подгоняет training objective, но не улучшает holdout ranking.

Это намекает:

> bottleneck не просто в количестве каналов.

---

# 15. EXP023 — BPE Multi-Scale Tower вместо backbone

Файл:

```text
CNN_EXP023_BPE200_GLOBAL_MULTISCALE_TOWER.ipynb
```

## Гипотеза

Попробовать идею multi-scale, вдохновлённую EPIC Solution.

Архитектура:

```text
BPE tokens
↓
32-ch stem
├─ fine branch: d=1,2
└─ tower:
   ×4   → 64 ch
   ×16  → 96 ch
   ×64  → 128 ch
   ×256 → 192 ch
↓
каждый tower level → project to 32
↓
upsample
↓
additive fusion
```

Deepest tower RF:

```text
≈6121 BPE tokens
```

## Важное отличие от EXP020

EXP020 имел сильный backbone:

```text
d=1,2,4,8,16,32,64,128,256,512
```

В EXP023 он был фактически заменён на:

```text
fine branch d=1,2 + tower
```

## Результат

| region | EXP020 | EXP023 |
|---:|---:|---:|
| 256 | 0.244332 | 0.184242 |
| 128 | 0.182128 | 0.130960 |
| 64 | 0.136020 | 0.093341 |
| **32** | **0.099950** | **0.066074** |

32-bp AP:

```text
-33.9%
```

Top10% TSS recall:

```text
65.06% → 57.36%
```

95% recall:

```text
51.26% → 54.25%
```

Training EMA:

```text
0.8229 → 0.8515
```

## Вывод

Провал не доказывает, что multi-scale плох.

Главная ошибка эксперимента:

> tower заменил уже работающий strong backbone.

---

# 16. EXP024 — Strong Backbone + Auxiliary Multi-Scale Tower

Файл:

```text
CNN_EXP024_BPE200_STRONG_BACKBONE_PLUS_MULTISCALE_TOWER.ipynb
```

## Гипотеза

Сохранить EXP020 backbone полностью и добавить Tower **параллельно**.

```text
                ┌─ strong EXP020 backbone ───────┐
BPE stem ───────┤                                ├→ fusion
                └─ ×4→×16→×64→×256 tower ───────┘
```

Strong branch:

```text
32 channels
d=1...512
RF=4093
```

Tower:

```text
64 → 96 → 128 → 192 ch
RF≈6121
```

Fusion:

```text
fused = strong + tower_sum
```

## Результат

| region | EXP020 | EXP024 |
|---:|---:|---:|
| 256 | 0.244332 | 0.204659 |
| 128 | 0.182128 | 0.144983 |
| 64 | 0.136020 | 0.104632 |
| **32** | **0.099950** | **0.074982** |

32-bp AP:

```text
≈ -25%
```

Top10% TSS recall:

```text
65.06% → 59.68%
```

95% recall:

```text
51.26% → 54.19%
```

Training EMA:

```text
EXP020 0.8229
EXP024 0.8776
```

## Вывод

Даже при сохранённом backbone принудительное:

```text
strong + tower
```

ухудшило и train, и validation.

Главная новая гипотеза:

> tower с random-init с самого начала загрязняет хорошее strong representation.

Нужно дать модели возможность самой решать, какие tower scales использовать.

---

# 17. EXP025 — Gated Multi-Scale Tower

Файл:

```text
CNN_EXP025_BPE200_GATED_MULTISCALE_TOWER.ipynb
```

## Статус

**Текущий эксперимент. Результаты ещё не зафиксированы в этой заметке.**

## Гипотеза

Strong EXP020 backbone остаётся.

Каждый scale Tower получает собственный learnable gate:

```text
fused =
    strong
    + g4   × tower_x4
    + g16  × tower_x16
    + g64  × tower_x64
    + g256 × tower_x256
```

Начальные значения:

```text
g4 = g16 = g64 = g256 = 0.01
```

То есть в начале:

```text
fused ≈ strong
```

и модель стартует почти как EXP020.

## Почему gates сделаны unconstrained

Не sigmoid, а обычные trainable scalar parameters.

Это позволяет:

```text
gate ≈ 0  → игнорировать scale
gate > 0  → добавлять scale
gate < 0  → использовать scale как подавляющий сигнал
```

И избегает очень маленького gradient у sigmoid, инициализированного около 0.01.

## Что логируется

Каждые 100 steps:

```text
g4
g16
g64
g256
```

В checkpoint сохраняются final gate values.

## Что хотим узнать

### Сценарий A

```text
gates растут
AP растёт
```

Tower полезен, проблема EXP024 была в forced fusion.

### Сценарий B

```text
gates остаются ≈0
```

Strong backbone сам по себе лучше, а current BPE-token tower пользы почти не даёт.

### Сценарий C

Например:

```text
g4   → заметно растёт
g16  → растёт
g64  → ~0
g256 → ~0
```

Тогда получим прямую диагностику:

> какие scales полезны именно для поиска TSS-region.

---

# 18. Главная линия развития EXP009 → EXP025

Эволюцию можно представить так:

```text
EXP009–014
│
├─ ищем лучший BPE/fusion
│
└→ BPE200 лучший для Presence
    BPE300 лучший для Intensity
        │
        ↓
EXP015
│
├─ очень узкий Local RF=59
└→ сильный провал
    ⇒ Local нуждается в сотнях bp контекста
        │
        ↓
EXP016
│
├─ Global-only
└→ spatial signal есть, но слабый
        │
        ↓
EXP017
│
├─ dilated coarse targets
└→ heads почти одинаковые
    ⇒ нужен настоящий regional objective
        │
        ↓
EXP018
│
├─ true regional classifier
└→ 32bp region лучший по AP lift / recall budget
        │
        ↓
EXP019
│
├─ target 1024 → 2048
└→ +31.9% AP32
    ⇒ очень сильный положительный результат
        │
        ↓
EXP020
│
├─ flank ±4096 → ±6144
└→ AP почти плато
        │
        ├→ EXP021 RF 4093→8189
        │   └→ хуже
        │
        ├→ EXP022 32→64 channels
        │   └→ почти без улучшения / хуже
        │
        └→ EXP023/024 multi-scale tower
            └→ forced tower ухудшает сеть
                │
                ↓
EXP025
│
└─ learnable per-scale gates
   ⇒ Tower должен сам доказать полезность
```

---

# 19. Текущий лучший Global baseline

На данный момент лучший проверенный вариант:

```text
EXP020
```

Конфигурация:

```text
BPE:            200
target:         2048 bp
flank:          ±6144 bp
total input:    14336 bp
Global channels: 32
RF:             4093 BPE tokens
regional heads: 256 / 128 / 64 / 32 bp
pooling:        mean + RMS
steps:          10k
```

32-bp detector:

```text
regional prevalence = 0.010929
AP                  = 0.099950
AP lift             = 9.145554×
```

TSS recall:

```text
top 1%  regions → 27.79%
top 2%          → 36.80%
top 5%          → 51.46%
top 10%         → 65.06%
top 20%         → 79.12%
top 50%         → 94.57%
```

Для target recall:

```text
90% TSS recall → 36.53% regions
95% TSS recall → 51.26% regions
99% TSS recall → 75.89% regions
```

Главная проблема:

> Global пока слишком слаб как фильтр: чтобы сохранить 95% TSS, приходится оставить примерно половину 32-bp регионов.

---

# 20. Что мы уже исключили как простое решение

По серии ablation видно:

```text
больше BPE merges     → не гарантирует рост
очень узкий Local RF  → плохо
просто dilated targets→ недостаточно
больше external flank → почти плато
больше RF             → хуже
32→64 channels        → не помогло
tower вместо backbone → плохо
forced tower + strong → плохо
```

То есть сейчас проблема уже не выглядит как:

```text
"нужно просто больше контекста"
```

или:

```text
"нужно просто больше параметров"
```

---

# 21. Главная новая гипотеза: мы упираемся в обучение

Очень вероятно, что 10k steps сейчас — это прежде всего **architecture screening budget**, а не полноценное обучение.

Исторические примеры:

## Старый Local CNN

```text
10k  AP ≈ 0.0274
100k AP ≈ 0.0512

рост ≈ ×1.87
```

## BPE200 shared

```text
10k  Presence AP ≈ 0.0355
100k Presence AP ≈ 0.0559

рост ≈ ×1.58
```

Поэтому для EXP020 нельзя исключать:

```text
10k AP32 = 0.10
100k AP32 → существенно выше
```

Но рост не будет линейным:

```text
10k → 100k может сильно помочь
100k → 1M не означает ещё ×2
```

Learning curve должна выйти на плато.

---

# 22. Почему нам не нужен AP≈0.95

Для EPIC это нереалистичная цель.

Даже сильная EPIC Solution CNN имеет примерно:

```text
base-level AP ≈ 0.1317
Spearman      ≈ 0.391
```

При этом сама задача очень шумная и экстремально несбалансированная.

Для нашего Global coarse detector гораздо полезнее operational metric:

```text
top X% regions → Y% TSS recall
```

Главная цель:

```text
top 10% regions → 90–95% TSS recall
```

Если Global достигнет этого, Local CNN сможет работать только по небольшой части пространства и уточнять точную TSS-coordinate.

---

# 23. Как будет выглядеть итоговая Global + Local система

Целевая архитектурная идея:

```text
DNA
│
├→ Global BPE CNN
│      ↓
│   ищет "стог"
│   (регион, где вероятен TSS)
│
└→ Local nucleotide CNN
       ↓
    внутри выбранной области
    ищет точную "иголку"
       ↓
    base-resolution TSS score
```

Global отвечает:

```text
"В каком 32-bp регионе есть TSS?"
```

Local отвечает:

```text
"Какой именно nucleotide является TSS?"
```

В будущем fusion может выглядеть так:

```text
Local nucleotide features
+
Global contextual features
↓
per-nucleotide head
```

---

# 24. Следующие рациональные шаги

## 1. Закончить EXP025

Смотреть:

```text
AP32
AP lift
top10% TSS recall
top20% TSS recall
fraction for 95% recall

g4
g16
g64
g256
```

## 2. Длинное обучение EXP020

Сохранить checkpoints:

```text
10k
25k
50k
75k
100k
```

и построить learning curve.

Это позволит отделить:

```text
architecture bottleneck
от
undertraining
```

## 3. BPE100 — дополнительный ablation

Идея разумная:

> меньше BPE merges → больше token positions → лучше локальная геометрия.

Но ожидается скорее тонкая настройка, чем прорыв, потому что BPE100 и BPE200 относительно близки.

## 4. Если gated Tower не помогает

Не продолжать бесконечно усложнять BPE-token tower.

Следующая архитектурная идея:

> строить multi-scale уже после отображения Global features обратно в bp-coordinate space.

Тогда `×4 / ×16 / ×64` снова будут физически соответствовать стабильным genomic scales.

---

# 25. Краткий итог

Главный результат всей серии EXP009–EXP025:

> Самым сильным улучшением Global оказался не рост параметров и не рост RF, а правильная постановка regional objective + увеличение target supervision 1024 → 2048.

Лучший проверенный Global:

```text
EXP020
AP32 ≈ 0.100
top10% regions → 65.1% TSS recall
95% TSS recall → 51.3% regions
```

Мы уже нашли несколько тупиков:

```text
слишком узкий Local
слишком большой RF
простое расширение channels
forced multi-scale fusion
```

Сейчас две наиболее важные неизвестные:

1. **насколько сильно EXP020 недообучен;**
2. **может ли learnable gated Tower извлечь пользу из multi-scale context без разрушения strong backbone.**

Именно EXP025 + длинная learning curve EXP020 должны дать ответ на эти два вопроса.
