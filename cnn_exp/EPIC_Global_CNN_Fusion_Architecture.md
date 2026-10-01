# Архитектура Global CNN + Fusion в EPIC

## Общая схема

Текущая архитектура использует две ветки:

```text
DNA window
│
├── Local CNN
│   └── local_features: [B, 32, 1284]
│
└── Global BPE-400
    └── global_features: [B, 32, T]
```

После этого глобальные BPE-признаки переводятся обратно в координаты локального окна:

```text
global_features
      │
      └── gather по local_map
              │
              ▼
      global_local: [B, 32, 1284]
```

---

## Global BPE CNN

Вход глобальной ветки — окно длиной:

```text
GLOBAL_LENGTH = 9216 bp
```

DNA сначала представляется последовательностью BPE-токенов.

Для каждого токена используются:

- `embedding` — 16 признаков;
- `softmask fraction` — 1 признак;
- нормализованная длина токена — 1 признак.

Итого:

```text
18 признаков на токен
```

### Проекция

```text
[B, T, 18]
   │
transpose
   ▼
[B, 18, T]
   │
Conv1d 1×1: 18 → 32
GELU
   ▼
[B, 32, T]
```

### Dilated residual CNN

Далее идут 10 residual-блоков:

```text
dilation:
1
2
4
8
16
32
64
128
256
512
```

Каждый блок:

```text
x
│
├──────────── residual ─────────────┐
│                                   │
└→ Conv1d(k=3, dilation=d) → GELU   │
                 │                  │
                 └→ Conv1d(k=3) ────┤
                                    ▼
                                  + x
                                    │
                                  GELU
```

Выход глобальной ветки:

```text
global_features: [B, 32, T]
```

Теоретический receptive field этой BPE-CNN:

```text
4093 BPE tokens
```

Но физический контекст ограничен входным окном `9216 bp`.

---

## Перенос Global → base-resolution

BPE-токены имеют переменную длину, поэтому Global CNN работает не в координатах отдельных bp.

Для каждого bp локального окна заранее хранится:

```text
local_map[bp] → индекс BPE-токена
```

После Global CNN выполняется:

```python
global_local = torch.gather(
    global_features,
    2,
    gather_index,
)
```

Результат:

```text
global_local: [B, 32, 1284]
```

Теперь Local CNN и Global BPE CNN находятся в одной координатной системе.

---

# Fusion для Presence

Local и Global признаки объединяются по каналам:

```text
local_features       [B, 32, 1284]
global_local         [B, 32, 1284]
        │
        └──── concatenate ────┐
                              ▼
                       [B, 64, 1284]
                              │
                       Conv1d 1×1
                          64 → 32
                              │
                            GELU
                              ▼
                  presence_features
                       [B, 32, 1284]
                              │
                       Conv1d 1×1
                           32 → 1
                              ▼
                       Presence logits
```

То есть Presence получает полный контекст:

```text
Presence = f(Local, Global)
```

---

# Soft-gated Fusion для Intensity

В текущем эксперименте Global-контекст не отрезается полностью от Intensity.

Сначала глобальные признаки отдельно проецируются:

```text
global_local
[B, 32, 1284]
      │
Conv1d 1×1: 32 → 32
GELU
      ▼
global_for_intensity
```

Далее:

```text
intensity_features =
    local_features
    +
    gate × global_for_intensity
```

где:

```text
gate = sigmoid(trainable_parameter)
```

Начальное значение:

```text
gate = 0.05
```

Поэтому в начале обучения Intensity почти полностью локальная:

```text
~95% Local
+ небольшой Global residual
```

Но модель может сама увеличить или уменьшить вклад Global-контекста.

После этого:

```text
intensity_features
      │
Conv1d 1×1: 32 → 1
      ▼
Intensity prediction
```

---

# Итоговая архитектура

```text
                           ┌─────────────────────────────┐
DNA local ─→ Local CNN ───→│ local_features [32 ch]     │
                           └──────────┬──────────────────┘
                                      │
                                      │
DNA 9216 bp                           │
    │                                 │
    ▼                                 │
 BPE-400                              │
    │                                 │
Embedding + softmask + length         │
    │                                 │
    ▼                                 │
10× Dilated Residual CNN              │
    │                                 │
    ▼                                 │
global_features [32 ch]               │
    │                                 │
local_map / gather                    │
    │                                 │
    ▼                                 │
global_local [32 ch]                  │
    │                                 │
    ├──────────────┐                  │
    │              │                  │
    ▼              │                  ▼
concat(Local, Global)          Local + gate×Proj(Global)
    │                                 │
Conv1d 64→32                       Intensity head
GELU                                32→1
    │                                 │
Presence head                         ▼
32→1                              Intensity
    │
    ▼
Presence

Expected signal ≈ Presence × Intensity
```

## Идея архитектуры

- **Local CNN** отвечает за точные локальные последовательностные признаки вокруг позиции.
- **Global BPE CNN** добавляет широкий контекст на несколько тысяч bp.
- **Presence** получает Global-контекст полностью.
- **Intensity** использует Local как основной источник и получает Global только через обучаемый residual-gate.
- Такой дизайн позволяет не заставлять обе головы одинаково использовать широкий контекст.
