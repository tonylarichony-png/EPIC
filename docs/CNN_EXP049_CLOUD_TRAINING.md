# CNN-EXP-049: обучение каскада в Colab или Kaggle

## Что реализовано

Облачный runner разделён на независимо возобновляемые стадии:

```text
inspect
  ↓
smoke
  ↓
train-a fold 0 → cache-a fold 0 ┐
train-a fold 1 → cache-a fold 1 ├─ OOF R16 cache train
train-a fold 2 → cache-a fold 2 ┘
  ↓
evaluate-a (OOF regional gate)
  ↓
train-b
  ↓
train-a --fold -1
  ↓
cache-a --fold -1 --cache-part validation
  ↓
evaluate-b
```

Код не зависит от DirectML и использует стандартный PyTorch CUDA.

## Какую платформу выбрать

### Kaggle

Подходит для длинного воспроизводимого запуска с GPU и неизменяемым входным
Dataset. Временный проект и R16-cache находятся в `/tmp`, чтобы не раздувать
notebook output. Последняя ячейка создаёт единственный переносимый файл
`exp049_run.zip`; в следующей сессии подключите предыдущий notebook output как
input и укажите его путь в `RESUME_BUNDLE`.

Новая VM всё равно заново подключает read-only Inputs — это поведение Kaggle.
Notebook не копирует распакованный геном: он создаёт символическую ссылку на
Dataset. Подготовленный split (~несколько MiB), checkpoints и R16-cache входят в
resume-архив, поэтому между сессиями пересчитывать их не нужно.

### Google Colab

Удобнее для ручного запуска и checkpoint resume через Google Drive. Но большой
R16-cache нельзя эффективно читать как memmap прямо с Drive. Поэтому notebook
работает в `/content/exp049_run` на SSD VM, а последняя ячейка последовательно
архивирует состояние в `/content/drive/MyDrive/EPIC/exp049_run.zip`. При новой
сессии `SETUP` автоматически восстанавливает этот архив обратно на локальный SSD.

Для первого запуска рекомендуется Kaggle. Для серии коротких интерактивных
запусков с частыми перезапусками — Colab + Drive.

## Kaggle: пошагово

1. Создайте три private Dataset и загрузите соответственно
   `epic-exp049-code.zip`, `jaNemVect1.1.tgz` и `exp047-risk-cache.zip`.
   Загружайте именно архивы; при подключении Dataset Kaggle может автоматически
   показать уже распакованное содержимое и не сохранить исходный `.zip` в
   `/kaggle/input`. Notebook поддерживает оба варианта.
2. Загрузите `CNN_EXP049_CLOUD_TRAIN.ipynb` как Kaggle Notebook, подключите эти
   Dataset как inputs и включите GPU accelerator.
3. Если slug Dataset отличается от примера, исправьте три пути в первой ячейке.
4. Выполните `SETUP`, `INSPECT`, `SMOKE`, затем обучите один `FOLD_TO_RUN`.
5. После остановки или завершения fold выполните последнюю ячейку: она создаст
   `/kaggle/working/exp049_run.zip`. Сохраните notebook version вместе с output.
6. Для следующего fold подключите предыдущий notebook output как input,
   исправьте `RESUME_BUNDLE`, поменяйте `FOLD_TO_RUN` и повторите. Не начинайте B,
   пока `evaluate-a` не увидит все три корректных OOF fold.

## Colab: пошагово

1. Положите три входных файла и notebook в `MyDrive/EPIC`.
2. Откройте notebook в Colab, выберите GPU runtime и выполните первые ячейки.
3. Обучайте один fold за сессию. После каждой сессии обязательно запускайте
   `EXPORT / RESUME`: состояние атомарно обновится в
   `MyDrive/EPIC/exp049_run.zip`.
4. В новой VM `SETUP` автоматически распакует этот архив на локальный SSD.
   Не удаляйте resume-архив, пока не сохранены финальные метрики и checkpoints.

## Входные файлы

Нужны два обязательных и один опциональный объект:

1. `epic-exp049-code.zip` — переносимый срез репозитория;
2. `jaNemVect1.1.tgz` — исходный архив Nematostella, 303,967,377 bytes.
3. `exp047-risk-cache.zip` — 22.8 MB, точные frozen risk-strata веса EXP047.

Без третьего объекта runner использует базовый natural-profile BCE и явно
маркирует запуск как архитектурный pilot. С risk-cache сохраняется risk-focused
negative weighting EXP047-B.

Локальный MD5 исходного архива должен быть:

```text
733dccb6393a3c07b6e0d7ffaed3686f
```

Не включайте в bundle старые `artifacts`, другие виды и локальное `.venv`.

## Быстрый запуск

Откройте:

```text
notebooks/experiments/CNN_experiments/CNN_EXP049_CLOUD_TRAIN.ipynb
```

В конфигурационной ячейке задайте пути к code bundle, архиву данных и
персистентному run-dir. Затем выполните `SETUP`, `INSPECT` и `SMOKE`.

## Активный EXP049B joint pilot

После завершённого A-v1 checkpoint `step_020000.pt` используйте раздел
`EXP049B PAIRED END-TO-END PILOT` в notebook. Он запускает две сопоставимые
ветви по 20K шагов:

```text
control: zero regional condition → B
joint:   online A-v1 → FiLM → B, gradient обновляет A и B
```

Joint runner вычисляет Network-A condition непосредственно из выровненного
8192-bp окна. R16 cache здесь не используется, поскольку он разрывает градиент.
Обе ветви получают одинаковую инициализацию B, последовательность sampled tiles
и nucleotide loss. После каждой ветви немедленно выполните `export_joint()` и
Quick Save с outputs. Переносимый результат:

```text
/kaggle/working/exp049b_joint_pilot.zip
```

Полный контракт и правило решения записаны в
`docs/CNN_EXP049B_JOINT_PILOT.md`.

После успешного парного результата на 20K используйте следующий notebook-раздел
`EXP049B PAIRED CONTINUATION: 20K → 50K`. Он восстанавливает оба checkpoint и
Adam state, продолжает каждый arm ещё 30K шагов с новым cosine LR и сохраняет
результат в отдельные `continuation_50k.pt`. После каждого ограниченного по
времени или завершённого запуска снова вызывается `export_joint()`.

## Эквивалентные команды CLI

Все команды выполняются из корня распакованного проекта.

Проверка данных и CUDA:

```bash
python scripts/run_cnn_exp049_cloud.py inspect --folds 3
python scripts/run_cnn_exp049_cloud.py smoke --device cuda
```

OOF Network A:

```bash
python scripts/run_cnn_exp049_cloud.py train-a --fold 0 --folds 3 --device cuda
python scripts/run_cnn_exp049_cloud.py cache-a --fold 0 --folds 3 --device cuda
```

Повторите для `fold 1` и `fold 2`, используя один и тот же `--run-dir` и
`--cache-dir`. Runner запрещает случайно построить in-sample train-cache из A,
обученной на тех же contigs. После всех трёх fold проверьте OOF-качество A:

```bash
python scripts/run_cnn_exp049_cloud.py evaluate-a --folds 3
```

Команда сохраняет regional AP и, при пороге с 99.5% recall активных R16,
долю отсекаемых пустых регионов. Это диагностический gate для A, а не замена
итоговому nucleotide-level AP сети B.

Network B:

```bash
python scripts/run_cnn_exp049_cloud.py train-b \
  --device cuda \
  --film-blocks 2,4,6,8 \
  --b-batch 2 \
  --b-steps 50000 \
  --risk-cache /path/train_negative_stratum_u8.dat \
  --risk-manifest /path/risk_strata_manifest.json
```

Финальная A и validation-cache:

```bash
python scripts/run_cnn_exp049_cloud.py train-a --fold -1 --device cuda
python scripts/run_cnn_exp049_cloud.py cache-a \
  --fold -1 \
  --cache-part validation \
  --device cuda
```

Natural validation AP:

```bash
python scripts/run_cnn_exp049_cloud.py evaluate-b --device cuda --eval-batch 2
```

## Возобновление после лимита сессии

`train-a` и `train-b` атомарно сохраняют:

- веса модели;
- состояние AdamW;
- состояние OneCycleLR;
- текущий step;
- полный training contract.

Повтор той же команды с теми же `--a-steps` или `--b-steps` продолжит обучение.
Параметр `--max-minutes 300` завершает стадию с checkpoint примерно через пять
часов, не дожидаясь принудительного завершения облачной сессии.

Если лимит сработал внутри `train-a`, следующая команда `cache-a` намеренно
завершится ошибкой: повторно запустите ту же ячейку, чтобы A продолжила обучение.
Кэш строится только после достижения последнего шага.

При resume нельзя менять не только total steps (OneCycleLR зависит от полного
числа шагов), но и остальные параметры training contract. Для другого режима
создайте новый `run-dir`; runner также отвергает старый или несовместимый кэш.

## Оценка диска

R16 cache хранится в `float16`, отдельно для plus/minus, по 17 признаков на бин.
Для 12 используемых Nematostella contigs расчётный объём составляет около
`0.86 GiB` вместе с train и validation. Перед стартом B проверьте, что
в manifest присутствуют все девять train contigs и у каждого стоит `oof=true`.

## Контракт loss

Portable B использует class-balanced natural-profile BCE, within-R16 pairwise
loss, count loss и residual regularization. При переданном `--risk-cache`
negative weights дополнительно умножаются на зафиксированные EXP047-B strata
multipliers. Источник cache и полный loss contract сохраняются в checkpoint.

Без risk-cache запуск проверяет архитектуру A→FiLM-B, но не является строго
изолированной проверкой только архитектуры относительно EXP047-B.
