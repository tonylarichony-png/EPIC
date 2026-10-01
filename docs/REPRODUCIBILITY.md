---
type: guide
status: active
last_reviewed: 2026-10-01
tags:
  - ml/reproducibility
  - ml/public-repository
---

# Воспроизводимость и публичный репозиторий

## Что находится в GitHub

- Obsidian vault: постановка задачи, EDA, гипотезы, карточки экспериментов,
  решения и error analysis;
- все официальные `.ipynb`, включая сохранённые outputs умеренного размера;
- reusable Python-код в `src/ml_project/`;
- CLI/cloud runners в `scripts/`;
- тесты;
- небольшие PNG/SVG/CSV, необходимые для чтения отчётов.

## Что намеренно не публикуется

- исходные genome/annotation/count данные;
- `data/processed/`;
- checkpoints (`.pt`, `.pth`, `.ckpt`, `.safetensors`);
- memmap/cache (`.dat`), полные prediction vectors и локальные artifacts;
- Kaggle/Colab resume bundles и любые `.zip`/`.tgz`;
- `.env`, credentials и локальное состояние Obsidian.

Это предотвращает случайную публикацию больших или машинно-зависимых файлов.
Сводные метрики и графики, необходимые для проверки выводов, хранятся отдельно
в `assets/` и карточках экспериментов.

## Окружение

Рекомендуемый вариант:

```powershell
conda env create -f environment-eda.yml
conda activate epic-eda
python -m ipykernel install --user --name epic-eda --display-name "Python (epic-eda)"
```

`environment-eda-lock.yml` и `conda-eda-win-64.lock.txt` фиксируют более точное
состояние Windows-окружения. CUDA/PyTorch для CNN зависит от cloud runtime и не
закреплён в CPU EDA environment.

## Данные

Подготовьте исходные файлы по
[справочнику jaNemVect1.1](jaNemVect1.1_file_guide.md) и разместите их под
`data/raw/`. Эта директория игнорируется Git, кроме `.gitkeep`.

Основной split:

```text
contig_holdout_v1
train contigs:      9
validation contigs: 3
primary metric:     genome-wide nucleotide Average Precision
secondary metric:   EPIC Spearman on positive targets
```

Полный контракт находится в [03_validation.md](03_validation.md).

## Проверка кода

Из корня репозитория:

```powershell
conda activate epic-eda
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
```

В bash используйте `export PYTHONPATH="$PWD/src"`.

Тесты, зависящие от отсутствующих raw/processed данных, должны либо использовать
fixtures, либо явно пропускаться. Они не должны незаметно читать validation
labels при обучении.

## Воспроизведение feature-экспериментов

1. Выполните `notebooks/01_train_validation_test.ipynb`.
2. Проверьте `docs/03_validation.md` и сформированный split manifest.
3. Откройте нужный notebook из `notebooks/experiments/`.
4. Сверьте результат с карточкой `experiments/EXP-xxx.md`.

## Воспроизведение CNN-экспериментов

Крупные CNN-запуски требуют CUDA и полного набора данных. Для EXP049 используйте:

- [cloud notebook](../notebooks/experiments/CNN_experiments/CNN_EXP049_CLOUD_TRAIN.ipynb);
- [инструкцию Kaggle/Colab](CNN_EXP049_CLOUD_TRAINING.md);
- `scripts/run_cnn_exp049_cloud.py`;
- `scripts/run_cnn_exp049a_rate_pilot.py`;
- `scripts/run_cnn_exp049b_joint_pilot.py`.

Checkpoints не хранятся в GitHub. Notebook сохраняет resume bundle в output
Kaggle/Drive; пользователь сам управляет этим объектом между сессиями.

## Provenance результатов

Главный публичный отчёт последних запусков:
[CNN_EXP045_049_RESULTS_REVIEW_2026-10-01.md](CNN_EXP045_049_RESULTS_REVIEW_2026-10-01.md).

В нём отдельно отмечено:

- какие значения получены из локальных JSON/CSV;
- какие перенесены из сохранённых Kaggle evaluation logs;
- какие результаты являются oracle-диагностикой;
- почему validation после многократного просмотра считается exploratory.

## Правило научной интерпретации

Proxy-метрики, balanced-batch loss, regional AP и label-oracle не заменяют
primary genome-wide nucleotide AP. Новый champion фиксируется только после
полного natural-background evaluation по неизменному validation contract.
