# Каталог notebooks

В репозитории сохранены исходные и выполненные notebooks, чтобы ход исследования
можно было проследить от подготовки данных до последних CNN-экспериментов.
GitHub отображает `.ipynb` в браузере; кнопка **Download raw file** скачивает
отдельный notebook без клонирования репозитория.

## С чего начать

| Notebook | Назначение |
|---|---|
| [01_train_validation_test.ipynb](01_train_validation_test.ipynb) | структура Nematostella и разделение по contig |
| [02_eda.ipynb](02_eda.ipynb) | основной EDA |
| [03_baseline.ipynb](03_baseline.ipynb) | воспроизводимый baseline workflow |
| [04_experiment.ipynb](04_experiment.ipynb) | единый runner контролируемого эксперимента |
| [05_diagnostics.ipynb](05_diagnostics.ipynb) | анализ OOF-ошибок и признаков |
| [06_model_screening.ipynb](06_model_screening.ipynb) | сравнение семейств моделей |
| [07_submission.ipynb](07_submission.ipynb) | full-train fit и сборка submission |

## Интерпретируемая feature-ветка

Последовательность `EXP-001`—`EXP-011` находится в
[experiments/](experiments/). Ключевые воспроизводимые checkpoints:

| Эксперимент | Notebook | Результат |
|---|---|---|
| EXP-001 | [dinucleotide baseline](experiments/EXP-001_dinucleotide_baseline.ipynb) | AP `0.001330258` |
| EXP-003 | [6-mer lookup](experiments/EXP-003_6mer_lookup.ipynb) | лучший lookup-контекст |
| EXP-007 | [GC201 bins](experiments/EXP-007_gc201_bins_logistic.ipynb) | AP `0.002053635` |
| EXP-009 | [CpG O/E201](experiments/EXP-009_cpg_oe201_logistic.ipynb) | AP `0.002363735`, feature champion |

Человеческие карточки и решения: [../experiments/_index.md](../experiments/_index.md).

## CNN-ветка

CNN notebooks находятся в
[experiments/CNN_experiments/](experiments/CNN_experiments/). Они сохраняют как
положительные, так и отрицательные результаты; номер эксперимента важнее даты
изменения файла.

### Ключевая последовательность EXP033—049

| Эксперимент | Notebook | Что проверялось | Итог |
|---|---|---|---|
| EXP033 | [base-level RF2053 / R32](experiments/CNN_experiments/CNN_EXP033_BASELEVEL_RF2053_REGIONAL_32_ONLY_10K.ipynb) | региональная постановка и error analysis | диагностическая ветка |
| EXP036 | [WIDTH64 RF1029](experiments/CNN_experiments/CNN_EXP036_WIDTH64_RF1029_STRONG_VALIDATION.ipynb) | direct nucleotide baseline | AP `0.033996892` |
| EXP041 | [WIDTH128 25k](experiments/CNN_experiments/CNN_EXP041_WIDTH128_25K_RF1029_STRONG_VALIDATION.ipynb) | width/training scaling | AP `0.074412694` |
| EXP044 | [WIDTH256 50k cosine](experiments/CNN_experiments/CNN_EXP044_WIDTH256_25K_TO_50K_COSINE_LR.ipynb) | зрелый schedule | AP `0.093361726` |
| EXP045 | [WIDTH512 50k](experiments/CNN_experiments/CNN_EXP045_WIDTH512_50K_TWO_PHASE_LR_RF1029.ipynb) | capacity scaling | AP `0.097630094` |
| EXP046 | [R32 feasibility frontier](experiments/CNN_experiments/CNN_EXP046_R32_FEASIBILITY_FRONTIER.ipynb) | oracle/региональная диагностика | oracle не является моделью |
| EXP046B | [frozen regional correction](experiments/CNN_experiments/CNN_EXP046B_FROZEN_REGIONAL_CORRECTION_R32.ipynb) | frozen R32 residual | AP `0.098867560`, reject |
| EXP047 | [paired risk loss](experiments/CNN_experiments/CNN_EXP047_WIDTH512_PAIRED_RISK_LOSS_CONTINUATION.ipynb) | matched loss comparison | **AP `0.104211926`** |
| EXP048 | [ultra-tail refinement](experiments/CNN_experiments/CNN_EXP048_ULTRA_TAIL_RISK_REFINEMENT_EXACT.ipynb) | top-0.1% hard negatives | AP `0.082120732`, reject |
| EXP049 | [Kaggle/Colab cloud train](experiments/CNN_experiments/CNN_EXP049_CLOUD_TRAIN.ipynb) | R16 A→FiLM→B joint cascade | AP `0.104208798`, champion не превзойдён |

Итоговое сравнение с графиками:
[CNN-EXP-045—049 review](../docs/CNN_EXP045_049_RESULTS_REVIEW_2026-10-01.md).

### Диагностические notebooks

- [EXP033 false-positive study](experiments/CNN_experiments/EPIC_EXP033_FALSE_POSITIVE_STUDY.ipynb)
- [EXP033 far-FP study](experiments/CNN_experiments/EPIC_EXP033_FALSE_POSITIVE_STUDY_V3_FAR_FP.ipynb)
- [EXP033 strength-matched TP](experiments/CNN_experiments/EXP033_strength_matched_tp.ipynb)
- [EXP048 tail diagnostic](experiments/CNN_experiments/EXP048_TAIL_DIAGNOSTIC.ipynb)

## Kaggle и Colab

Публичный notebook EXP049 не содержит genome archive, risk-cache или
checkpoints. Подключение input datasets и resume-архивов описано в
[CNN_EXP049_CLOUD_TRAINING.md](../docs/CNN_EXP049_CLOUD_TRAINING.md).

```text
notebooks/experiments/CNN_experiments/CNN_EXP049_CLOUD_TRAIN.ipynb
```

Для запуска понадобится самостоятельно добавить `jaNemVect1.1.tgz` и при
необходимости EXP047 risk-cache. Эти файлы намеренно исключены из Git.

## Важные ограничения

- Outputs в notebooks являются частью научного журнала, но не заменяют JSON/CSV
  источники метрик и карточки решений.
- Абсолютные локальные пути из старых Windows notebooks нужно заменить на свою
  директорию данных.
- Некоторые старые notebooks были exploratory и не соответствуют финальному
  validation protocol; статус проверяйте по карточке эксперимента.
- `.ipynb_checkpoints`, datasets, checkpoints и cloud ZIP-бандлы не публикуются.
