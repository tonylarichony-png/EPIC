---
id: ISSUE-002
type: issue
status: resolved
severity: medium
priority: high
owner:
created: 2026-09-18
resolved: 2026-09-18
experiment: EXP-009
tags:
  - ml/issue
  - ml/compatibility
  - ml/experiment
---

# ISSUE-002 — EXP-009: ошибка Series.reset_index при сохранении

← [[issues/_index.md|Реестр проблем]] · [[experiments/EXP-009.md|EXP-009]] · [[README.md|Dashboard]]

## Кратко

- **Что произошло:** ячейка `save` падала при подготовке `cv_gate_checks.csv`.
- **Ошибка:** `TypeError: Series.reset_index() got an unexpected keyword argument 'names'`.
- **Влияние:** рассчитанные результаты оставались в памяти notebook, но immutable run и CSV-артефакты не создавались.
- **Потеря результатов:** нет, если kernel после ошибки не перезапускался.

## Root cause

Вызов

```python
cv_gate_checks.rename("passed").reset_index(names="criterion")
```

использует неподходящий аргумент API: `cv_gate_checks` является `Series`, а
`Series.reset_index()` не принимает параметр `names=` в используемом pandas.
Похожий параметр есть у `DataFrame.reset_index()`, откуда и возникла ошибка.
Это ошибка слоя сохранения, а не ошибка подсчёта признака, обучения или метрик
EXP-009.

## Решение

Имя индекса задаётся отдельной совместимой операцией:

```python
cv_gate_checks.rename("passed").rename_axis("criterion").reset_index()
```

Результат остаётся тем же: DataFrame с колонками `criterion` и `passed`.

## Проверка и защита от повторения

- Добавлен notebook contract test, запрещающий `reset_index(names=...)` в ячейке сохранения EXP-009.
- После исправления достаточно повторно выполнить только ячейку `save`, а затем ячейку синхронизации отчёта; пересчитывать признаки, CV и final fit не требуется.
- Для кода, который должен работать в зафиксированном окружении проекта, имя индекса задавать через `rename_axis(...).reset_index()`.

## Закрытие

- **Статус:** resolved.
- **Дата закрытия:** 2026-09-18.
- **Остаточный риск:** если kernel уже перезапущен, переменные результатов утрачены и соответствующие вычислительные ячейки придётся восстановить повторным запуском.
