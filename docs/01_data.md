---
type: stage
stage: data
status: draft
owner:
last_reviewed:
tags:
  - ml/stage
  - ml/data
---

# 01 — Исходные данные

← [[docs/00_problem.md|Problem]] · [[README.md|Dashboard]] · Далее → [[docs/02_eda.md|EDA]]

> [!abstract] Результат этапа
> Понятно, какие исходные файлы получены, откуда они взялись, для чего предназначены, где хранятся и с какой неизменённой версией продолжается работа. Содержимое и качество значений исследуются на этапе EDA.

## Обзор датасета

- **Единица наблюдения:** `= [[docs/00_problem]].observation_unit`
- **Период данных:**
- **Место хранения:** `data/raw/`.
- **Правила доступа:**

## Источники

| Источник | Содержание | Ключи | Период | Обновление | Владелец / поставщик |
|---|---|---|---|---|---|
|  |  |  |  |  | `= [[docs/00_problem]].data_provider` |

## Реестр файлов

Автоматически формируется из `data/raw/` через [[notebooks/01_data.ipynb]].

<!-- auto:data-file-report:start -->
> Запустите ячейку синхронизации в [[notebooks/01_data.ipynb]].
<!-- auto:data-file-report:end -->

## Исходное разбиение от поставщика

Здесь описывается только состав данных в полученном snapshot. Это ещё не окончательная model-ready выборка.

```text
Source snapshot
├── train / labeled data     → признаки + target
├── inference / test data    → признаки без известного target
└── service files, если есть → формат результата / справочная информация
```

- **Обучающие / размеченные данные:**
- **Данные для финального прогноза / inference:**
- **Служебные файлы:**
- **Какие изменения уже внесены в raw-данные:** никаких / описать.

Окончательные фильтры, исключения и преобразования фиксируются в [[docs/04_features.md]] после [[docs/02_eda.md|EDA]] и определения правил [[docs/03_validation.md|validation]].

## Схема данных

<!-- auto:data-schema:start -->
> Запустите ячейку синхронизации в [[notebooks/01_data.ipynb]].
<!-- auto:data-schema:end -->

## Версии датасета

<!-- auto:data-versions:start -->
> Запустите ячейку синхронизации в [[notebooks/01_data.ipynb]].
<!-- auto:data-versions:end -->

Версия должна быть воспроизводима через immutable snapshot, hash, DVC, lakehouse version, SQL + дату среза или другой однозначный механизм.

```dataviewjs
await dv.view("system/conditional-section", {
  id: "data-privacy",
  task: "show-data-privacy",
  file: "sections/data/privacy-security.md"
});
```

<!-- conditional-section:data-privacy:start -->
<!-- conditional-section:data-privacy:end -->

## Известные ограничения

- …

## Stage Gate: Data

- [ ] Названы владельцы и источники данных.
- [ ] Перечислены исходные файлы и назначение каждого файла.
- [ ] Определены единица наблюдения, ключи и структура файлов.
- [ ] Зафиксировано исходное разбиение файлов от поставщика.
- [ ] Зафиксирована хотя бы одна версия датасета.
- [ ] Privacy-раздел заполнен или отключён в [[PROJECT_CONFIG.md]] как неприменимый.
- [ ] Подтверждено, что исходные файлы в `data/raw/` не изменяются.

> [!success] Следующий этап
> После выполнения Stage Gate можно переходить к [[docs/02_eda.md|02 — EDA]].
