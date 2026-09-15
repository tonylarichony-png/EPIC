---
type: registry
entity: eda-finding
tags:
  - ml/eda
  - ml/finding
---

# EDA-наблюдения

Здесь автоматически создаются карточки важных результатов из
[[notebooks/02_eda_hypotheses.ipynb]]. Каждая карточка хранит вопрос, метод,
вывод, график и/или таблицы, а также возможную гипотезу. Небольшие таблицы
встраиваются в Markdown, большие дополнительно сохраняются полностью как CSV.

Сводный список также отображается в [[docs/02_eda.md#Сохранённые EDA-наблюдения]].

## Все наблюдения

```dataview
TABLE WITHOUT ID
  link(file.path, id + " — " + title) AS "Наблюдение",
  created AS "Создано",
  features AS "Признаки",
  status AS "Статус"
FROM "eda/findings"
WHERE type = "eda-finding"
SORT id ASC
```
