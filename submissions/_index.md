---
type: registry
entity: kaggle-submissions
---

# Kaggle submissions

Каждый SUB-run выбирает один уже измеренный Candidate ID, восстанавливает его
точный feature/model pipeline, выполняет full-train fit и создаёт локальный CSV.

Выбор и официальный запуск: [[notebooks/07_submission.ipynb]].

<!-- auto:submission-registry:start -->

Реестр появится после первого сохранения.

<!-- auto:submission-registry:end -->

## Как читать

- `Candidate` связывает файл с конкретным EXP/MS-результатом;
- fitted pipeline и CSV хранятся локально в `artifacts/submissions/<SUB-ID>/`;
- Public score заполняется вручную в SUB-карточке после загрузки на Kaggle;
- Kaggle score не заменяет локальную валидацию и не назначает champion.
