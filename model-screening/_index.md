---
type: registry
entity: model-screening
---

# Реестр группового screening моделей

Screening начинается после завершения основной серии feature experiments.
Каждый запуск берёт один принятый feature reference, одну группу стартовых
моделей и неизменный validation contract.

Настройки и параметры: `src/ml_project/model_screening_config.py`.
Официальный запуск: [[notebooks/06_model_screening.ipynb]].

<!-- auto:model-screening-registry:start -->

Реестр появится после первого сохранения.

<!-- auto:model-screening-registry:end -->

## Как читать результат

- `Δ vs champion > 0` — средняя primary metric лучше feature champion;
- `Wins/Ties/Losses` — парное сравнение на тех же folds;
- `Shortlist` — модель стоит проверить tuning-ом, но она ещё не новый champion;
- решение и объяснение хранятся в отдельной `MS-xxx` карточке.
