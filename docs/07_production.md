---
type: stage
stage: production
status: draft
owner:
last_reviewed:
tags:
  - ml/stage
  - ml/production
---

# 07 — Production

← [[docs/06_error_analysis.md|Error analysis]] · [[README.md|Dashboard]]

> [!abstract] Результат этапа
> Модель воспроизводимо поставляется, имеет контракт входа и выхода, контролируемый rollout, мониторинг, владельцев, rollback и правила retraining.

> [!tip] Kaggle inference
> Конкурсные full-train fit, проверенные CSV и внешние scores ведутся отдельно в
> [[submissions/_index.md]] через [[notebooks/07_submission.ipynb]]. Submission
> не назначает production champion автоматически.

## Production scope

- **Режим:** batch / online / streaming / embedded.
- **Потребитель:**
- **Частота / SLA:**
- **Latency budget:**
- **Throughput:**
- **Cost budget:**
- **Fallback:**
- **Human-in-the-loop:**

## Inference contract

### Вход

| Поле | Тип | Required | Допустимый диапазон | Поведение при ошибке |
|---|---|---|---|---|
|  |  | yes / no |  |  |

### Выход

| Поле | Тип | Смысл | Диапазон / schema |
|---|---|---|---|
|  |  |  |  |

### Семантика

- **Версия контракта:**
- **Момент предсказания:**
- **Threshold / post-processing:**
- **Idempotency:**
- **Timeout / retry:**

## Model artifact

| Поле | Значение |
|---|---|
| Model version |  |
| Source experiment |  |
| Dataset version |  |
| Code commit |  |
| Environment / image |  |
| Artifact URI / registry |  |
| Hash |  |
| Owner |  |
| Approval |  |

## Train-serving pipeline

```text
Data → validation → features → model → post-processing → artifact
                                                   ↓
Request → online features → model → decision → logging
```

- Общий код preprocessing:
- Feature store / online source:
- Проверка schema:
- Проверка parity:
- Версионирование зависимостей:

## Тестирование

- [ ] Unit tests для трансформаций.
- [ ] Schema / contract tests.
- [ ] Проверка train-serving parity.
- [ ] Интеграционный inference test.
- [ ] Нагрузочный тест.
- [ ] Проверка fallback и timeout.
- [ ] Reproducibility test артефакта.
- [ ] Проверка на regression set из [[docs/06_error_analysis.md]].

## Rollout

- **Стратегия:** shadow / canary / A/B / phased / full.
- **Размер первого трафика:**
- **Критерии расширения:**
- **Критерии остановки:**
- **Продолжительность наблюдения:**
- **Ответственный:**

## Rollback

- **Предыдущая стабильная версия:**
- **Как переключить:**
- **Ожидаемое время:**
- **Кто имеет доступ:**
- **Как восстановить данные / очередь:**

## Monitoring

### Data quality

| Сигнал | Метрика | Порог | Частота | Реакция |
|---|---|---:|---|---|
| Schema |  |  |  |  |
| Missing rate |  |  |  |  |
| Range / categories |  |  |  |  |
| Volume |  |  |  |  |

### Drift

| Сигнал | Reference | Метод | Порог | Реакция |
|---|---|---|---:|---|
| Feature drift |  | PSI / KS / JS / custom |  |  |
| Prediction drift |  |  |  |  |
| Segment mix |  |  |  |  |

### Model quality

| Метрика | Label delay | Окно | Порог | Реакция |
|---|---|---|---:|---|
|  |  |  |  |  |

### Operational

- latency;
- error rate;
- throughput;
- resource usage;
- dependency failures;
- fallback rate.

## Логирование и обратная связь

- Какие inputs / outputs логируются?
- Где хранится model version?
- Как prediction связывается с будущим label?
- Что нельзя логировать из-за privacy?
- Как собирается feedback пользователя?

## Retraining policy

- **Trigger:** schedule / drift / quality drop / data volume / manual.
- **Минимальный объём новых labels:**
- **Частота:**
- **Автоматические проверки:**
- **Нужно ли ручное одобрение:**
- **Champion–challenger:**
- **Когда новая модель отклоняется:**

## Runbook и инциденты

| Ситуация | Как обнаружить | Первое действие | Владелец | Эскалация |
|---|---|---|---|---|
| Нет входных данных |  |  |  |  |
| Резкий drift |  |  |  |  |
| Падение качества |  |  |  |  |
| Высокая latency |  |  |  |  |
| Некорректные ответы |  |  |  |  |

Инцидент с долгой историей оформляйте через [[templates/issue.md]].

## Безопасность и governance

- Контроль доступа:
- PII / secrets:
- Audit trail:
- Model card / ограничения использования:
- Fairness / regulatory review:
- Срок хранения:

## Stage Gate: Production

- [ ] Определены SLA, владелец и потребитель.
- [ ] Версионируются вход, выход и model artifact.
- [ ] Train-serving pipeline проверен на parity.
- [ ] Пройдены unit, contract, integration и load tests.
- [ ] Определены rollout и rollback.
- [ ] Настроены data, drift, quality и operational monitoring.
- [ ] Prediction можно связать с будущим label.
- [ ] Есть retraining policy и правила одобрения.
- [ ] Есть runbook и владельцы инцидентов.
- [ ] Ограничения модели доведены до пользователей.

> [!success] Завершение цикла
> После выполнения Stage Gate обновите [[README.md|Dashboard]] и зафиксируйте итог. Новые ошибки или изменения данных могут вернуть проект на предыдущий этап.
