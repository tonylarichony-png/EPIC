---
type: registry
entity: experiment
---

# Реестр экспериментов

Перед новой контролируемой проверкой откройте
[[README.md#Как начать новый эксперимент|короткую инструкцию в карточке проекта]].
Новый геномный эксперимент создаётся командой `.\new-epic-experiment.cmd`:
она создаёт отдельный отслеживаемый notebook и карточку с тем же `EXP-ID`.
Общий data contract находится в `src/ml_project/epic_data.py`, а изменяемые
sampling, признаки и модель остаются видимыми в notebook. Полный процесс:
[[docs/epic_experiment_workflow.md]].

Карточки с ещё не проверенными направлениями находятся отдельно в
[[experiment-ideas/_index.md|банке идей]] и не занимают номера `EXP`.

Когда feature experiments завершены, следующий этап ведётся отдельно:
[[model-screening/_index.md|групповой screening моделей]] на принятом feature set.

## Сейчас выполняется

- [[experiments/EXP-001.md|EXP-001]] остаётся первым воспроизводимым baseline.
- Серия whole-k-mer завершена: 4-mer и 6-mer приняты, 8-mer отклонён по
  заранее заданному правилу.
- Исторический champion последовательностного lookup: **6-mer из EXP-003**.
- [[experiments/EXP-005.md|EXP-005]] завершён: hierarchical `2/4/6-mer`
  logistic regression улучшила AP на 0.347%, но не прошла порог +1%.
- [[experiments/EXP-006.md|EXP-006]] завершён: linear GC201 повысил Spearman,
  но снизил AP на 2.015% против EXP-005 и проиграл на всех validation-contig.
- [[experiments/EXP-007.md|EXP-007]] завершён и принят: fixed GC201 bins дали
  validation AP **0.002053635** и EPIC Spearman **0.104302**.
- [[experiments/EXP-008.md|EXP-008]] завершён и отклонён: interaction дал
  только **+0.519% AP**, а Spearman немного снизился.
- [[experiments/EXP-009.md|EXP-009]] завершён и принят: fixed CpG O/E201 bins
  дали validation AP **0.002363735** и EPIC Spearman **0.115517**.
- Текущий champion: **EXP-009/run_001**.
- [[experiments/EXP-010.md|EXP-010]]: AP-screening всех 15 пар завершён; 45 fit сошлись. Лидируют GG (+4.445%), TA (+3.920%), TG (+2.860%) по mean train-CV AP к EXP-009. Полный Spearman-gate и validation не подтверждены, решение `pending`.
- Полные результаты: [[eda/findings/EDA-005.md|EDA-005 — региональные динуклеотиды]]. CV-значения не смешиваются с validation-метриками ниже.
- Следующий новый идентификатор: **EXP-011**.
- Предложение следующей проверки: [[experiment-ideas/IDEA-006_gg_ta_complementarity.md|дополняют ли GG и TA друг друга]]. Это план, расчёты ещё не запускались.

## Автоматический реестр

<!-- auto:experiment-registry:start -->

| ID | Название | Статус | AP reference → candidate | EPIC Spearman | Решение |
|---|---|---|---:|---:|---|
| [[experiments/EXP-001.md|EXP-001]] | Динуклеотидный baseline | completed | 0.000670644 → **0.001330258** | **0.090720** | `adopt` |
| [[experiments/EXP-002.md|EXP-002]] | 4-mer lookup | completed | 0.001330258 → **0.001486787** (+11.77%) | 0.090720 → **0.094232** | `adopt` |
| [[experiments/EXP-003.md|EXP-003]] | 6-mer lookup | completed | 0.001486787 → **0.001600537** (+7.65%) | 0.094232 → **0.097186** | `adopt` |
| [[experiments/EXP-004.md|EXP-004]] | 8-mer lookup | completed | 0.001600537 → 0.001603481 (+0.184%) | 0.097186 → 0.092203 | `reject` |
| [[experiments/EXP-005.md|EXP-005]] | Hierarchical 2/4/6-mer LR | completed | 0.001600537 → 0.001606098 (+0.347%) | 0.097186 → 0.097569 | `reject` |
| [[experiments/EXP-006.md|EXP-006]] | Hierarchical LR + linear GC201 | completed | 0.001606098 → 0.001573742 (−2.015% vs EXP-005) | 0.097569 → 0.101205 | `reject` |
| [[experiments/EXP-007.md|EXP-007]] | Hierarchical LR + fixed GC201 bins | completed | 0.001600537 → **0.002053635** (+28.31%) | 0.097186 → **0.104302** | `adopt` |
| [[experiments/EXP-008.md|EXP-008]] | EXP-007 + interaction(k2, GC201-bin) | completed | 0.002053635 → 0.002064290 (+0.519%) | 0.104302 → 0.104167 | `reject` |
| [[experiments/EXP-009.md|EXP-009]] | EXP-007 + fixed CpG O/E201 bins | completed | 0.002053635 → **0.002363735** (+15.10%) | 0.104302 → **0.115517** | `adopt` |

<!-- auto:experiment-registry:end -->

EXP-010 пока не включён в таблицу завершённых validation-сравнений: его отдельный AP-screening описан выше и в карточке эксперимента.

## Все заметки в Obsidian

```query
path:"experiments" -file:"_index"
```
