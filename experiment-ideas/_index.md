---
type: registry
entity: experiment-idea
---

# Банк идей для экспериментов

Здесь лежат **не результаты и не pre-registration**, а подсказки, собранные из
EDA. Идентификатор `IDEA-*` нельзя цитировать как проведённый эксперимент.

Когда выбираем идею для проверки:

1. создаём новый `EXP-xxx` командой `.\new-epic-experiment.cmd`;
2. переносим только актуальную гипотезу и параметры;
3. заранее фиксируем критерий по Average Precision и EPIC Spearman;
4. запускаем отдельный notebook и только после этого записываем результат.

## Идеи

| ID | Вопрос | Зависимость |
|---|---|---|
| [[experiment-ideas/IDEA-001_context_width.md\|IDEA-001]] | Какая ширина позиционного контекста достаточна? | первая возможная проверка |
| [[experiment-ideas/IDEA-002_sequence_composition.md\|IDEA-002]] | Добавляет ли широкий состав ДНК информацию? | после выбора reference-контекста |
| [[experiment-ideas/IDEA-003_local_interactions.md\|IDEA-003]] | Полезны ли локальные взаимодействия оснований? | после выбора reference-контекста |
| [[experiment-ideas/IDEA-004_lowercase.md\|IDEA-004]] | Даёт ли lowercase независимый сигнал? | после проверки состава |
| [[experiment-ideas/IDEA-005_regional_dinucleotide_screening.md\|IDEA-005 — исходный план EXP-010]] | Какие региональные пары добавляют сигнал сверх GC и CpG? | AP-screening завершён; результаты в EDA-005, полный gate не закрыт |
| [[experiment-ideas/IDEA-006_gg_ta_complementarity.md\|IDEA-006 — план EXP-011]] | Дополняют ли региональные GG и TA друг друга? | EXP-010 выделил двух AP-лидеров; сравнить одиночные и совместную модель |

Текущий принятый champion — `EXP-009/run_001`. EXP-010 зарегистрирован, его полный gate ещё не закрыт. Следующий свободный номер — `EXP-011`; IDEA-006 сама по себе его не регистрирует.
