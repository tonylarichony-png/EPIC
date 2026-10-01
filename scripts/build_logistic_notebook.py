"""Build the readable, executable EPIC notebook; calculations live in the engine."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))

def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Nematostella: EDA для логистической регрессии

**Данные:** `jaNemVect1.1.tgz` · **Target:** `csRNA-r1 + csRNA-r2 > 0` · **Seed:** 42.

Цель — выбрать понятные признаки из ДНК и разумную ширину контекста для следующего baseline.
Это **исследовательская проверка (pilot)** на выборке официального train, с результатами прямо в notebook.
Официальные test-ответы не читаются. Логрегрессия оценивает наличие сигнала; интенсивность — отдельная задача.

### Навигация

1. Воспроизводимость и предыдущий анализ.
2. Target, реплики, контиги, ограничения данных.
3. Предрегистрация: что и как сравниваем.
4. Позиционный сигнал, состав окна и аудит признаков.
5. Реальные эксперименты с логрегрессией и стабильность.
6. Интерпретация коэффициентов и практический shortlist.

Банк идей — это не проведённые эксперименты:
[IDEA-001](../experiment-ideas/IDEA-001_context_width.md),
[IDEA-002](../experiment-ideas/IDEA-002_sequence_composition.md),
[IDEA-003](../experiment-ideas/IDEA-003_local_interactions.md),
[IDEA-004](../experiment-ideas/IDEA-004_lowercase.md).
Обзор предыдущих работ: [eda_prior_review.md](../docs/eda_prior_review.md).

**Запуск:** из Miniforge Prompt — `conda activate epic-eda`, затем `jupyter lab`.
Выбрать ядро **Python (EPIC EDA · Miniforge)** и выполнить **Restart Kernel and Run All**.
В корне проекта также есть `run-eda.cmd`. Установка: `conda env create -f environment-eda.yml`.
Точные установленные пакеты: `environment-eda-lock.yml` и `conda-eda-win-64.lock.txt`.
""")

code("""
from pathlib import Path
import sys, json, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/eda_nematostella.py').exists())
sys.path.insert(0, str(ROOT / 'scripts'))
import eda_logistic_context as engine

SEED = 42
ASSETS = ROOT / 'assets/eda/logistic_context'
ASSETS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 10, 'axes.grid': False})
pd.set_option('display.max_columns', 18)
pd.set_option('display.max_colwidth', 95)
pd.set_option('display.max_rows', 25)

def show_plot(fig, name):
    fig.savefig(ASSETS / name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

environment = {'python': sys.version.split()[0], 'executable': sys.executable,
               **{p: importlib.metadata.version(p) for p in
                  ['numpy','pandas','scipy','scikit-learn','matplotlib','nbclient']}}
display(pd.Series(environment, name='Environment').to_frame())
assert Path(sys.prefix).name == 'epic-eda', 'Выберите conda kernel epic-eda перед полным запуском.'
""")

md("""
## 1. Что уже было измерено

Предыдущий `scripts/eda_nematostella.py` полностью проверил FASTA/FAI, whitelist/template и шесть
train-TXT против независимо развёрнутых BED. Интервалы BED могут покрывать несколько позиций.
В нём были только описательные связи: ±50 bp для нуклеотидов и окно 201 bp для GC.
Он **не обучал логрегрессию и не выбирал ширину**. Его JSON ниже используется как проверяемый источник.

В соседнем `epic_solution` есть переданные результаты CNN/lookup, в том числе на demo-test.
Они получены на другом протоколе. Окна CNN 8–16 kb не задают ширину для логрегрессии.
Все 12 train-контигов ранее просмотрены в EDA, поэтому локальные folds не называются нетронутым holdout.
""")
code("""
prior = json.loads((ROOT / 'artifacts/eda/nematostella/summary.json').read_text(encoding='utf-8'))
display(pd.Series({k: prior[k] for k in
                   ['assembly','archive_md5','created_utc','test_labels_read','model_fitted','elapsed_seconds']}))
display(pd.DataFrame(prior['txt_audit']).T)
assert prior['test_labels_read'] is False and prior['model_fitted'] is False
""")

md("""
## 2. Target и единица наблюдения

Одна строка — **позиция + цепь**, label — наличие суммарного csRNA-сигнала.
Нулевой count имеет смысл только внутри train-whitelist. Отсутствие сигнала вне whitelist не создаёт отрицательный пример.
Обе цепи одного контига всегда находятся в одном fold.

Основная метрика — Average Precision. Вторичная диагностика — ROC AUC и Pearson между **dense ranks**
counts и scores на положительных позициях. Это определение из
[официальных метрик EPIC](https://epic.autosome.org/docs/performance-evaluation-metrics), проверенных 15.09.2026.
Обычный `spearmanr` использует другую обработку одинаковых значений.
""")
code("""
target = prior['target']
display(pd.Series({k: target[k] for k in ['positions','positive','positive_fraction','singletons_fraction','top_1pct_read_share']}))
display(pd.Series(prior['replicates'], name='Replicate diagnostics').to_frame())
bins = pd.DataFrame(target['bins'])
contigs_prior = pd.DataFrame(prior['contigs'])
fig, ax = plt.subplots(1, 2, figsize=(12,4), layout='constrained')
ax[0].bar(bins['label'], 100 * bins['positions'] / target['positive'], color='#287b8e')
ax[0].tick_params(axis='x', rotation=30)
ax[0].set(xlabel='Pooled count среди положительных', ylabel='Доля положительных, %', title='Слабый сигнал преобладает')
ax[1].bar(contigs_prior['contig'].str.replace('NC_', '', regex=False),
          100 * contigs_prior['positive_fraction'], color='#bc642f')
ax[1].tick_params(axis='x', rotation=65)
ax[1].set(ylabel='Положительных позиций, %', title='Частота сигнала по train-контигам')
show_plot(fig, 'target_contigs.png')
display(Markdown(f'''**Наблюдение:** сигнал есть у {100*target['positive_fraction']:.5f}% позиций;
константный score имеет AP = {target['positive_fraction']:.7f}.
{100*target['singletons_fraction']:.2f}% положительных имеют count=1.
**Интерпретация:** accuracy почти целиком определяется нулями; слабый сигнал нельзя автоматически удалить.
**Надёжность:** полные train-данные, но биологическая воспроизводимость ограничена;
Jaccard повторностей не является доказанным потолком качества.
**Следующее действие:** контролируем отбор отрицательных и оцениваем устойчивость по контигам.'''))
""")

md("""
### Train и будущий inference: проверка доступных последовательностей

Сравниваем только ДНК и whitelist, без test-labels. Близость среднего GC не доказывает отсутствия drift:
регистровая разметка, локальные распределения и новые мотивы могут отличаться.
Время, партия и стадия развития не образуют доступные колонки для модельного анализа;
в метаданных набора указаны смешанные стадии, причинный эффект стадии здесь не оцениваем.
""")
code("""
sequence_compare = pd.DataFrame(prior['sequence']).T[
    ['bp','contigs','intervals','gc_fraction_acgt','lower_fraction','n_or_other_fraction']]
display(sequence_compare)
display(pd.DataFrame(prior['segments']))
display(Markdown(f\"\"\"**Наблюдение:** GC среди A/C/G/T: train {100*prior['sequence']['train']['gc_fraction_acgt']:.3f}%,
test {100*prior['sequence']['test']['gc_fraction_acgt']:.3f}%; lowercase:
{100*prior['sequence']['train']['lower_fraction']:.2f}% против {100*prior['sequence']['test']['lower_fraction']:.2f}%.
**Вывод:** один общий GC близок, однако перенос регистрового признака требует проверки.
Сильные сигналы и N не удаляем на основании этой таблицы.\"\"\"))
""")

md("""
## 3. Идеи проверок до создания первого эксперимента

| Идея | Вопрос | Можно менять | Следует фиксировать |
|---|---|---|---|
| IDEA-001 | Сколько позиционного контекста полезно? | 11, 21, 51, 101, 201, 401 bp | Строки, folds, C=1, L2, reference coding |
| IDEA-002 | Нужен ли широкий состав ДНК? | Состав окон 101, 201, 401, 801 bp | Позиционное окно, выбранное ранее |
| IDEA-003 | Помогает ли сочетание соседних букв? | Локальные динуклеотидные взаимодействия | Позиционное окно, выбранное ранее |
| IDEA-004 | Даёт ли регистр дополнительный сигнал? | Lowercase-фракция | Позиционное окно + выбранный состав |

Эти строки не регистрируют `EXP`. Первый настоящий эксперимент создаётся
отдельно и получает номер `EXP-001`.

**Контроли:** константный score и динуклеотид `[-1,0]`. В координатах модели 0 — предсказываемая позиция;
отрицательные смещения находятся upstream по предсказываемой цепи. Для minus выполняется reverse complement.
N и отсутствующий за краем контига контекст учитываются явно; окно никогда не склеивает разные контиги.

**Вычислительный бюджет:** 1 seed; 3 группы контигов; до 1500 положительных и 6000 отрицательных на контиг.
Выбор без замещения, одинаковые строки для всех кандидатов. Разбиение строится по длинам без использования target.
Подбор C/threshold не выполняется. Метрики служат выбору shortlist для следующей проверки.

### Почему нельзя показать обычный AP на этой выборке

Выборка обогащена положительными. Для evaluation каждая строка получает вес
`число позиций её класса в контиге / число выбранных строк этого класса в контиге`.
Так восстанавливается численность классов. Полученный **weighted sampled AP — оценка**, а не точный AP
на всех 311 млн позиций. Редкие сильные ложные срабатывания могут не попасть в выборку;
bootstrap по 12 контигам не охватывает всю неопределённость отрицательного subsampling.
Обучение использует балансировку классов отдельно от evaluation; scores не считаются калиброванными вероятностями.
""")

md("""
## 3.1. Как из исходных файлов получается таблица для логрегрессии

### Target

Одна строка соответствует ключу `(contig, coordinate, strand)` внутри train-whitelist.
BED-интервалы каждой csRNA-повторности сначала раскрываются до отдельных нуклеотидов. Затем для одинакового ключа:

```text
count = csRNA_r1_count + csRNA_r2_count
target = 1, если count > 0, иначе 0
```

Если ключ отсутствует в разреженном BED, но входит в whitelist, его count равен нулю. Исходный `count`
сохраняется для диагностики интенсивности, но логрегрессия обучается на бинарном `target`.
RNA-seq, соседние counts и test-ответы в target или признаки не входят.

### Строки pilot-выборки

Полный train содержит 311 871 646 position-strand, поэтому плоский CSV со всеми признаками не создаётся.
Для model-probe без замещения выбираются на каждом из 12 контигов 1 500 положительных и 6 000 отрицательных:
всего 90 000 строк. Обе цепи контига относятся к одному fold.

`artifacts/eda/logistic_context/sample_rows.csv.gz` — компактный реестр строк:

| Поле | Назначение | Входит в X? |
|---|---|---|
| `row_id`, `template_index` | воспроизводимый порядок и связь с template | нет |
| `contig`, `coordinate_0based`, `strand` | извлечение контекста и grouped split | нет |
| `fold` | номер удерживаемой группы контигов | нет |
| `target` | бинарный y: pooled count > 0 | это y |
| `count` | исходная величина сигнала для диагностики | нет |
| `population_weight` | поправка на вероятность отбора класса внутри контига | sample weight только при evaluation |

### Признаки X

Последовательность извлекается непосредственно из FASTA по координатам CSV и ориентируется по цепи.
Для minus используется reverse complement, поэтому отрицательные offsets всегда означают upstream.

- Позиционные буквы: для каждого offset бинарные колонки `A/C/G/N-or-edge`; `T` — reference.
  Например, `pos51` имеет `51 × 4 = 204` разреженных столбца, `pos401` — 1 604.
- Состав окна: `GC / ACGT`, доля `N`, доля контекста за краем контига и индикатор отсутствия ACGT.
- Lowercase: доля строчных букв проверяется отдельной абляцией.
- Взаимодействия: индикаторы конкретных соседних пар в диапазоне `[-5,+5]`.

Матрица `X` создаётся в формате `scipy.sparse.csr_matrix`, а не CSV. Числовые агрегаты добавляются
как несколько плотных столбцов и стандартизуются только по train fold. Координата, contig, strand,
count и сигнальные треки модель не получает.

### Что сохраняем на диск

- `sample_rows.csv.gz` — строки, target, count, fold и веса;
- `oof_predictions.csv.gz` — те же ключи и OOF-scores кандидатов;
- обычные CSV — метрики, коэффициенты и результаты по контигам;
- полную feature-матрицу не сохраняем: она детерминированно пересобирается из FASTA и конфигурации кандидата.
""")
code("""
state = engine.prepare(ROOT, seed=SEED)
descriptive = engine.descriptive(state, ROOT)
print('Descriptive outputs:', list(descriptive))
""")
code("""
display(state['rows'].head(10))
display(pd.Series({
    'rows': len(state['rows']),
    'positive_rows': int(state['rows']['target'].sum()),
    'negative_rows': int((state['rows']['target'] == 0).sum()),
    'contigs': state['rows']['contig'].nunique(),
    'folds': state['rows']['fold'].nunique(),
    'pos51_sparse_columns': 51 * 4,
    'pos401_sparse_columns': 401 * 4,
}, name='Model-probe contract').to_frame())
""")
code("""
display(descriptive['contigs'])
display(descriptive['duplicate_audit'])
display(descriptive['feature_audit'])
""")
md("""
**Интерпретация аудита:** ID/координаты нужны для адресации и группировки, csRNA/rnaseq — для target и диагностики.
Эти поля не поступают в модель как признаки. Регистр описывает исходную FASTA-разметку и может отражать
техническую обработку; поэтому проверяется отдельной абляцией. В прежнем train не было положительных в центре N,
но это не гарантия для других данных. Межконтиговые повторы требуют отдельного контроля даже после split по контигам.
Проверка точных совпадений в выборке не исключает гомологию и приближённые повторы всего генома.
""")

md("""
## 4. Как меняется связь с расстоянием

**Вопрос:** сигнал сосредоточен у центра или остаётся на расстоянии сотен нуклеотидов?
Сравниваем частоты A/C/G/T/N в ориентированных окнах положительных и отрицательных примеров.
Log2 enrichment равен 0 при одинаковой частоте; +1 означает двукратное обогащение.
Это ассоциация с target, не причинный эффект и не самостоятельное доказательство нужной ширины модели.
""")
code("""
enrichment = descriptive['position_enrichment']
heat = enrichment.pivot(index='base', columns='offset', values='log2_enrichment')
fig, ax = plt.subplots(figsize=(13,3.4), layout='constrained')
limit = max(0.3, float(np.nanquantile(np.abs(heat.to_numpy()), 0.98)))
im = ax.imshow(heat.to_numpy(), aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit,
               extent=[heat.columns.min()-.5, heat.columns.max()+.5, len(heat)-.5,-.5])
ax.set_yticks(range(len(heat)), heat.index)
ax.axvline(0, color='black', lw=.7)
ax.set(xlabel='Смещение по предсказываемой цепи, bp', ylabel='Буква', title='Позиционное обогащение: log2(pos/neg)')
fig.colorbar(im, ax=ax, label='log2 enrichment (цвет обрезан по 98% квантилю)')
show_plot(fig, 'position_enrichment.png')
display(enrichment.loc[enrichment['offset'].between(-3,3)])
display(Markdown('**Вывод:** локальные пики задают кандидатов на позиционные признаки. '
                 'Слабые дальние ассоциации могут отражать общий состав окна; варианты проверки сохранены в IDEA-001 и IDEA-002.'))
""")

md("""
## 5. Состав ДНК на разных масштабах

**Вопрос:** достаточно ли короткого позиционного ядра и нескольких понятных сводных признаков?
Изучаем GC, N, края контига и lowercase на нескольких масштабах.
Определения признаков и знаменатели приведены в аудите выше. Постоянные признаки не дают обучаемого эффекта;
редкие флаги могут иметь недостаточно примеров для оценки. Отдельный test-target для такого анализа не нужен.
""")
code("""
multi = descriptive['multiscale_summary']
display(multi)
features_plot = [v for v in multi['feature'].unique() if 'gc' in v.lower() or 'lower' in v.lower()][:4]
fig, axes = plt.subplots(1, len(features_plot), figsize=(5*len(features_plot),3.8), squeeze=False, layout='constrained')
for ax, feature in zip(axes[0], features_plot):
    sub = multi[multi['feature'] == feature]
    for target_value, part in sub.groupby('target'):
        ax.plot(part['window_bp'], part['weighted_mean'], marker='o', label=f'target={target_value}')
    ax.set(xlabel='Ширина окна, bp', ylabel='Взвешенное среднее', title=feature)
    ax.legend()
show_plot(fig, 'multiscale_composition.png')
display(Markdown('**Интерпретация:** различие средних показывает связь состава с label. '
                 'Из-за корреляции соседних масштабов не добавляем все окна автоматически. '
                 'IDEA-002 предлагает менять один масштаб при фиксированном позиционном ядре; эксперимент ещё не проводился.'))
""")

md("""
## 6. Обучение: одинаковые данные, разные признаки

Следующая ячейка **реально обучает** модели. Перед fit движок сохраняет конфигурацию предрегистрации.
Все fit-преобразования числовых признаков выполняются только на train fold.
OOF означает, что score строки получен моделью, которая не обучалась на её контиге.
Число итераций и сходимость проверяются отдельно: не сошедшийся запуск не следует автоматически принимать.
""")
code("""
results = engine.run_experiments(state, ROOT)
print('Experiment outputs:', list(results))
display(results['scores'])
display(results['summary'])
""")
md("""
## 7. Ширина контекста и устойчивость результата

Показываем каждый fold; стандартное отклонение по трём folds не является строгим доверительным интервалом.
Предпочитаем короткий контекст, если усложнение не даёт устойчивой прибавки.
Все кандидаты сравниваются на тех же данных: выбор по этим результатам требует новой подтверждающей проверки.
""")
code("""
scores = results['scores']
fig, ax = plt.subplots(figsize=(13,5), layout='constrained')
order = list(dict.fromkeys(scores['model']))
for fold, part in scores.groupby('fold'):
    values = part.set_index('model')['weighted_ap'].reindex(order)
    ax.plot(range(len(order)), values, marker='o', alpha=.8, label=f'Fold {fold}')
ax.axhline(target['positive_fraction'], color='gray', ls=':', label='Train prevalence (ориентир)')
ax.set_xticks(range(len(order)), order, rotation=60, ha='right')
ax.set(ylabel='Weighted sampled AP', title='Добавление контекста и блоков признаков')
ax.legend(ncol=4)
show_plot(fig, 'model_comparison.png')
display(results['paired_bootstrap'])
display(Markdown('**Надёжность:** парный bootstrap пересэмплирует контиги, сохраняя сравнение моделей на одинаковых группах. '
                 'Это разведочная оценка стабильности, условная на текущей выборке и OOF-моделях; '
                 'она не исправляет многократный выбор кандидатов и не является подтверждающим тестом.'))
""")

md("""
## 8. Почему модель повышает или снижает score

Логрегрессия складывает вклад признаков в log-odds. Позиционные коэффициенты используют **T как базовую букву**:
положительный коэффициент A в позиции −1 означает повышение score при A вместо T при фиксированных остальных признаках.
Для категорий динуклеотида базовая пара — TT.
Числовые коэффициенты возвращены из train-fold scaling в исходные единицы: для долей эффект указан
на **+0.10 (10 процентных пунктов)**; для индикаторов — переход 0 → 1.
В модели с взаимодействиями главный эффект буквы соответствует соседям в базовой категории T.
При замене буквы в реальном окне нужно также прибавить изменившиеся взаимодействия.
Экспонента коэффициента взаимодействия A:B — отношение
`odds(A,B) × odds(T,T) / [odds(A,T) × odds(T,B)]` при остальных фиксированных признаках.
`exp(coefficient)` — отношение шансов в обучаемой модели, а не биологическое причинное действие.
Коррелированные признаки перераспределяют вклад; знак и разброс по folds помогают увидеть нестабильность.
""")
code("""
coef = results['coefficients']
coef = coef[~coef['feature'].str.startswith('intercept_')].copy()
display(results['recommendation'])
# Полные коэффициенты сохраняются в CSV; для чтения показываем наиболее сильные эффекты каждой модели.
view = coef.assign(abs_coef=coef['coef_mean'].abs()).sort_values('abs_coef', ascending=False)
display(view.groupby('model', sort=False).head(5).drop(columns='abs_coef'))
model_for_plot = next((m for m in coef['model'].unique() if m == 'pos51'), coef['model'].iloc[0])
part = view[view['model'] == model_for_plot].head(20).sort_values('coef_mean')
fig, ax = plt.subplots(figsize=(10,6), layout='constrained')
ax.barh(part['feature'], part['coef_mean'], color=np.where(part['coef_mean']>=0,'#287b8e','#bc642f'))
ax.errorbar(part['coef_mean'], range(len(part)), xerr=part['coef_sd'].fillna(0), fmt='none', color='#444', capsize=2)
ax.axvline(0,color='black',lw=.8)
ax.set(xlabel='Средний коэффициент ± SD по folds', title=f'{model_for_plot}: 20 наиболее сильных коэффициентов')
show_plot(fig, 'coefficients.png')
""")

md("""
## 9. Решение и границы применимости

**Берём как обязательную основу:** ориентированную ДНК, позиционные индикаторы, явное представление N/краёв;
контиги/координаты оставляем метаданными. **Проверяем по результатам:** длину позиционного окна,
GC на отдельном масштабе, локальные взаимодействия и добавку регистра.

Рекомендация ниже вычисляется из выполненных экспериментов. Она задаёт shortlist, не production champion.
Если длинное окно выигрывает слабо или нестабильно, следующий baseline начинает с более короткого.

**Что остаётся:** увеличить отрицательный background/выполнить полный validation whitelist потоками;
проверить устойчивость на дополнительных seeds и отложенном подтверждающем split;
оценить отдельно интенсивность сигнала. Вероятности требуют калибровки на естественной частоте классов.
Полноценный перенос на другие виды и сравнение с CNN здесь не измеряются.

### Связь с шаблоном

Карточки `experiment-ideas/IDEA-001_*.md` … `IDEA-004_*.md` содержат только
подсказки. Результатов экспериментов в них нет.
Наблюдения — `eda/findings/EDA-001.md` … `EDA-004.md`; конкретные признаки и протокол —
`docs/04_features.md` и `docs/03_validation.md`. Полные локальные таблицы/OOF находятся в
`artifacts/eda/logistic_context`, графики и компактные отчёты — в `assets/eda/logistic_context`.
""")
code("""
display(Markdown('### Измеренная рекомендация'))
display(results['recommendation'])
print('Полные артефакты:', ROOT / 'artifacts/eda/logistic_context')
print('Графики:', ASSETS)
print('Проверьте статус converged во всех строках scores перед переносом решения в baseline.')
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'display_name': 'Python (EPIC EDA · Miniforge)', 'language': 'python', 'name': 'epic-eda'}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.12'}
nb.metadata['epic'] = {'purpose': 'interpretable exploratory logistic-regression EDA', 'seed':42,
                       'experiment_ideas':['IDEA-001','IDEA-002','IDEA-003','IDEA-004'],
                       'experiments_run': []}
nbf.write(nb, ROOT / 'notebooks/02_eda_logistic_context.ipynb')
print('Wrote notebook:', len(cells), 'cells')
