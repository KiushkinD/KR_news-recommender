# Результаты экспериментов

Основные выводы представлены в [отчёте о качестве](evaluation/evaluation_report.md).
Численные результаты по группам сценариев — в
[main_results.csv](evaluation/main_results.csv).

## Исходные данные оценки

- `test_scenarios.json` — сценарии и инструкция разметки.
- `test_runs.json` — 56 сохранённых выдач.
- `test_pool_manifest.json` — состав пула статей для оценки.
- `test_annotations.csv` — ручные оценки 315 пар «сценарий — статья».

## Расчёт качества

Каталог `evaluation/` содержит итоговый отчёт и таблицы метрик:

- `evaluation_per_run.csv` — показатели отдельных выдач;
- `personalization_deltas.csv` — изменения при добавлении истории;
- `evaluation_metrics.json` — полный расчёт, параметры и контрольные суммы;
- `assessment_distribution.csv`, `query_pools.csv`, `coverage.csv` — распределение
  оценок, состав пулов запросов и доля непустых выдач;
- `analysis_protocol.json` — условия расчёта в машиночитаемом виде.

В `input_snapshot/` сохранены копии входных файлов эксперимента.
`reference_calculation/` и `normalization_changes.csv` позволяют проверить
влияние объединения пулов одинаковых запросов на нормализацию NDCG.

## Подготовка данных и время работы

- `preprocessing_report.json` — отбор и очистка корпуса.
- `encoder_benchmark.json` — предварительное измерение E5 на 1 000 статьях.
- `embedding_report.json`, `tfidf_report.json` — параметры представлений;
  рекомендатель использует их для проверки согласованности данных.
- [performance_report.md](performance_report.md) — время обращений к системе.
  `performance_report.json` и `performance_samples.csv` содержат условия
  эксперимента и отдельные измерения.

## Дополнительные проверки

[additional_comparison/report.md](additional_comparison/report.md) — сравнение
собственного поиска E5 без истории с функцией `semantic_search` библиотеки
Sentence Transformers. В том же каталоге сохранены выдачи и измерения времени.

`refactoring_check/report.json` — проверка совпадения всех 56 выдач после
изменений структуры кода: порядок статей и численные оценки сохранены.
