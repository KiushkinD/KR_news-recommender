# Результаты офлайн-оценки

Проверено 315 размеченных пар, 16 сценариев и 56 выдач. Вес текущего запроса α = 0.8. Положительная оценка для Precision — только 2.

## Средние метрики

| kind | config_id | criterion | n_scenarios | precision_at_5 | precision_at_10 | ndcg_at_10 | n_ndcg_defined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| feed | e5_personalized | user_relevance | 4 | 0.950 | 0.900 | 0.969 | 4 |
| feed | tfidf_personalized | user_relevance | 4 | 0.500 | 0.500 | 0.590 | 4 |
| query | e5_personalized | query_relevance | 12 | 0.717 | 0.700 | 0.797 | 12 |
| query | e5_personalized | user_relevance | 12 | 0.333 | 0.350 | 0.807 | 12 |
| query | e5_query_only | query_relevance | 12 | 0.667 | 0.700 | 0.803 | 12 |
| query | e5_query_only | user_relevance | 12 | 0.283 | 0.283 | 0.761 | 12 |
| query | tfidf_personalized | query_relevance | 12 | 0.533 | 0.475 | 0.679 | 12 |
| query | tfidf_personalized | user_relevance | 12 | 0.200 | 0.167 | 0.680 | 12 |
| query | tfidf_query_only | query_relevance | 12 | 0.500 | 0.467 | 0.669 | 12 |
| query | tfidf_query_only | user_relevance | 12 | 0.167 | 0.158 | 0.670 | 12 |

## Изменение при добавлении истории

Персонализированный вариант минус вариант без истории, среднее парных разностей.

| context_type | method | criterion | n_scenarios | precision_at_5 | precision_at_10 | ndcg_at_10 |
| --- | --- | --- | --- | --- | --- | --- |
| broad | e5 | query_relevance | 4 | 0.150 | 0.025 | 0.034 |
| broad | e5 | user_relevance | 4 | 0.150 | 0.200 | 0.163 |
| broad | tfidf | query_relevance | 4 | 0.000 | 0.000 | 0.001 |
| broad | tfidf | user_relevance | 4 | 0.000 | 0.000 | 0.001 |
| history_aligned | e5 | query_relevance | 4 | 0.050 | 0.000 | -0.004 |
| history_aligned | e5 | user_relevance | 4 | 0.000 | 0.000 | -0.011 |
| history_aligned | tfidf | query_relevance | 4 | 0.100 | 0.025 | 0.028 |
| history_aligned | tfidf | user_relevance | 4 | 0.100 | 0.025 | 0.030 |
| history_conflict | e5 | query_relevance | 4 | -0.050 | -0.025 | -0.050 |
| history_conflict | e5 | user_relevance | 4 | 0.000 | 0.000 | -0.016 |
| history_conflict | tfidf | query_relevance | 4 | 0.000 | 0.000 | 0.000 |
| history_conflict | tfidf | user_relevance | 4 | 0.000 | 0.000 | 0.001 |

## Пустые выдачи

- feed_03 / tfidf_personalized: history_oov. Некоторые запросы истории отсутствуют в словаре TF-IDF и пропущены.; Не удалось представить интересы из истории. Используйте другие запросы или E5.

## Границы интерпретации

- NDCG нормируется по идеальному Top-10 размеченного пула каждого сценария, а не всего корпуса.
- Средние вычисляются по сценариям, а не по числу размеченных статей.
- При нулевом IDCG NDCG не определена; число определённых значений указано в таблице.
- Всего шесть разных текущих запросов с двумя историями каждый и четыре профиля ленты. Парные сценарии не являются независимыми запросами.
- Итоговые оценки выставлены автором вручную. Критерии и отдельные примеры обсуждались с ИИ; независимого второго оценщика не было. Это не эксперимент с реальными пользователями и не доказательство роста CTR.
- Полезность оценивается по заданным потребности и истории, а не по реальному поведению.
- Общий пул ограничен Top-10 участвующих конфигураций; результаты не распространяются автоматически на новые методы.
- Проверки согласованности разметки проведены до расчёта метрик; оценки не подбираются для улучшения результатов.
- Корпус — архив 2018 года; актуальность сегодняшних новостей не оценивается.
