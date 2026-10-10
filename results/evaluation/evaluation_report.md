# Результаты офлайн-оценки

Это дополнительный анализ после просмотра исходных результатов, а не заранее зарегистрированный протокол. Выдачи, оценки, модели и α не изменялись. Входные данные проверяет evaluate_reference.py.

Проверено 315 пар, 56 выдач, 16 контекстов; разных запросов — 6.

## Основные таблицы

| context_type | config_id | criterion | n_scenarios | n_unique_queries | precision_at_5 | precision_at_10 | ndcg_at_10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| broad | e5_personalized | query_relevance | 4 | 2 | 0.650 | 0.625 | 0.672 |
| broad | e5_personalized | user_relevance | 4 | 2 | 0.300 | 0.375 | 0.610 |
| broad | e5_query_only | query_relevance | 4 | 2 | 0.500 | 0.600 | 0.639 |
| broad | e5_query_only | user_relevance | 4 | 2 | 0.150 | 0.175 | 0.447 |
| broad | tfidf_personalized | query_relevance | 4 | 2 | 0.600 | 0.450 | 0.662 |
| broad | tfidf_personalized | user_relevance | 4 | 2 | 0.050 | 0.050 | 0.485 |
| broad | tfidf_query_only | query_relevance | 4 | 2 | 0.600 | 0.450 | 0.661 |
| broad | tfidf_query_only | user_relevance | 4 | 2 | 0.050 | 0.050 | 0.484 |
| history_aligned | e5_personalized | query_relevance | 4 | 4 | 0.800 | 0.750 | 0.874 |
| history_aligned | e5_personalized | user_relevance | 4 | 4 | 0.700 | 0.675 | 0.843 |
| history_aligned | e5_query_only | query_relevance | 4 | 4 | 0.750 | 0.750 | 0.877 |
| history_aligned | e5_query_only | user_relevance | 4 | 4 | 0.700 | 0.675 | 0.853 |
| history_aligned | tfidf_personalized | query_relevance | 4 | 4 | 0.550 | 0.500 | 0.691 |
| history_aligned | tfidf_personalized | user_relevance | 4 | 4 | 0.550 | 0.450 | 0.670 |
| history_aligned | tfidf_query_only | query_relevance | 4 | 4 | 0.450 | 0.475 | 0.663 |
| history_aligned | tfidf_query_only | user_relevance | 4 | 4 | 0.450 | 0.425 | 0.640 |
| history_conflict | e5_personalized | query_relevance | 4 | 4 | 0.700 | 0.725 | 0.830 |
| history_conflict | e5_personalized | user_relevance | 4 | 4 | 0.000 | 0.000 | 0.968 |
| history_conflict | e5_query_only | query_relevance | 4 | 4 | 0.750 | 0.750 | 0.877 |
| history_conflict | e5_query_only | user_relevance | 4 | 4 | 0.000 | 0.000 | 0.984 |
| history_conflict | tfidf_personalized | query_relevance | 4 | 4 | 0.450 | 0.475 | 0.663 |
| history_conflict | tfidf_personalized | user_relevance | 4 | 4 | 0.000 | 0.000 | 0.886 |
| history_conflict | tfidf_query_only | query_relevance | 4 | 4 | 0.450 | 0.475 | 0.663 |
| history_conflict | tfidf_query_only | user_relevance | 4 | 4 | 0.000 | 0.000 | 0.886 |
| history_only | e5_personalized | user_relevance | 4 | 0 | 0.950 | 0.900 | 0.969 |
| history_only | tfidf_personalized | user_relevance | 4 | 0 | 0.500 | 0.500 | 0.590 |

query_relevance — соответствие текущему запросу без учёта истории. user_relevance — совместная релевантность запросу и интересам истории; для ленты — релевантность интересам. Это оценка заданного сценария, не измерение реальной удовлетворённости пользователя. Оба критерия сохранены во всех сценариях с запросом.

## Изменение при добавлении истории

С историей минус без истории, среднее парных разностей по контекстам.

| context_type | method | criterion | n_scenarios | precision_at_5 | precision_at_10 | ndcg_at_10 |
| --- | --- | --- | --- | --- | --- | --- |
| broad | e5 | query_relevance | 4 | 0.150 | 0.025 | 0.033 |
| broad | e5 | user_relevance | 4 | 0.150 | 0.200 | 0.163 |
| broad | tfidf | query_relevance | 4 | 0.000 | 0.000 | 0.001 |
| broad | tfidf | user_relevance | 4 | 0.000 | 0.000 | 0.001 |
| history_aligned | e5 | query_relevance | 4 | 0.050 | 0.000 | -0.004 |
| history_aligned | e5 | user_relevance | 4 | 0.000 | 0.000 | -0.011 |
| history_aligned | tfidf | query_relevance | 4 | 0.100 | 0.025 | 0.028 |
| history_aligned | tfidf | user_relevance | 4 | 0.100 | 0.025 | 0.030 |
| history_conflict | e5 | query_relevance | 4 | -0.050 | -0.025 | -0.047 |
| history_conflict | e5 | user_relevance | 4 | 0.000 | 0.000 | -0.016 |
| history_conflict | tfidf | query_relevance | 4 | 0.000 | 0.000 | 0.000 |
| history_conflict | tfidf | user_relevance | 4 | 0.000 | 0.000 | 0.001 |

## Доля непустых выдач

| context_type | config_id | n_runs | n_nonempty | n_empty | nonempty_fraction |
| --- | --- | --- | --- | --- | --- |
| broad | e5_personalized | 4 | 4 | 0 | 1.000 |
| broad | e5_query_only | 4 | 4 | 0 | 1.000 |
| broad | tfidf_personalized | 4 | 4 | 0 | 1.000 |
| broad | tfidf_query_only | 4 | 4 | 0 | 1.000 |
| history_aligned | e5_personalized | 4 | 4 | 0 | 1.000 |
| history_aligned | e5_query_only | 4 | 4 | 0 | 1.000 |
| history_aligned | tfidf_personalized | 4 | 4 | 0 | 1.000 |
| history_aligned | tfidf_query_only | 4 | 4 | 0 | 1.000 |
| history_conflict | e5_personalized | 4 | 4 | 0 | 1.000 |
| history_conflict | e5_query_only | 4 | 4 | 0 | 1.000 |
| history_conflict | tfidf_personalized | 4 | 4 | 0 | 1.000 |
| history_conflict | tfidf_query_only | 4 | 4 | 0 | 1.000 |
| history_only | e5_personalized | 4 | 4 | 0 | 1.000 |
| history_only | tfidf_personalized | 4 | 3 | 1 | 0.750 |

Пустые выдачи остаются в расчёте качества и дают нулевой Precision.



В конфликтующих контекстах: 80 пар, 0 двоек полезности, 47 пар с query=2 и user=1. Это совместимо с принятым критерием: статья отвечает запросу, но не совпадает с интересами истории. Само отсутствие двоек не доказывает ошибку разметки. Оно ограничивает достижимый Precision внутри наблюдаемого пула.


