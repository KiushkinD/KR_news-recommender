"""Дополнительное сравнение с точным поиском Sentence Transformers.

Запуск из корня проекта:
    .venv/bin/python -u src/compare_external_search.py

Использует готовые E5-векторы и замороженную разметку. Основные файлы
эксперимента не перезаписываются. Внешняя часть — адаптация библиотечной
функции semantic_search, а не самостоятельный новостной сервис.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dcg(grades):
    return sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(grades[:10]))


def metrics(grades, pool):
    ideal = dcg(sorted(pool, reverse=True))
    return {
        'precision_at_5': sum(g == 2 for g in grades[:5]) / 5,
        'precision_at_10': sum(g == 2 for g in grades[:10]) / 10,
        'ndcg_at_10': dcg(grades) / ideal if ideal else None,
    }


def quality_rows(scenarios, labels, rankings):
    """Общий query-IDCG для потребности; отдельный user-IDCG для контекста."""
    pools = defaultdict(dict)
    by_context = defaultdict(dict)
    for (sid, aid), grades in labels.items():
        by_context[sid][aid] = grades
    for s in scenarios:
        key = (s['query'], s['information_need'])
        for aid, grades in by_context[s['id']].items():
            g = grades['query_relevance']
            if aid in pools[key] and pools[key][aid] != g:
                raise ValueError('Несогласованные оценки текущей потребности')
            pools[key][aid] = g
    rows = []
    for s in scenarios:
        sid = s['id']
        for implementation, by_query in rankings.items():
            ids = by_query[s['query']]
            missing = [aid for aid in ids if (sid, aid) not in labels]
            for criterion in ('query_relevance', 'user_relevance'):
                row = {'scenario_id': sid, 'context_type': s['context_type'],
                       'implementation': implementation, 'criterion': criterion,
                       'returned_items': len(ids), 'unjudged_items': len(missing),
                       'precision_at_5': None, 'precision_at_10': None,
                       'ndcg_at_10': None}
                if not missing:
                    grades = [labels[sid, aid][criterion] for aid in ids]
                    pool = (list(pools[s['query'], s['information_need']].values())
                            if criterion == 'query_relevance' else
                            [g[criterion] for g in by_context[sid].values()])
                    row.update(metrics(grades, pool))
                rows.append(row)
    return rows


def quality_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['context_type'], row['implementation'], row['criterion']].append(row)
    result = []
    for (context, implementation, criterion), group in sorted(groups.items()):
        complete = all(r['unjudged_items'] == 0 for r in group)
        row = {'context_type': context, 'implementation': implementation,
               'criterion': criterion, 'n_contexts': len(group),
               'n_fully_judged': sum(r['unjudged_items'] == 0 for r in group),
               'precision_at_5': None, 'precision_at_10': None, 'ndcg_at_10': None}
        if complete:
            for metric in ('precision_at_5', 'precision_at_10', 'ndcg_at_10'):
                values = [r[metric] for r in group if r[metric] is not None]
                row[metric] = statistics.mean(values) if values else None
        result.append(row)
    return result


def timing_summary(values):
    x = sorted(v * 1000 for v in values)
    position = (len(x) - 1) * .95
    lo, hi = math.floor(position), math.ceil(position)
    return {'calls': len(x), 'mean_ms': statistics.mean(x),
            'median_ms': statistics.median(x),
            'p95_ms': x[lo] + (x[hi] - x[lo]) * (position - lo)}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-dir', type=Path, default=None)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    root = args.project_root.resolve()
    inputs = args.input_dir or root / 'results/evaluation/input_snapshot'
    if args.input_dir is None and not (inputs / 'test_scenarios.json').exists():
        inputs = root / 'results'
    names = ['test_scenarios.json', 'test_runs.json', 'test_pool_manifest.json',
             'test_annotations.csv']
    before = {name: digest(inputs / name) for name in names}
    print('Проверяем сохранённые сценарии, выдачи и разметку…', flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = root / 'results' / ('additional_comparison_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    from evaluate_reference import evaluate
    evaluate(inputs, inputs / 'test_annotations.csv', out / 'reference_validation')
    scenario_doc = read_json(inputs / 'test_scenarios.json')
    scenarios = [s for s in scenario_doc['scenarios'] if s['kind'] == 'query']
    queries = list(dict.fromkeys(s['query'] for s in scenarios))
    frozen = {}
    for run in read_json(inputs / 'test_runs.json')['runs']:
        if run['config_id'] == 'e5_query_only':
            frozen[run['scenario_id']] = [i['article_id'] for i in run['result']['items']]
    labels = {}
    with (inputs / 'test_annotations.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['query_relevance']:
                labels[row['scenario_id'], row['article_id']] = {
                    c: int(row[c]) for c in ('query_relevance', 'user_relevance')}

    print('Загружаем библиотеки и готовые ресурсы E5…', flush=True)
    import numpy as np
    import torch
    from sentence_transformers import util
    from recommender import NewsRecommender
    engine = NewsRecommender(root, device=args.device)
    engine._encode([queries[0]], 'e5')
    # CPU comparison follows the prototype; query encoding uses its device.
    corpus_tensor = torch.from_numpy(np.array(engine._embeddings, dtype=np.float32, copy=True))

    def synchronize():
        device = engine._encoder.device.type
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize(engine._encoder.device)

    def external(query):
        vector = engine._encode([query], 'e5')
        hits = util.semantic_search(torch.from_numpy(vector), corpus_tensor,
                                    top_k=10, score_function=util.dot_score)[0]
        items = []
        for rank, hit in enumerate(hits, 1):
            index = int(hit['corpus_id'])
            article = engine.articles.iloc[index]
            score = float(hit['score'])
            if not math.isfinite(score):
                raise ValueError('Неконечный балл библиотечного поиска')
            items.append({'rank': rank, 'article_id': article['article_id'],
                          'title': article['title'], 'date': article['date'],
                          'topic': article['topic'] or 'Не указано', 'url': article['url'],
                          'score': score, 'query_score': score, 'history_score': None})
        return {'method': 'st_semantic_search', 'query': query, 'mode': 'query_only',
                'status': 'ok', 'history_used': [], 'effective_alpha': 1.0,
                'items': items}

    implementations = {
        'prototype_e5_query_only': lambda q: engine.recommend(q, [], method='e5',
                                                              alpha=1.0, top_n=10),
        'st_semantic_search': external,
    }
    rankings = {name: {} for name in implementations}
    runs = []
    print('Прогрев и сохранение выдач: {} разных запросов…'.format(len(queries)), flush=True)
    for query in queries:
        for name, call in implementations.items():
            result = call(query)
            ids = [item['article_id'] for item in result['items']]
            if len(ids) != 10 or len(set(ids)) != 10:
                raise ValueError('Ожидалась непустая выдача из десяти разных статей')
            rankings[name][query] = ids
            runs.append({'implementation': name, 'query': query, 'result': result})
    synchronize()
    timings = []
    rng = random.Random(20261010)
    for repeat in range(args.repeats):
        tasks = [(query, name) for query in queries for name in implementations]
        rng.shuffle(tasks)
        for query, name in tasks:
            synchronize()
            started = perf_counter()
            result = implementations[name](query)
            synchronize()
            elapsed = perf_counter() - started
            if [item['article_id'] for item in result['items']] != rankings[name][query]:
                raise ValueError('Выдача меняется между повторами')
            timings.append({'repeat': repeat + 1, 'query': query,
                            'implementation': name, 'elapsed_seconds': elapsed})
        print('  Повтор {}/{} завершён'.format(repeat + 1, args.repeats), flush=True)

    comparisons = []
    for s in scenarios:
        project_ids = rankings['prototype_e5_query_only'][s['query']]
        external_ids = rankings['st_semantic_search'][s['query']]
        comparisons.append({'scenario_id': s['id'], 'query': s['query'],
                            'prototype_matches_frozen_order': project_ids == frozen[s['id']],
                            'external_matches_prototype_order': external_ids == project_ids,
                            'external_matches_prototype_set': set(external_ids) == set(project_ids),
                            'top10_overlap': len(set(external_ids) & set(project_ids))})
    quality = quality_rows(scenarios, labels, rankings)
    summary = quality_summary(quality)
    after = {name: digest(inputs / name) for name in names}
    if after != before:
        raise RuntimeError('Входные файлы изменились во время дополнительного опыта')
    report = {
        'experiment': 'additional comparison of adapted exact semantic search',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_url': 'https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html',
        'corpus_size': len(engine.article_ids), 'n_unique_queries': len(queries),
        'n_contexts': len(scenarios), 'repeats': args.repeats,
        'model': engine.embedding_report['model'],
        'model_revision': engine.embedding_report.get('model_revision'),
        'max_seq_length': engine.embedding_report['max_seq_length'],
        'encoder_device': str(engine._encoder.device), 'ranking_device': 'cpu',
        'platform': platform.platform(), 'python': platform.python_version(),
        'versions': {p: version(p) for p in ('numpy', 'torch', 'sentence-transformers')},
        'torch_cpu_threads': torch.get_num_threads(),
        'input_sha256': before, 'inputs_unchanged': True,
        'matches_frozen_all_contexts': all(c['prototype_matches_frozen_order'] for c in comparisons),
        'timings': {name: timing_summary([r['elapsed_seconds'] for r in timings
                                         if r['implementation'] == name])
                    for name in implementations},
        'quality': summary,
        'limitations': [
            'External component is the library function adapted to shared E5 resources, not a news website.',
            'Query encoding and metadata retrieval are included; loading, tensor preparation and UI are excluded.',
            'Prototype uses float64 accumulation and a full ID-aware sort; library search uses float32 top-k.',
            'Ties and small floating-point differences can change top-k order.',
            'No quality mean is reported for a group with unjudged returned articles.',
            'Usability with participants, feed support of the external example and NRMS training are not measured.',
        ],
    }
    for name, value in [('report.json', report), ('runs.json', runs)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                                encoding='utf-8')
    write_csv(out / 'timings.csv', timings)
    write_csv(out / 'quality_per_context.csv', quality)
    write_csv(out / 'quality_summary.csv', summary)
    write_csv(out / 'ranking_comparison.csv', comparisons)
    lines = ['# Дополнительное сравнение точного семантического поиска', '',
             'Использованы общий encoder, готовые embeddings и исходная разметка.',
             'Внешний компонент — адаптация функции semantic_search без истории.', '',
             '| Реализация | Вызовов | Среднее, мс | P95, мс |', '|---|---:|---:|---:|']
    for name, values in report['timings'].items():
        lines.append('| {} | {} | {:.2f} | {:.2f} |'.format(
            name, values['calls'], values['mean_ms'], values['p95_ms']))
    lines += ['', 'Метрики и покрытие оценками: quality_summary.csv.',
              'Совпадение с замороженным E5: {}.'.format(report['matches_frozen_all_contexts']),
              'Входы остались неизменными; выдачи и замеры сохранены в этой папке.',
              'При новых неразмеченных статьях качество соответствующей группы не вычисляется.',
              'Если исходный E5 не совпал с замороженной выдачей, переносить сравнение в основной тест нельзя.',
              'Разница времени относится к реализациям, точности чисел и способу отбора, а не к качеству модели.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for name, values in report['timings'].items():
        print('{}: {:.2f} мс, P95 {:.2f} мс'.format(name, values['mean_ms'], values['p95_ms']))
    print('Совпадение собственного E5 с замороженным:', report['matches_frozen_all_contexts'])
    print('Отчёт:', out / 'report.md', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        raise SystemExit('Ошибка дополнительного сравнения: {}'.format(error))
