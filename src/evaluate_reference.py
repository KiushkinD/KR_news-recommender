"""Проверка входов и контрольный расчёт метрик по отдельным пулам сценариев."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

METRICS = ('precision_at_5', 'precision_at_10', 'ndcg_at_10')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dcg(grades):
    return sum((2 ** grade - 1) / math.log2(rank + 2)
               for rank, grade in enumerate(grades[:10]))


def metrics(grades, pool_grades):
    ideal = dcg(sorted(pool_grades, reverse=True))
    return {'precision_at_5': sum(g == 2 for g in grades[:5]) / 5,
            'precision_at_10': sum(g == 2 for g in grades[:10]) / 10,
            'ndcg_at_10': dcg(grades) / ideal if ideal > 0 else None}


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for group, members in sorted(groups.items()):
        entry = dict(zip(keys, group))
        entry['n_scenarios'] = len(members)
        for metric in METRICS:
            values = [r[metric] for r in members if r[metric] is not None]
            entry[metric] = sum(values) / len(values) if values else None
        entry['n_ndcg_defined'] = sum(r['ndcg_at_10'] is not None for r in members)
        out.append(entry)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows, keys):
    def value(v):
        if isinstance(v, float) and abs(v) < 0.0005:
            v = 0.0
        return '—' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
    return '\n'.join(['| ' + ' | '.join(keys) + ' |',
                      '| ' + ' | '.join(['---'] * len(keys)) + ' |'] +
                     ['| ' + ' | '.join(value(row[k]) for k in keys) + ' |' for row in rows])


def evaluate(input_dir, annotation_path, output_dir):
    scenario_doc = read_json(input_dir / 'test_scenarios.json')
    manifest = read_json(input_dir / 'test_pool_manifest.json')
    run_doc = read_json(input_dir / 'test_runs.json')
    require(scenario_doc['split'] == run_doc['split'] == 'test', 'Expected test split')
    require(manifest['dataset_id'] == run_doc['dataset_id'], 'Dataset IDs differ')
    require(scenario_doc['scenarios'] == manifest['scenarios'], 'Scenario definitions differ')
    require(scenario_doc['alpha'] == manifest['alpha'] == run_doc['alpha'] == 0.8,
            'Expected frozen alpha=0.8')
    require(scenario_doc['top_n'] == manifest['top_n'] == run_doc['top_n'] == 10,
            'Expected pool depth=10')
    identity_keys = ('scenarios', 'assignments', 'corpus_model_text_sha256',
                     'model_revision', 'alpha', 'top_n')
    identity = {k: manifest[k] for k in identity_keys}
    fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                                            sort_keys=True).encode('utf-8')).hexdigest()
    require(fingerprint == manifest['dataset_id'], 'Manifest fingerprint mismatch')
    protocol = scenario_doc['protocol']
    require(protocol['precision_positive_label'] == 2 and
            protocol['ndcg_gain'] == '2**relevance - 1', 'Metric protocol changed')
    scenarios = {s['id']: s for s in scenario_doc['scenarios']}
    require(len(scenarios) == 16 and len(scenario_doc['scenarios']) == 16,
            'Expected 16 unique scenarios')
    assignments = manifest['assignments']
    expected = {(r['scenario_id'], r['article_id']): r['pool_id'] for r in assignments}
    require(len(expected) == len(assignments) == manifest['pool_size'], 'Duplicate pool pairs')
    require(len(set(expected.values())) == len(expected), 'Duplicate pool IDs')
    require(dict(Counter(s for s, a in expected)) == manifest['pool_counts'], 'Pool counts differ')
    require(set(s for s, a in expected) == set(scenarios), 'Unknown/missing pool scenario')
    with annotation_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == manifest['annotation_columns'], 'CSV columns differ')
        annotation_rows = list(reader)
    labels = {}
    for row in annotation_rows:
        key = row['scenario_id'], row['article_id']
        require(key in expected and expected[key] == row['pool_id'], 'CSV identity differs: ' + str(key))
        require(key not in labels, 'Duplicate CSV pair: ' + str(key))
        scenario = scenarios[key[0]]
        q, u = row['query_relevance'], row['user_relevance']
        require(u in ('0', '1', '2'), 'Missing/invalid user grade: ' + row['pool_id'])
        require(q == '' if scenario['kind'] == 'feed' else q in ('0', '1', '2'),
                'Missing/invalid query grade: ' + row['pool_id'])
        require(q != '0' or u == '0', 'q=0 requires u=0: ' + row['pool_id'])
        labels[key] = {'query_relevance': int(q) if q else None, 'user_relevance': int(u)}
    require(set(labels) == set(expected), 'CSV is incomplete')
    # Query relevance must not depend on history for an identical information need.
    consistency = {}
    for (sid, aid), grades in labels.items():
        scenario = scenarios[sid]
        if scenario['kind'] == 'query':
            key = scenario['query'], scenario['information_need'], aid
            if key in consistency:
                require(consistency[key] == grades['query_relevance'],
                        'Same-query grades conflict: ' + sid + '/' + aid)
            consistency[key] = grades['query_relevance']
    expected_runs = {(sid, method + '_' + variant) for sid, s in scenarios.items()
                     for method in ('tfidf', 'e5')
                     for variant in (('query_only', 'personalized') if s['kind'] == 'query'
                                     else ('personalized',))}
    seen_runs, pool_from_runs, detail = set(), set(), []
    for run in run_doc['runs']:
        sid, config = run['scenario_id'], run['config_id']
        key = sid, config
        require(key in expected_runs and key not in seen_runs, 'Unknown/duplicate run: ' + str(key))
        seen_runs.add(key)
        s, result = scenarios[sid], run['result']
        require(config == run['method'] + '_' + run['variant'], 'Configuration mismatch')
        require(result['method'] == run['method'] and result['query'] == s['query'], 'Run inputs differ')
        history = [] if run['variant'] == 'query_only' else s['history']
        require(result['history_considered'] == history, 'Run history differs')
        alpha = 1.0 if run['variant'] == 'query_only' else 0.8
        require(result['alpha_requested'] == alpha, 'Run alpha differs')
        items = result['items']
        require(len(items) <= 10, 'Run exceeds pool depth')
        require([i['rank'] for i in items] == list(range(1, len(items) + 1)), 'Rank order differs')
        ids = [i['article_id'] for i in items]
        require(len(set(ids)) == len(ids), 'Duplicate article in ranking')
        require(all((sid, aid) in labels for aid in ids), 'Unjudged ranked article')
        pool_from_runs.update((sid, aid) for aid in ids)
        criteria = ('user_relevance',) if s['kind'] == 'feed' else ('query_relevance', 'user_relevance')
        for criterion in criteria:
            ranked_grades = [labels[sid, aid][criterion] for aid in ids]
            pool_grades = [g[criterion] for (pool_sid, aid), g in labels.items() if pool_sid == sid]
            row = {'scenario_id': sid, 'pair_id': s['pair_id'] or '',
                   'kind': s['kind'], 'context_type': s['context_type'],
                   'config_id': config, 'method': run['method'], 'variant': run['variant'],
                   'criterion': criterion, 'returned_items': len(ids)}
            row.update(metrics(ranked_grades, pool_grades))
            detail.append(row)
    require(seen_runs == expected_runs, 'Missing test configurations')
    require(pool_from_runs == set(labels), 'Pool is not exactly the union of test rankings')
    overall = summarize(detail, ('kind', 'config_id', 'criterion'))
    grouped = summarize(detail, ('context_type', 'config_id', 'criterion'))
    by_key = {(r['scenario_id'], r['config_id'], r['criterion']): r for r in detail}
    deltas = []
    for row in detail:
        if row['kind'] != 'query' or row['variant'] != 'personalized':
            continue
        baseline = by_key[row['scenario_id'], row['method'] + '_query_only', row['criterion']]
        delta = {k: row[k] for k in ('scenario_id', 'context_type', 'method', 'criterion')}
        delta.update({m: row[m] - baseline[m] if row[m] is not None and baseline[m] is not None else None
                      for m in METRICS})
        deltas.append(delta)
    delta_groups = summarize(deltas, ('context_type', 'method', 'criterion'))
    empty_runs = [{'scenario_id': r['scenario_id'], 'config_id': r['config_id'],
                   'status': r['result']['status'], 'messages': r['result'].get('messages', [])}
                  for r in run_doc['runs'] if not r['result']['items']]
    hashes = {name: hashlib.sha256((input_dir / name).read_bytes()).hexdigest()
              for name in ('test_scenarios.json', 'test_pool_manifest.json', 'test_runs.json')}
    hashes['annotations'] = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_doc = {'dataset_id': manifest['dataset_id'], 'input_sha256': hashes,
                  'annotation_pairs': len(labels), 'test_runs': len(seen_runs),
                  'protocol': protocol, 'overall': overall, 'by_context': grouped,
                  'empty_runs': empty_runs,
                  'personalization_delta_by_context': delta_groups,
                  'per_run': detail, 'personalization_delta_per_scenario': deltas}
    (output_dir / 'evaluation_metrics.json').write_text(
        json.dumps(result_doc, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_csv(output_dir / 'evaluation_per_run.csv', detail)
    write_csv(output_dir / 'evaluation_summary.csv', overall)
    write_csv(output_dir / 'evaluation_by_context.csv', grouped)
    write_csv(output_dir / 'personalization_deltas.csv', deltas)
    keys = ['kind', 'config_id', 'criterion', 'n_scenarios', *METRICS, 'n_ndcg_defined']
    delta_keys = ['context_type', 'method', 'criterion', 'n_scenarios', *METRICS]
    report = ('# Результаты офлайн-оценки\n\n'
              f'Проверено {len(labels)} размеченных пар, {len(scenarios)} сценариев и {len(seen_runs)} выдач. '
              'Вес текущего запроса α = 0.8. Положительная оценка для Precision — только 2.\n\n'
              '## Средние метрики\n\n' + table(overall, keys) + '\n\n'
              '## Изменение при добавлении истории\n\n'
              'Персонализированный вариант минус вариант без истории, среднее парных разностей.\n\n'
              + table(delta_groups, delta_keys) + '\n\n'
              '## Пустые выдачи\n\n'
              + ('\n'.join('- ' + r['scenario_id'] + ' / ' + r['config_id'] + ': '
                           + r['status'] + '. ' + '; '.join(r['messages']) for r in empty_runs)
                 if empty_runs else 'Нет.') + '\n\n'
              '## Границы интерпретации\n\n'
              '- NDCG нормируется по идеальному Top-10 размеченного пула каждого сценария, а не всего корпуса.\n'
              '- Средние вычисляются по сценариям, а не по числу размеченных статей.\n'
              '- При нулевом IDCG NDCG не определена; число определённых значений указано в таблице.\n'
              '- Всего шесть разных текущих запросов с двумя историями каждый и четыре профиля ленты. '
              'Парные сценарии не являются независимыми запросами.\n'
              '- Пользователи и истории синтетические; итоговые оценки выставлены автором вручную одним оценщиком. '
              'Это не пользовательский эксперимент и не доказательство роста CTR.\n'
              '- Полезность оценивается по заданным потребности и истории, а не по реальному поведению.\n'
              '- Общий пул ограничен Top-10 участвующих конфигураций; результаты не распространяются автоматически на новые методы.\n'
              '- Проверки согласованности разметки проведены до расчёта метрик; оценки не подбираются для улучшения результатов.\n'
              '- Корпус — архив 2018 года; актуальность сегодняшних новостей не оценивается.\n')
    (output_dir / 'evaluation_report.md').write_text(report, encoding='utf-8')
    return result_doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    source = args.input_dir or args.project_root / 'results'
    annotations = args.annotations or source / 'test_annotations.csv'
    output = args.output_dir or args.project_root / 'results'
    try:
        result = evaluate(source, annotations, output)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, 'Ошибка проверки: ' + str(exc) + '\n')
    print(f"Проверка пройдена: {result['annotation_pairs']} пар, {result['test_runs']} выдач.")
    for row in result['overall']:
        values = ' '.join(f'{m}={row[m]:.3f}' if row[m] is not None else f'{m}=undefined'
                          for m in METRICS)
        print(row['kind'], row['config_id'], row['criterion'], values)
    print('Результаты сохранены:', output)


if __name__ == '__main__':
    main()
