"""Расчёт метрик сохранённых выдач с общим query-IDCG для одинаковых запросов."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import evaluate_reference as reference
from evaluate_reference import METRICS, dcg, evaluate, read_json, require, summarize, table, write_csv


INPUT_NAMES = ('test_scenarios.json', 'test_pool_manifest.json', 'test_runs.json')
SUPPORTED_CRITERIA = {
    'broad': ('query_relevance', 'user_relevance'),
    'history_aligned': ('query_relevance', 'user_relevance'),
    'history_conflict': ('query_relevance', 'user_relevance'),
    'history_only': ('user_relevance',),
}
PROTOCOL = {
    'version': 3,
    'analysis_type': 'post_hoc_exploratory_reanalysis',
    'reason': 'Assessor clarified that user_relevance measures joint relevance to current need and historical interests; retain both criteria. Use shared query-IDCG for identical queries.',
    'unchanged': ['rankings', 'scenarios', 'labels', 'models', 'alpha=0.8', 'precision_positive_label=2', 'gain=2**relevance-1'],
    'supported_criteria_by_context': SUPPORTED_CRITERIA,
    'query_idcg': 'Union of judged articles with identical query and information_need, deduplicated by article_id.',
    'user_idcg': 'Original judged pool of each individual scenario.',
    'group_means': 'Equal-weight scenario means within each context; undefined NDCG excluded with counts.',
    'user_relevance_interpretation': 'Joint scenario relevance, not observed satisfaction and not history relevance alone. Lack of grade 2 in conflicting contexts does not establish a labelling error.',
    'no_relabelling': True,
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


REFERENCE_TOLERANCE = 1e-12


def compare_reference_values(actual, expected, location):
    """Compare structure exactly and computed floats within a tiny tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                'Reference keys differ: ' + location)
        for key in expected:
            compare_reference_values(actual[key], expected[key], location + '/' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                'Reference list differs: ' + location)
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_reference_values(left, right, location + '/' + str(index))
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=REFERENCE_TOLERANCE,
                                 abs_tol=REFERENCE_TOLERANCE),
                f'Reference number differs: {location}: saved={actual!r}, calculated={expected!r}')
    else:
        require(type(actual) is type(expected) and actual == expected,
                f'Reference value differs: {location}: saved={actual!r}, calculated={expected!r}')


def compare_reference_csv(saved, calculated):
    with saved.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        saved_columns, saved_rows = reader.fieldnames, list(reader)
    with calculated.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    require(saved_columns == columns and len(saved_rows) == len(rows),
            'Reference CSV shape differs: ' + saved.name)
    for index, (left, right) in enumerate(zip(saved_rows, rows)):
        for column in columns:
            location = f'{saved.name}/row_{index + 1}/{column}'
            if column in METRICS and left[column] != '' and right[column] != '':
                compare_reference_values(float(left[column]), float(right[column]), location)
            else:
                require(left[column] == right[column],
                        f'Reference CSV value differs: {location}: saved={left[column]!r}, calculated={right[column]!r}')


def preserve_reference(saved_dir, calculated_dir):
    # Validate all structured reports before copying any missing files.
    # Archived Markdown is retained as historical text; JSON and CSV carry data.
    for path in sorted(calculated_dir.iterdir()):
        target = saved_dir / path.name
        if not target.exists():
            continue
        if path.suffix == '.json':
            compare_reference_values(read_json(target), read_json(path), path.name)
        elif path.suffix == '.csv':
            compare_reference_csv(target, path)
    for path in sorted(calculated_dir.iterdir()):
        target = saved_dir / path.name
        if not target.exists():
            shutil.copyfile(path, target)


def query_key(scenario):
    return scenario['query'], scenario['information_need']


def add_query_counts(rows, summaries, scenarios):
    for summary in summaries:
        members = [row for row in rows if all(row[key] == summary[key]
                   for key in ('context_type', 'config_id', 'criterion'))]
        summary['n_unique_queries'] = len({query_key(scenarios[row['scenario_id']])
                                         for row in members if row['kind'] == 'query'})


def run_analysis(source, annotations, output):
    input_hashes = {name: file_hash(source / name) for name in INPUT_NAMES}
    input_hashes['annotations'] = file_hash(annotations)
    previous = output / 'evaluation_metrics.json'
    if previous.exists():
        require(read_json(previous)['input_sha256'] == input_hashes,
                'Inputs differ from this report. Use a new --output-dir to preserve the previous analysis.')
    require(source.resolve() != output.resolve(), 'Use a separate output directory')
    for name in INPUT_NAMES:
        require(not (output / name).resolve() == (source / name).resolve(), 'Output overlaps inputs')

    # The reference evaluator performs all identity, pool and ranking checks.
    # It writes into a temporary directory, never over the original reports.
    with TemporaryDirectory(prefix='news-evaluation-reference-') as directory:
        reference_dir = Path(directory)
        original = evaluate(source, annotations, reference_dir)
        scenarios = {s['id']: s for s in read_json(source / 'test_scenarios.json')['scenarios']}
        require(set(s['context_type'] for s in scenarios.values()) == set(SUPPORTED_CRITERIA),
                'Unexpected scenario groups')
        with annotations.open(encoding='utf-8-sig', newline='') as stream:
            annotation_rows = list(csv.DictReader(stream))

        query_pools = defaultdict(dict)
        labels = {}
        for row in annotation_rows:
            sid, aid = row['scenario_id'], row['article_id']
            scenario = scenarios[sid]
            q = int(row['query_relevance']) if row['query_relevance'] else None
            labels[sid, aid] = (q, int(row['user_relevance']))
            if scenario['kind'] == 'query':
                pool = query_pools[query_key(scenario)]
                require(aid not in pool or pool[aid] == q, 'Inconsistent same-query label')
                pool[aid] = q

        rankings = read_json(source / 'test_runs.json')['runs']
        ranking_by_key = {(r['scenario_id'], r['config_id']): r for r in rankings}
        detail, changes = [], []
        for old in original['per_run']:
            row = dict(old)
            if row['criterion'] == 'query_relevance':
                scenario = scenarios[row['scenario_id']]
                pool = query_pools[query_key(scenario)]
                ideal = dcg(sorted(pool.values(), reverse=True))
                ranked = ranking_by_key[row['scenario_id'], row['config_id']]['result']['items']
                grades = [labels[row['scenario_id'], item['article_id']][0] for item in ranked]
                row['ndcg_at_10'] = dcg(grades) / ideal if ideal > 0 else None
                if row['ndcg_at_10'] != old['ndcg_at_10']:
                    changes.append({'scenario_id': row['scenario_id'], 'config_id': row['config_id'],
                                    'criterion': row['criterion'], 'ndcg_v1': old['ndcg_at_10'],
                                    'ndcg_v2': row['ndcg_at_10']})
            for metric in METRICS:
                require(row[metric] is None or 0 <= row[metric] <= 1 + 1e-12, 'Metric outside [0,1]')
            detail.append(row)

        main_rows = detail
        group_keys = ('context_type', 'config_id', 'criterion')
        groups = summarize(main_rows, group_keys)
        add_query_counts(main_rows, groups, scenarios)
        by_key = {(r['scenario_id'], r['config_id'], r['criterion']): r for r in detail}
        deltas = []
        for row in main_rows:
            if row['kind'] != 'query' or row['variant'] != 'personalized':
                continue
            baseline = by_key[row['scenario_id'], row['method'] + '_query_only', row['criterion']]
            delta = {k: row[k] for k in ('scenario_id', 'context_type', 'method', 'criterion')}
            for metric in METRICS:
                delta[metric] = row[metric] - baseline[metric] if row[metric] is not None and baseline[metric] is not None else None
            deltas.append(delta)
        delta_groups = summarize(deltas, ('context_type', 'method', 'criterion'))

        distributions = []
        for sid, scenario in scenarios.items():
            criteria = ('user_relevance',) if scenario['kind'] == 'feed' else ('query_relevance', 'user_relevance')
            for criterion in criteria:
                index = 0 if criterion == 'query_relevance' else 1
                grades = [value[index] for (pool_sid, aid), value in labels.items() if pool_sid == sid]
                counts = Counter(grades)
                distributions.append({'scenario_id': sid, 'context_type': scenario['context_type'],
                                      'criterion': criterion, 'pool_size': len(grades),
                                      'grade_0': counts[0], 'grade_1': counts[1], 'grade_2': counts[2]})

        query_pool_summary = []
        for key, pool in query_pools.items():
            counts = Counter(pool.values())
            query_pool_summary.append({'query': key[0], 'information_need': key[1], 'pool_size': len(pool),
                                       'grade_0': counts[0], 'grade_1': counts[1], 'grade_2': counts[2],
                                       'idcg_at_10': dcg(sorted(pool.values(), reverse=True))})
        coverage_groups = defaultdict(list)
        for ranking in rankings:
            coverage_groups[scenarios[ranking['scenario_id']]['context_type'], ranking['config_id']].append(ranking)
        coverage = []
        for (context, config), members in sorted(coverage_groups.items()):
            nonempty = sum(bool(r['result']['items']) for r in members)
            coverage.append({'context_type': context, 'config_id': config, 'n_runs': len(members),
                             'n_nonempty': nonempty, 'n_empty': len(members) - nonempty,
                             'nonempty_fraction': nonempty / len(members)})

        conflict_ids = {sid for sid, s in scenarios.items() if s['context_type'] == 'history_conflict'}
        conflict_grades = [g for (sid, aid), g in labels.items() if sid in conflict_ids]
        audit = {'conflict_pairs': len(conflict_grades),
                 'conflict_user_grade_2': sum(u == 2 for q, u in conflict_grades),
                 'conflict_query_2_user_1': sum(q == 2 and u == 1 for q, u in conflict_grades)}
        # Equal query-only rankings must now have identical query NDCG.
        baseline_values = defaultdict(list)
        baseline_ids = defaultdict(list)
        for row in detail:
            if row['variant'] == 'query_only' and row['criterion'] == 'query_relevance':
                key = query_key(scenarios[row['scenario_id']]), row['config_id']
                baseline_values[key].append(row['ndcg_at_10'])
                baseline_ids[key].append([i['article_id'] for i in ranking_by_key[row['scenario_id'], row['config_id']]['result']['items']])
        for key, ids in baseline_ids.items():
            require(all(items == ids[0] for items in ids), 'Query-only rankings differ for one query')
            require(all(value == baseline_values[key][0] for value in baseline_values[key]),
                    'Same-query baseline NDCG differs')

        output.mkdir(parents=True, exist_ok=True)
        snapshot = output / 'input_snapshot'
        snapshot.mkdir(exist_ok=True)
        copies = {name: source / name for name in INPUT_NAMES}
        copies['test_annotations.csv'] = annotations
        for name, path in copies.items():
            target = snapshot / name
            if target.exists():
                require(file_hash(target) == file_hash(path), 'Snapshot differs: ' + name)
            else:
                shutil.copyfile(path, target)
        reference_output = output / 'reference_calculation'
        reference_output.mkdir(exist_ok=True)
        preserve_reference(reference_output, reference_dir)

    result = {'analysis_version': 3, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'dataset_id': original['dataset_id'], 'input_sha256': input_hashes,
              'analysis_code_sha256': file_hash(Path(__file__)),
              'reference_code_sha256': file_hash(Path(reference.__file__)),
              'reference_comparison': {'input_check': 'exact SHA-256', 'report_check': 'exact structure and identifiers; float tolerance', 'relative_tolerance': REFERENCE_TOLERANCE, 'absolute_tolerance': REFERENCE_TOLERANCE},
              'annotation_pairs': original['annotation_pairs'], 'test_runs': original['test_runs'],
              'n_unique_queries': len(query_pools), 'protocol_v1': original['protocol'],
              'protocol_v3': PROTOCOL, 'main_by_context': groups,
              'per_run': detail, 'personalization_deltas': deltas, 'delta_by_context': delta_groups,
              'assessment_distribution': distributions, 'query_pools': query_pool_summary,
              'coverage': coverage, 'empty_runs': original['empty_runs'], 'label_audit': audit,
              'normalization_changes': changes}
    write_json(output / 'evaluation_metrics.json', result)
    write_csv(output / 'main_results.csv', groups)
    write_csv(output / 'evaluation_per_run.csv', detail)
    write_csv(output / 'personalization_deltas.csv', deltas)
    write_csv(output / 'assessment_distribution.csv', distributions)
    write_csv(output / 'query_pools.csv', query_pool_summary)
    write_csv(output / 'coverage.csv', coverage)
    write_csv(output / 'normalization_changes.csv', changes)
    write_json(output / 'analysis_protocol.json', PROTOCOL)
    keys = ['context_type', 'config_id', 'criterion', 'n_scenarios', 'n_unique_queries', *METRICS]
    delta_keys = ['context_type', 'method', 'criterion', 'n_scenarios', *METRICS]
    report = ('# Результаты офлайн-оценки\n\n'
              'Это дополнительный анализ после просмотра исходных результатов, а не заранее зарегистрированный протокол. '
              'Выдачи, оценки, модели и α не изменялись. Входные данные проверяет evaluate_reference.py.\n\n'
              f"Проверено {result['annotation_pairs']} пар, {result['test_runs']} выдач, {len(scenarios)} контекстов; разных запросов — {len(query_pools)}.\n\n"
              '## Основные таблицы\n\n' + table(groups, keys) + '\n\n'
              'query_relevance — соответствие текущему запросу без учёта истории. user_relevance — совместная релевантность запросу и интересам истории; '
              'для ленты — релевантность интересам. Это оценка заданного сценария, не измерение реальной удовлетворённости пользователя. '
              'Оба критерия сохранены во всех сценариях с запросом.\n\n'
              '## Изменение при добавлении истории\n\n'
              'С историей минус без истории, среднее парных разностей по контекстам.\n\n' + table(delta_groups, delta_keys) + '\n\n'
              '## Доля непустых выдач\n\n' + table(coverage, ['context_type', 'config_id', 'n_runs', 'n_nonempty', 'n_empty', 'nonempty_fraction']) + '\n\n'
              'Пустые выдачи остаются в расчёте качества и дают нулевой Precision.\n\n'
              '## Аудит разметки\n\n'
              f"В конфликтующих контекстах: {audit['conflict_pairs']} пар, {audit['conflict_user_grade_2']} двоек полезности, "
              f"{audit['conflict_query_2_user_1']} пар с query=2 и user=1. Это совместимо с принятым критерием: статья отвечает запросу, но не совпадает с интересами истории. Само отсутствие двоек не доказывает ошибку разметки. Оно ограничивает достижимый Precision внутри наблюдаемого пула.\n\n"
              '## Ограничения\n\n'
              '- Query-IDCG общий для одинаковых запроса и потребности; user-IDCG индивидуален для сценария.\n'
              '- Эталон ограничен размеченным пулом, полнота по всему корпусу неизвестна.\n'
              '- Precision считает только оценку 2, NDCG учитывает и оценку 1. При IDCG=0 NDCG не определена.\n'
              '- При конфликтующей истории нулевой Precision совместной релевантности не означает, что текущий запрос не удовлетворён; для этого есть query_relevance.\n'
              '- Средние равновесны по контекстам; одинаковые запросы и повторяющиеся профили не являются независимыми наблюдениями.\n'
              '- Число сценариев мало; статистическая значимость и реальное пользовательское поведение не проверялись.\n'
              '- Итоговые оценки выставлены автором вручную; независимого второго оценщика не было.\n'
              '- Корпус является архивом 2018 года; текущая новостная актуальность не оценивается.\n'
              '- reference_calculation содержит воспроизведённые первоначальные расчёты; input_snapshot — неизменённые входные файлы.\n')
    (output / 'evaluation_report.md').write_text(report, encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    source = args.input_dir or args.project_root / 'results'
    annotations = args.annotations or source / 'test_annotations.csv'
    output = args.output_dir or args.project_root / 'results/evaluation'
    try:
        result = run_analysis(source, annotations, output)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, 'Ошибка проверки: ' + str(exc) + '\n')
    print(f"Проверено: {result['annotation_pairs']} пар, {result['test_runs']} выдач.")
    print(f"Разных текущих запросов: {result['n_unique_queries']}. Оценки и выдачи сохранены.")
    for row in result['main_by_context']:
        print(row['context_type'], row['config_id'], row['criterion'],
              ' '.join(f'{m}={row[m]:.3f}' if row[m] is not None else f'{m}=undefined' for m in METRICS))
    print('Отчёт:', output / 'evaluation_report.md')


if __name__ == '__main__':
    main()
