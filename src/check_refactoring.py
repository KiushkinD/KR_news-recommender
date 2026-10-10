"""Сравнение нового рекомендателя с сохранёнными экспериментальными выдачами.

Запуск из проекта: .venv/bin/python -u src/check_refactoring.py
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_results(expected, actual, tolerance):
    differences = []
    max_score_delta = 0.0

    def compare(left, right, location):
        nonlocal max_score_delta
        if isinstance(left, dict) and isinstance(right, dict):
            left_keys, right_keys = set(left), set(right)
            if location == 'result':
                left_keys.discard('elapsed_seconds')
                right_keys.discard('elapsed_seconds')
            if left_keys != right_keys:
                differences.append(location + ': изменился набор полей')
            for key in sorted(left_keys & right_keys):
                compare(left[key], right[key], location + '.' + key)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(location + ': изменилась длина списка')
            for index, (old, new) in enumerate(zip(left, right)):
                compare(old, new, '{}[{}]'.format(location, index))
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isfinite(left) or not math.isfinite(right):
                differences.append(location + ': неконечное число')
                return
            delta = abs(left - right)
            if location.endswith(('.score', '.query_score', '.history_score')):
                max_score_delta = max(max_score_delta, delta)
                equal = delta <= tolerance
            else:
                equal = left == right
            if not equal:
                differences.append('{}: {} -> {}'.format(location, left, right))
        elif left != right or type(left) is not type(right):
            differences.append('{}: {!r} -> {!r}'.format(location, left, right))

    compare(expected, actual, 'result')
    return differences, max_score_delta


def run(root, device, tolerance):
    paths = [root / 'results/test_runs.json', root / 'results/test_scenarios.json']
    annotation = root / 'results/test_annotations.csv'
    if annotation.exists():
        paths.append(annotation)
    before = {str(path.relative_to(root)): fingerprint(path) for path in paths}
    frozen = json.loads(paths[0].read_text(encoding='utf-8-sig'))
    scenarios_doc = json.loads(paths[1].read_text(encoding='utf-8-sig'))
    scenarios = {s['id']: s for s in scenarios_doc['scenarios']}
    runs = frozen['runs']
    if len(scenarios) != 16 or len(runs) != 56:
        raise ValueError('Ожидались 16 сценариев и 56 сохранённых выдач')
    keys = [(r['scenario_id'], r['config_id']) for r in runs]
    if len(set(keys)) != len(keys):
        raise ValueError('В сохранённых выдачах есть повторные конфигурации')
    expected_keys = set()
    for scenario in scenarios.values():
        variants = ('query_only', 'personalized') if scenario['kind'] == 'query' else ('personalized',)
        expected_keys.update((scenario['id'], method + '_' + variant)
                             for method in ('tfidf', 'e5') for variant in variants)
    if set(keys) != expected_keys:
        raise ValueError('Набор выдач не соответствует сохранённым сценариям')

    # Импортируется именно recommender.py рядом с этим скриптом в src/.
    from recommender import NewsRecommender
    print('Загружаем текущую реализацию и готовые представления…', flush=True)
    engine = NewsRecommender(root, device=device)
    comparisons = []
    current_runs = []
    for index, saved in enumerate(runs, 1):
        scenario = scenarios[saved['scenario_id']]
        method = saved['method']
        config = saved['config_id']
        if method not in ('tfidf', 'e5') or not config.startswith(method + '_'):
            raise ValueError('Неизвестная конфигурация: ' + config)
        history = [] if config.endswith('_query_only') else list(scenario['history'])
        expected = saved['result']
        try:
            actual = engine.recommend(
                query=scenario['query'], history=history, method=method,
                alpha=expected['alpha_requested'], top_n=frozen['top_n'],
            )
            differences, delta = compare_results(expected, actual, tolerance)
            current_runs.append(dict(saved, result=actual))
        except Exception as error:
            differences = [type(error).__name__ + ': ' + str(error)]
            delta = None
        entry = {'scenario_id': scenario['id'], 'config_id': config,
                 'passed': not differences, 'max_score_delta': delta,
                 'differences': differences}
        comparisons.append(entry)
        print('{}/{} {} {}: {}'.format(index, len(runs), scenario['id'], config,
                                      'совпадает' if entry['passed'] else 'РАСХОЖДЕНИЕ'), flush=True)

    after = {str(path.relative_to(root)): fingerprint(path) for path in paths}
    unchanged = before == after
    passed = sum(entry['passed'] for entry in comparisons)
    success = unchanged and passed == len(runs)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'passed': success, 'runs_checked': len(runs), 'runs_matched': passed,
              'score_absolute_tolerance': tolerance, 'input_files_unchanged': unchanged,
              'input_sha256_before': before, 'input_sha256_after': after,
              'recommender_sha256': fingerprint(root / 'src/recommender.py'),
              'device_requested': device,
              'max_score_delta': max((e['max_score_delta'] for e in comparisons
                                      if e['max_score_delta'] is not None), default=0.0),
              'comparisons': comparisons}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = root / 'results' / ('refactoring_check_' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'current_runs.json').write_text(
        json.dumps(dict(frozen, runs=current_runs), ensure_ascii=False, indent=2,
                   allow_nan=False) + '\n', encoding='utf-8')
    print('\nСовпало выдач: {}/{}'.format(passed, len(runs)))
    print('Максимальное расхождение оценок: {:.3e}'.format(report['max_score_delta']))
    print('Исходные экспериментальные файлы сохранены:', unchanged)
    print('Отчёт:', output / 'report.json')
    print('Проверка пройдена.' if success else 'Проверка НЕ пройдена: см. differences в отчёте.')
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--score-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    if not math.isfinite(args.score_tolerance) or args.score_tolerance < 0:
        parser.error('Допуск должен быть конечным неотрицательным числом')
    try:
        success = run(args.project_root.resolve(), args.device, args.score_tolerance)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, 'Ошибка проверки: {}\n'.format(error))
    parser.exit(0 if success else 1)


if __name__ == '__main__':
    main()
