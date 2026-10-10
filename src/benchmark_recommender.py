"""Измерение времени рекомендаций на сохранённых сценариях."""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from recommender import NewsRecommender


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values):
    if not values:
        return {'n': 0, 'mean_ms': None, 'median_ms': None, 'p95_ms': None,
                'min_ms': None, 'max_ms': None, 'std_ms': None}
    milliseconds = [v * 1000 for v in values]
    return {'n': len(values), 'mean_ms': statistics.mean(milliseconds),
            'median_ms': statistics.median(milliseconds),
            'p95_ms': percentile(milliseconds, .95), 'min_ms': min(milliseconds),
            'max_ms': max(milliseconds),
            'std_ms': statistics.stdev(milliseconds) if len(values) > 1 else 0.0}


def environment():
    result = {'os': platform.platform(), 'macos': platform.mac_ver()[0],
              'architecture': platform.machine(), 'python': platform.python_version(),
              'cpu': platform.processor(), 'versions': {}}
    for package in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'sentence-transformers'):
        try:
            result['versions'][package] = version(package)
        except PackageNotFoundError:
            result['versions'][package] = None
    if platform.system() == 'Darwin':
        for name, field in [('machdep.cpu.brand_string', 'cpu'), ('hw.memsize', 'ram_bytes')]:
            try:
                value = subprocess.check_output(['sysctl', '-n', name], text=True,
                                                stderr=subprocess.DEVNULL).strip()
                result[field] = int(value) if field == 'ram_bytes' else value
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
    return result


def cases_from_scenarios(scenarios):
    cases = []
    for scenario in scenarios:
        variants = ['query_only', 'query_and_history'] if scenario['kind'] == 'query' else ['history_only']
        for method in ('tfidf', 'e5'):
            for variant in variants:
                cases.append({'scenario_id': scenario['id'], 'method': method, 'mode': variant,
                              'query': scenario['query'],
                              'history': [] if variant == 'query_only' else scenario['history'],
                              'alpha': 1.0 if variant == 'query_only' else .8, 'top_n': 10})
    return cases


def synchronize(engine):
    encoder = getattr(engine, '_encoder', None)
    if encoder is None:
        return
    import torch
    device = encoder.device
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(engine, case):
    synchronize(engine)
    start = perf_counter()
    result = engine.recommend(case['query'], list(case['history']), method=case['method'],
                              alpha=case['alpha'], top_n=case['top_n'])
    synchronize(engine)
    elapsed = perf_counter() - start
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError('Invalid elapsed time')
    return {'scenario_id': case['scenario_id'], 'method': case['method'], 'mode': case['mode'],
            'elapsed_seconds': elapsed, 'status': result['status'],
            'returned_items': len(result['items'])}


def run(root, repeats, device):
    scenario_path = root / 'results/test_scenarios.json'
    document = json.loads(scenario_path.read_text(encoding='utf-8'))
    if document['alpha'] != .8 or document['top_n'] != 10:
        raise ValueError('Expected the frozen alpha=0.8 and Top-10 protocol')
    cases = cases_from_scenarios(document['scenarios'])
    info = environment()
    print('Загружаем корпус и проверяем согласованность…', flush=True)
    started = perf_counter()
    engine = NewsRecommender(root, device=device)
    corpus_load = perf_counter() - started
    first_calls = []
    # First calls are not full cold starts: existing model and OS file caches may be used.
    for method in ('tfidf', 'e5'):
        print(f'{method}: первый вызов с загрузкой представления…', flush=True)
        first_calls.append(measure(engine, next(c for c in cases if c['method'] == method)))
    print('Прогреваем все сценарии…', flush=True)
    for case in cases:
        measure(engine, case)
    actual_device = str(engine._encoder.device) if getattr(engine, '_encoder', None) is not None else None
    samples = []
    rng = random.Random(42)
    for repeat in range(1, repeats + 1):
        shuffled = list(cases)
        rng.shuffle(shuffled)
        for case in shuffled:
            row = measure(engine, case)
            row['repeat'] = repeat
            samples.append(row)
        print(f'  Повтор {repeat}/{repeats}: {len(shuffled)} вызовов', flush=True)
    groups = defaultdict(list)
    for row in samples:
        groups[row['method'], row['mode']].append(row)
    summaries = []
    for (method, mode), members in sorted(groups.items()):
        entry = {'method': method, 'mode': mode, 'statuses': dict(Counter(r['status'] for r in members))}
        entry['all_calls'] = summarize([r['elapsed_seconds'] for r in members])
        entry['successful_calls'] = summarize([r['elapsed_seconds'] for r in members
                                               if r['status'] == 'ok' and r['returned_items'] > 0])
        summaries.append(entry)
    arrays = {}
    matrix = getattr(engine, '_tfidf', None)
    if matrix is not None:
        arrays['tfidf_csr_bytes'] = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    embeddings = getattr(engine, '_embeddings', None)
    if embeddings is not None:
        arrays['e5_array_bytes'] = int(embeddings.nbytes)
    output = root / 'results'
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'environment': info, 'requested_device': device, 'e5_device': actual_device,
              'tfidf_device': 'cpu', 'corpus_size': len(engine.article_ids),
              'scenario_file_sha256': hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
              'corpus_model_text_sha256': engine.embedding_report['corpus_model_text_sha256'],
              'model_revision': engine.embedding_report.get('model_revision'),
              'repeats': repeats, 'cases_per_repeat': len(cases), 'top_n': 10, 'alpha': .8,
              'corpus_loading_seconds': corpus_load, 'first_calls': first_calls,
              'representation_array_bytes': arrays, 'summary': summaries,
              'measurement_scope': 'Wall time of recommend plus final accelerator synchronization. Includes input encoding, profile, full ranking and result metadata. Excludes browser rendering and waiting for concurrent users.',
              'limitations': ['Single process and one hardware configuration; no concurrency or network benchmark.',
                              'First calls may use previously downloaded models and filesystem cache; not a cache-free cold start.',
                              'Repeated queries are re-encoded: recommendation outputs are not cached.',
                              'Array sizes are logical representation sizes, not process RAM or total model memory.',
                              'Successful-only summaries exclude OOV/empty outputs; all-call summaries and statuses are also reported.'],
              'samples': samples}
    json_path = output / 'performance_report.json'
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'performance_samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    lines = ['# Время получения рекомендаций', '',
             f'Корпус: {len(engine.article_ids):,} статей. Повторов: {repeats}. Top-10. E5: {actual_device}; TF-IDF: CPU.', '',
             'В таблице — успешные вызовы после прогрева. Пустые выдачи учитываются отдельно в JSON.', '',
             '| Метод | Режим | Успешных вызовов | Среднее, мс | Медиана, мс | P95, мс |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for entry in summaries:
        stats = entry['successful_calls']
        show = lambda value: '—' if value is None else f'{value:.2f}'
        lines.append(f"| {entry['method']} | {entry['mode']} | {stats['n']} | {show(stats['mean_ms'])} | {show(stats['median_ms'])} | {show(stats['p95_ms'])} |")
        print(entry['method'], entry['mode'], 'успешных:', stats['n'], 'среднее, мс:', show(stats['mean_ms']),
              'P95, мс:', show(stats['p95_ms']), flush=True)
    lines += ['', 'Первый вызов и загрузка корпуса приведены отдельно в JSON и не входят в таблицу.',
              'P95 — эмпирический 95-й процентиль с линейной интерполяцией, а не доверительный интервал.',
              'TF-IDF выполняется на CPU, E5 использует указанное устройство: это сравнение реализованных конфигураций.',
              'Результаты относятся к данной аппаратной конфигурации и последовательным вызовам в одном процессе.',
              'Размеры массивов в JSON не являются измерением общего потребления оперативной памяти.',
              'Время интерфейса, сети и очереди запросов в эти измерения не входит.']
    (output / 'performance_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\nОтчёт сохранён:', json_path, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be at least 2')
    run(args.project_root, args.repeats, args.device)


if __name__ == '__main__':
    main()
