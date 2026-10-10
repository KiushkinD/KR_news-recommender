"""Сравнение весов истории на development-сценариях с отчётом HTML и JSON."""

import argparse
import html
import json
from pathlib import Path
from time import perf_counter

from recommender import NewsRecommender

FINANCE = ["инфляция", "ключевая ставка", "банковский сектор"]
SPORT = ["футбол", "хоккей", "спортивные соревнования"]
ALPHAS = [1.0, 0.8, 0.6]
SCENARIOS = [
    {"id": "dev_01", "name": "Широкий запрос, экономические интересы",
     "query": "события в России", "history": FINANCE,
     "assessment": "Сохранить связь с событиями в России; проверить, помогает ли история выбрать экономически интересные материалы."},
    {"id": "dev_02", "name": "Широкий запрос, спортивные интересы",
     "query": "события в России", "history": SPORT,
     "assessment": "Сохранить связь с событиями в России; проверить, помогает ли история выбрать спортивные материалы."},
    {"id": "dev_03", "name": "Конкретный запрос согласуется с историей",
     "query": "ключевая ставка", "history": FINANCE,
     "assessment": "Отдавать приоритет новостям о ключевой ставке и непосредственно связанных решениях; общая экономика сама по себе недостаточна."},
    {"id": "dev_04", "name": "Конкретный запрос противоречит истории",
     "query": "ключевая ставка", "history": SPORT,
     "assessment": "Удовлетворить текущую потребность в новостях о ключевой ставке. Спортивные интересы не должны уводить выдачу от запроса."},
]


def escape(value):
    return html.escape(str(value), quote=True)


def write_html(path, runs, engine):
    parts = ["""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Проверка персонализации</title><style>
body{font-family:system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 20px;color:#18212b;background:#f5f7fa}
h1{font-size:28px}h2{font-size:22px}h3{font-size:17px;margin-top:0}
.scenario{margin:32px 0;padding:24px;background:white;border-radius:12px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.column{padding:16px;border:1px solid #dce3eb;border-radius:8px}
li{margin-bottom:18px}ol{padding-left:22px}.meta{font-size:13px;color:#586575}
details p{font-size:14px;white-space:pre-wrap;max-height:320px;overflow:auto}
summary{cursor:pointer}a{color:#145aa5}.note{background:#e8eef7;padding:16px;border-radius:8px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style><body><h1>Проверка персонализации: development-сценарии</h1>
<p class="note">В каждом столбце показан Top-5. Полный Top-10 сохранён в development_runs.json.
Пересечение выдач показывает изменение состава, но не измеряет качество.
Вес выбирайте по полезности результатов и сохранению текущей потребности.
Эти сценарии не входят в итоговый test.</p>"""]
    article_text = engine.articles.set_index("article_id")["text"]
    lookup = {(run["scenario_id"], run["method"], run["alpha"]): run for run in runs}
    for scenario in SCENARIOS:
        parts.append(f'<section class="scenario"><h2>{escape(scenario["id"])} — {escape(scenario["name"])}</h2>')
        parts.append(f'<p><b>Запрос:</b> {escape(scenario["query"])}<br><b>История:</b> {escape("; ".join(scenario["history"]))}</p>')
        parts.append(f'<p>{escape(scenario["assessment"])}</p>')
        for method in ["tfidf", "e5"]:
            parts.append(f'<h3>{escape(method.upper())}</h3><div class="grid">')
            for alpha in ALPHAS:
                run = lookup[(scenario["id"], method, alpha)]
                result = run["result"]
                label = " — без истории" if alpha == 1.0 else ""
                parts.append(f'<div class="column"><h3>α = {alpha:.1f}{label}</h3>')
                parts.append(f'<p class="meta">Общих статей с α=1.0 в Top-10: {run["common_with_query_only"]}.<br>Прогретый вызов: {result["elapsed_seconds"]:.3f} с.</p>')
                for message in result["messages"]:
                    parts.append(f'<p>{escape(message)}</p>')
                parts.append('<ol>')
                for item in result["items"][:5]:
                    q = f'{item["query_score"]:.4f}' if item["query_score"] is not None else '—'
                    h = f'{item["history_score"]:.4f}' if item["history_score"] is not None else '—'
                    parts.append(f'<li><a href="{escape(item["url"])}" target="_blank" rel="noopener noreferrer">{escape(item["title"])}</a>')
                    parts.append(f'<div class="meta">{escape(item["date"])} · {escape(item["topic"])}<br>score={item["score"]:.4f}; query={q}; history={h}</div>')
                    parts.append(f'<details><summary>Полный текст</summary><p>{escape(article_text.loc[item["article_id"]])}</p></details></li>')
                parts.append('</ol></div>')
            parts.append('</div>')
        parts.append('</section>')
    parts.append('</body></html>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.project_root / "results"
    print("Загружаем рекомендательную систему…", flush=True)
    engine = NewsRecommender(args.project_root, device=args.device)
    runs, warmup_seconds = [], {}
    for method in ["tfidf", "e5"]:
        print(f"{method}: загрузка представления и прогрев…", flush=True)
        started = perf_counter()
        engine.recommend(SCENARIOS[0]["query"], [], method=method, alpha=1.0, top_n=10)
        engine.recommend(SCENARIOS[0]["query"], SCENARIOS[0]["history"],
                         method=method, alpha=0.8, top_n=10)
        warmup_seconds[method] = perf_counter() - started
        for scenario in SCENARIOS:
            baseline_ids = set()
            for alpha in ALPHAS:
                history = [] if alpha == 1.0 else scenario["history"]
                result = engine.recommend(scenario["query"], history,
                                          method=method, alpha=alpha, top_n=10)
                ids = {item["article_id"] for item in result["items"]}
                if alpha == 1.0:
                    baseline_ids = ids
                runs.append({"scenario_id": scenario["id"], "method": method,
                             "alpha": alpha,
                             "common_with_query_only": len(ids & baseline_ids),
                             "result": result})
            print(f"  {scenario['id']}: три веса проверены", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "development_runs.json"
    html_path = output / "development_comparison.html"
    report = {"split": "development", "scenarios": SCENARIOS,
              "alphas": ALPHAS, "top_n": 10, "warmup_seconds": warmup_seconds,
              "timing_note": "One warm call per configuration; descriptive timing, not a repeated latency benchmark.",
              "runs": runs}
    write_html(html_path, runs, engine)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nГотово: {len(runs)} выдачи.")
    print(f"Отчёт для просмотра: {html_path}")
    print(f"Полные результаты: {json_path}")
    print("Выдачи следует оценить по содержимому. Пересечение списков не является оценкой релевантности.")


if __name__ == "__main__":
    main()
