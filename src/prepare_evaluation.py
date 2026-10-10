"""Подготовка тестовых выдач и формы разметки без указания методов и позиций."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from recommender import NewsRecommender

ALPHA = 0.8
TOP_N = 10
SEED = 20261005
ANNOTATION_COLUMNS = ["scenario_id", "pool_id", "article_id",
                      "query_relevance", "user_relevance", "comment"]

PROTOCOL = {
    "scope": "Новости фиксированного архива за 2018 год. Актуальность на сегодняшнюю дату не оценивается.",
    "query_relevance": {
        "0": "Не отвечает текущему запросу и описанной информационной потребности.",
        "1": "Связана с запросом косвенно или раскрывает лишь второстепенную часть потребности.",
        "2": "Непосредственно раскрывает тему запроса и помогает удовлетворить описанную потребность.",
    },
    "user_relevance": {
        "0": "Не полезна пользователю в данном сценарии.",
        "1": "Частично полезна, связь с текущей потребностью или интересами ограничена.",
        "2": "Явно полезна с учётом описанной потребности и интересов пользователя.",
    },
    "rules": [
        "Оценивайте заголовок и полный текст, а не только отдельные совпадающие слова.",
        "Для широкого запроса соответствие интересам истории помогает отличить особенно полезные новости от общетематических.",
        "Для конкретного запроса текущая потребность имеет приоритет. Совпадение с историей не компенсирует уход от запроса.",
        "Если query_relevance = 0, то user_relevance = 0 в сценариях с текущим запросом.",
        "В ленте без текущего запроса оценивается только user_relevance; query_relevance остаётся пустой.",
        "Одна статья может получить разные оценки в разных пользовательских сценариях.",
        "Приоритет источника, популярность, личное согласие с новостью и совпадение рубрики не являются критериями релевантности.",
    ],
    "precision_positive_label": 2,
    "ndcg_gain": "2**relevance - 1",
    "ndcg_ideal_ranking": "Идеальное ранжирование размеченного объединённого пула данного сценария, первые 10 статей.",
    "zero_idcg_policy": "При IDCG = 0 NDCG не определена: исключить из среднего NDCG и явно указать число таких сценариев. Precision остаётся 0.",
    "short_list_policy": "Знаменатели Precision@5 и Precision@10 равны 5 и 10; отсутствующие позиции дают нулевой вклад.",
}


def make_scenarios():
    space = ["астрономия", "космические исследования", "спутники"]
    medicine = ["медицинские исследования", "вакцины", "генетика"]
    cinema = ["кинопремьеры", "режиссёры", "кинофестивали"]
    books = ["книги", "литературные премии", "писатели"]
    pairs = [
        ("новости науки и технологий", "broad",
         "Новости о научных исследованиях, открытиях и технологических разработках.",
         space, medicine),
        ("новости культуры", "broad",
         "Новости культурной жизни: новые произведения, события, премии и деятельность авторов.",
         cinema, books),
        ("экономические санкции против России", "specific",
         "Новости об экономических санкциях, их введении, изменении и последствиях для России. Спортивные дисквалификации сами по себе не отвечают этой потребности.",
         ["международная торговля", "российские компании", "финансовые рынки"],
         ["футбол", "хоккей", "теннис"]),
        ("чемпионат мира по футболу 2018", "specific",
         "Новости о матчах, участниках, результатах и организации футбольного чемпионата мира 2018 года. Общая экономика без содержательной связи с турниром не подходит.",
         ["футбол", "сборные команды", "результаты матчей"],
         ["инфляция", "валютный курс", "банковский сектор"]),
        ("ипотека и покупка жилья", "specific",
         "Новости об ипотечных кредитах, условиях приобретения жилья и изменениях на жилищном рынке, значимых для покупателя.",
         ["рынок недвижимости", "цены на квартиры", "ипотечные кредиты"],
         ["космические полёты", "астрономия", "исследования планет"]),
        ("запуски космических ракет", "specific",
         "Новости о подготовке, проведении и результатах запусков космических ракет. Запуски военных ракет без космической миссии не подходят.",
         ["космические миссии", "спутники", "ракетные двигатели"],
         cinema),
    ]
    scenarios = []
    for pair_index, (query, kind, need, history_a, history_b) in enumerate(pairs, 1):
        for user_index, history in enumerate([history_a, history_b]):
            index = len(scenarios) + 1
            context = "broad" if kind == "broad" else (
                "history_aligned" if user_index == 0 else "history_conflict")
            note = (
                "Запрос широкий: история уточняет, какие материалы особенно полезны. Связь с общей темой запроса сохраняется."
                if kind == "broad" else
                "Запрос конкретный: оценка полезности прежде всего определяется текущей потребностью, даже если история относится к другой теме."
            )
            scenarios.append({"id": f"test_{index:02d}", "pair_id": f"pair_{pair_index:02d}",
                              "kind": "query", "context_type": context,
                              "query": query, "history": list(history),
                              "information_need": need, "assessment_note": note})
    for index, (name, history) in enumerate([
        ("Космос и астрономия", space), ("Медицинская наука", medicine),
        ("Кино", cinema), ("Книги и литература", books),
    ], 1):
        scenarios.append({"id": f"feed_{index:02d}", "pair_id": None,
                          "kind": "feed", "context_type": "history_only",
                          "query": "", "history": list(history),
                          "information_need": f"Персональная лента по интересам: {name}.",
                          "assessment_note": "Оцените полезность статьи для человека с указанной историей. Совпадение одного слова без содержательной связи с интересами недостаточно."})
    return scenarios


HTML_TEMPLATE = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Разметка тестовых рекомендаций</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;background:#f3f6fa;color:#192534;margin:0}
main{max-width:1150px;margin:24px auto;padding:0 20px}h1{font-size:26px;margin:0 0 12px}h2{font-size:21px;margin:0 0 8px}
.card{background:white;border:1px solid #dce4ef;border-radius:12px;padding:20px;margin:16px 0}.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button,select,textarea{font:inherit;border:1px solid #b8c6d8;border-radius:7px;padding:9px;background:white}button{cursor:pointer}button.primary{background:#175fcc;color:white;border-color:#175fcc}
button:disabled{opacity:.45;cursor:default}select{max-width:100%}textarea{width:100%;min-height:70px}label{display:block;font-weight:600;margin-bottom:6px}
.muted{font-size:14px;color:#526175}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.article{white-space:pre-wrap;max-height:47vh;overflow:auto;padding-right:10px}
.error{color:#a02929;min-height:24px}progress{width:100%;height:15px}details{margin:12px 0}summary{cursor:pointer;font-weight:600}.hidden{display:none}
@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:0 12px}.article{max-height:40vh}}
</style></head><body><main>
<h1>Разметка тестовых рекомендаций</h1>
<p class="muted">Методы, позиции и оценки алгоритмов скрыты. Статьи перемешаны внутри сценария.
Оценки относятся к паре «сценарий — статья». Регулярно скачивайте CSV, чтобы сохранить свою работу.</p>
<div class="bar"><button id="export" class="primary">Скачать оценки CSV</button><button id="import">Загрузить сохранённый CSV</button>
<input id="file" class="hidden" type="file" accept=".csv,text/csv"></div>
<p id="storage" class="muted"></p><p id="progressText"></p><progress id="progress" value="0" max="1"></progress>
<details class="card"><summary>Правила оценки</summary><div id="rules"></div></details>
<div class="bar"><select id="scenarioSelect" aria-label="Сценарий"></select><button id="nextMissing">Следующая незавершённая</button></div>
<section class="card"><h2 id="scenarioTitle"></h2><p id="query"></p><p id="history"></p><p id="need"></p><p id="note" class="muted"></p></section>
<section class="card"><p id="position" class="muted"></p><h2 id="title"></h2><p id="date" class="muted"></p><div id="text" class="article"></div></section>
<section class="card"><div class="grid"><div><label for="q">Соответствие текущему запросу</label><select id="q"><option value="">Выберите оценку</option><option value="0">0 — нерелевантна</option><option value="1">1 — частично релевантна</option><option value="2">2 — релевантна</option></select></div>
<div><label for="u">Полезность пользователю</label><select id="u"><option value="">Выберите оценку</option><option value="0">0 — не полезна</option><option value="1">1 — частично полезна</option><option value="2">2 — явно полезна</option></select></div></div>
<p id="error" class="error"></p><label for="comment">Комментарий — необязательно</label><textarea id="comment"></textarea>
<div class="bar"><button id="previous">Назад</button><button id="next" class="primary">Далее</button></div></section>
<p class="muted">Пустая оценка означает отсутствие разметки. Она не заменяется нулём. В ленте без нового запроса заполняется только полезность пользователю.</p>
</main><script type="application/json" id="payload">__PAYLOAD__</script><script>
"use strict";
const data = JSON.parse(document.getElementById('payload').textContent);
const rows = data.rows, scenarios = new Map(data.scenarios.map(s=>[s.id,s]));
const byPool = new Map(rows.map(r=>[r.pool_id,r]));
const key = 'news-recommender-test-' + data.dataset_id;
const el = id=>document.getElementById(id);
const grade = v=>['','0','1','2'].includes(v);
let state={}, index=0;
try { state=JSON.parse(localStorage.getItem(key)||'{}'); }
catch(e) { state={}; }
if (!state || typeof state!=='object' || Array.isArray(state)) state={};
for(const [id,value] of Object.entries(state)){
  if(!byPool.has(id)||!value||!grade(String(value.q??''))||!grade(String(value.u??''))) delete state[id];
  else state[id]={q:String(value.q??''),u:String(value.u??''),comment:String(value.comment??'')};
}
const currentState = row=>state[row.pool_id]||{q:'',u:'',comment:''};
function isDone(row){
  const s=currentState(row), scenario=scenarios.get(row.scenario_id);
  if(!['0','1','2'].includes(s.u)) return false;
  if(scenario.kind==='feed') return s.q==='';
  return ['0','1','2'].includes(s.q) && !(s.q==='0'&&s.u!=='0');
}
function persist(){
  try {localStorage.setItem(key,JSON.stringify(state));el('storage').textContent='Оценки сохраняются в этом браузере. CSV позволяет перенести их и сохранить отдельно.';}
  catch(e){el('storage').textContent='Автосохранение недоступно. Скачивайте CSV перед закрытием страницы.';}
}
function updateProgress(){
  const complete=rows.filter(isDone).length;
  el('progressText').textContent=`Завершено: ${complete} из ${rows.length}`;
  el('progress').max=rows.length;el('progress').value=complete;
}
function validateCurrent(){
  const row=rows[index],s=currentState(row),scenario=scenarios.get(row.scenario_id);
  el('error').textContent=scenario.kind==='query'&&s.q==='0'&&s.u!==''&&s.u!=='0'
    ? 'Если статья нерелевантна текущему запросу, полезность в этом сценарии должна быть 0.' : '';
}
function render(){
  const row=rows[index],scenario=scenarios.get(row.scenario_id),s=currentState(row);
  el('scenarioSelect').value=scenario.id;
  el('scenarioTitle').textContent=scenario.id+' — '+(scenario.kind==='feed'?'Персональная лента':'Текущий запрос и история');
  el('query').textContent='Запрос: '+(scenario.query||'отсутствует');
  el('history').textContent='История: '+scenario.history.join('; ');
  el('need').textContent='Потребность: '+scenario.information_need;
  el('note').textContent=scenario.assessment_note;
  el('position').textContent=`Статья ${index+1} из ${rows.length} · идентификатор разметки ${row.pool_id}`;
  el('title').textContent=row.title;el('date').textContent='Опубликована: '+row.date;
  el('text').textContent=row.text;
  el('q').disabled=scenario.kind==='feed';el('q').value=scenario.kind==='feed'?'':s.q;
  el('u').value=s.u;el('comment').value=s.comment;
  el('previous').disabled=index===0;el('next').disabled=index===rows.length-1;
  validateCurrent();updateProgress();
}
function saveCurrent(){
  const row=rows[index],scenario=scenarios.get(row.scenario_id);
  state[row.pool_id]={q:scenario.kind==='feed'?'':el('q').value,u:el('u').value,comment:el('comment').value};
  persist();validateCurrent();updateProgress();
}
for(const scenario of data.scenarios){
  const option=document.createElement('option');option.value=scenario.id;
  option.textContent=scenario.id+' — '+(scenario.query||scenario.information_need);
  el('scenarioSelect').append(option);
}
el('scenarioSelect').onchange=()=>{index=rows.findIndex(r=>r.scenario_id===el('scenarioSelect').value);render();};
el('previous').onclick=()=>{index=Math.max(0,index-1);render();};
el('next').onclick=()=>{index=Math.min(rows.length-1,index+1);render();};
el('nextMissing').onclick=()=>{
  for(let offset=1;offset<=rows.length;offset++){
    const candidate=(index+offset)%rows.length;
    if(!isDone(rows[candidate])){index=candidate;render();return;}
  }
  alert('Все статьи размечены. Скачайте CSV и сохраните его в results/test_annotations.csv.');
};
el('q').onchange=saveCurrent;el('u').onchange=saveCurrent;el('comment').oninput=saveCurrent;
function csvText(){
  const quote=v=>'"'+String(v).replaceAll('"','""')+'"';
  const lines=[data.columns.map(quote).join(',')];
  for(const row of rows){
    const s=currentState(row);
    lines.push([row.scenario_id,row.pool_id,row.article_id,s.q,s.u,s.comment].map(quote).join(','));
  }
  return '\ufeff'+lines.join('\r\n')+'\r\n';
}
el('export').onclick=()=>{
  const blob=new Blob([csvText()],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob);
  const link=document.createElement('a');link.href=url;link.download='test_annotations.csv';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
function parseCSV(text){
  text=text.replace(/^\ufeff/,'');let result=[],row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){
      if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;
    }else if(c===','&&!quoted){row.push(field);field='';}
    else if((c==='\n'||c==='\r')&&!quoted){
      if(c==='\r'&&text[i+1]==='\n')i++;
      row.push(field);if(row.some(v=>v!==''))result.push(row);row=[];field='';
    }else field+=c;
  }
  if(quoted)throw Error('Незакрытая кавычка в CSV.');
  if(field!==''||row.length){row.push(field);result.push(row);}
  return result;
}
function importCSV(text){
  const parsed=parseCSV(text);if(!parsed.length)throw Error('CSV пуст.');
  const header=parsed.shift(), positions=data.columns.map(c=>header.indexOf(c));
  if(positions.some(p=>p<0))throw Error('В CSV отсутствуют необходимые столбцы.');
  const pending={},seen=new Set();
  for(const values of parsed){
    if(values.length!==header.length)throw Error('Некорректное число столбцов в CSV.');
    const [scenarioId,poolId,articleId,q,u,comment]=positions.map(p=>values[p]);
    const row=byPool.get(poolId);
    if(!row||row.article_id!==articleId||row.scenario_id!==scenarioId)throw Error('CSV относится к другому пулу статей.');
    if(seen.has(poolId))throw Error('Повторный идентификатор разметки в CSV.');seen.add(poolId);
    if(!grade(q)||!grade(u))throw Error('Допустимые оценки: пусто, 0, 1, 2.');
    if(scenarios.get(scenarioId).kind==='feed'&&q!=='')throw Error('Для ленты оценка текущего запроса должна быть пустой.');
    pending[poolId]={q,u,comment};
  }
  Object.assign(state,pending);persist();render();return seen.size;
}
el('import').onclick=()=>el('file').click();
el('file').onchange=async()=>{
  const file=el('file').files[0];if(!file)return;
  try{const count=importCSV(await file.text());alert('Загружено записей: '+count);}
  catch(e){alert('Не удалось загрузить CSV: '+e.message);}
  finally{el('file').value='';}
};
const rules=el('rules');
for(const [heading,field] of [['Соответствие запросу','query_relevance'],['Полезность пользователю','user_relevance']]){
  const h=document.createElement('h3');h.textContent=heading;rules.append(h);
  const list=document.createElement('ul');
  for(const [value,description] of Object.entries(data.protocol[field])){
    const li=document.createElement('li');li.textContent=value+' — '+description;list.append(li);
  }rules.append(list);
}
const list=document.createElement('ul');
for(const text of [data.protocol.scope,...data.protocol.rules]){
  const li=document.createElement('li');li.textContent=text;list.append(li);
}rules.append(list);
persist();render();
</script></body></html>'''


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    output = args.project_root / "results"
    html_path = args.project_root / "data/processed/evaluation/test_annotation.html"
    paths = {
        "scenarios": output / "test_scenarios.json",
        "runs": output / "test_runs.json",
        "manifest": output / "test_pool_manifest.json",
        "annotations": output / "test_annotations.csv",
        "html": html_path,
    }
    existing = [str(path) for key, path in paths.items() if key != "scenarios" and path.exists()]
    if existing:
        parser.error("Тестовые файлы уже существуют. Не перезаписываем разметку и результаты: " + "; ".join(existing))
    scenarios = make_scenarios()
    scenario_document = {"split": "test", "alpha": ALPHA, "top_n": TOP_N,
                         "seed": SEED, "protocol": PROTOCOL, "scenarios": scenarios}
    if paths["scenarios"].exists():
        previous = json.loads(paths["scenarios"].read_text(encoding="utf-8"))
        if previous != scenario_document:
            parser.error("Существующий протокол теста отличается. Автоматически менять его нельзя.")
    output.mkdir(parents=True, exist_ok=True)
    # Persist the definitions before looking at the algorithms' test results.
    paths["scenarios"].write_text(json.dumps(scenario_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Зафиксировано 12 сценариев с запросом и 4 сценария ленты.", flush=True)
    print("Загружаем рекомендательную систему…", flush=True)
    engine = NewsRecommender(args.project_root, device=args.device)
    runs = []
    for scenario in scenarios:
        for method in ["tfidf", "e5"]:
            settings = [("personalized", scenario["history"], ALPHA)]
            if scenario["kind"] == "query":
                settings.insert(0, ("query_only", [], 1.0))
            for variant, history, alpha in settings:
                result = engine.recommend(scenario["query"], history, method=method,
                                          alpha=alpha, top_n=TOP_N)
                config_id = method + "_" + variant
                runs.append({"scenario_id": scenario["id"], "config_id": config_id,
                             "method": method, "variant": variant, "result": result})
        print(f"  {scenario['id']}: выдачи получены", flush=True)
    pool = []
    article_lookup = engine.articles.set_index("article_id")
    for index, scenario in enumerate(scenarios):
        ids = sorted({item["article_id"] for run in runs if run["scenario_id"] == scenario["id"]
                      for item in run["result"]["items"]})
        random.Random(SEED + index).shuffle(ids)
        if not ids:
            raise RuntimeError(f"Пул пуст для сценария {scenario['id']}")
        for position, article_id in enumerate(ids, 1):
            article = article_lookup.loc[article_id]
            pool.append({"scenario_id": scenario["id"],
                         "pool_id": f"{scenario['id']}_{position:03d}",
                         "article_id": article_id,
                         "title": str(article["title"]), "date": str(article["date"]),
                         "text": str(article["text"])})
    assignments = [{key: row[key] for key in ["scenario_id", "pool_id", "article_id"]}
                   for row in pool]
    identity_data = {"scenarios": scenarios, "assignments": assignments,
                     "corpus_model_text_sha256": engine.embedding_report["corpus_model_text_sha256"],
                     "model_revision": engine.embedding_report.get("model_revision"),
                     "alpha": ALPHA, "top_n": TOP_N}
    dataset_id = hashlib.sha256(json.dumps(identity_data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    payload = {"dataset_id": dataset_id, "columns": ANNOTATION_COLUMNS,
               "protocol": PROTOCOL, "scenarios": scenarios, "rows": pool}
    # Escape HTML-significant characters inside the JSON script element.
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(HTML_TEMPLATE.replace("__PAYLOAD__", payload_json), encoding="utf-8")
    with paths["annotations"].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for row in assignments:
            writer.writerow(dict(row, query_relevance="", user_relevance="", comment=""))
    manifest = dict(identity_data, dataset_id=dataset_id,
                    annotation_columns=ANNOTATION_COLUMNS,
                    pool_size=len(pool),
                    pool_counts={s["id"]: sum(r["scenario_id"] == s["id"] for r in pool) for s in scenarios},
                    html_path=str(html_path))
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["runs"].write_text(json.dumps({"split": "test", "alpha": ALPHA,
                                         "top_n": TOP_N, "dataset_id": dataset_id,
                                         "runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    query_pairs = sum(s["kind"] == "query" for s in scenarios)
    feed_pairs = sum(s["kind"] == "feed" for s in scenarios)
    print(f"\nГотово. Сценариев с запросом: {query_pairs}; лент: {feed_pairs}")
    print(f"Выдач алгоритмов: {len(runs)}")
    print(f"Пар «сценарий — статья» для оценки: {len(pool)}")
    for scenario in scenarios:
        print(f"  {scenario['id']}: {manifest['pool_counts'][scenario['id']]} статей")
    print(f"Форма разметки: {html_path}")
    print(f"CSV для итоговых оценок: {paths['annotations']}")
    print("До завершения разметки используйте HTML-форму; test_runs.json содержит открытые методы и позиции.")
    print("После разметки скачайте CSV и замените им results/test_annotations.csv.")


if __name__ == "__main__":
    main()
