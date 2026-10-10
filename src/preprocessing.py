"""Подготовка корпуса Lenta.ru за 2018 год без выборки и фильтрации рубрик."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def normalize_whitespace(series):
    """Нормализует пробелы, сохраняя регистр и пунктуацию."""
    return series.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=root / "data/raw/lenta-ru-news.csv.bz2")
    parser.add_argument("--output", type=Path,
                        default=root / "data/processed/news_2018.csv")
    parser.add_argument("--report", type=Path,
                        default=root / "results/preprocessing_report.json")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")
    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")
    paths = [p.resolve() for p in (args.input, args.output, args.report)]
    if len(set(paths)) != len(paths):
        parser.error("Input, output and report must have different paths")

    columns = ["url", "title", "text", "topic", "tags", "date"]
    parts = []
    raw_rows = invalid_dates = 0
    print("Начинаем чтение датасета…", flush=True)
    for chunk in pd.read_csv(args.input, usecols=columns, dtype=str,
                             chunksize=args.chunksize):
        dates = pd.to_datetime(chunk["date"], errors="coerce")
        invalid_dates += int(dates.isna().sum())
        mask = dates.ge("2018-01-01") & dates.lt("2019-01-01")
        selected = chunk.loc[mask].copy()
        selected["date"] = dates.loc[mask].dt.strftime("%Y-%m-%d")
        if not selected.empty:
            parts.append(selected)
        raw_rows += len(chunk)
        print(f"Прочитано строк: {raw_rows:,}", flush=True)

    if not parts:
        raise SystemExit("Нет статей за 2018 год. Выходные файлы не записаны.")
    df = pd.concat(parts, ignore_index=True)
    year_rows = len(df)

    for column in ["url", "title", "text", "topic", "tags"]:
        df[column] = normalize_whitespace(df[column])
    empty_counts = {column: int(df[column].eq("").sum())
                    for column in ["url", "title", "text"]}
    invalid_content = df[["url", "title", "text"]].eq("").any(axis=1)
    empty_removed = int(invalid_content.sum())
    df = df.loc[~invalid_content].copy()

    # Считаем удаления последовательно, чтобы не учесть одну строку дважды.
    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").copy()
    url_duplicates_removed = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset=["title", "text"], keep="first").copy()
    content_duplicates_removed = before - len(df)
    if df.empty:
        raise SystemExit("После очистки корпус пуст. Выходные файлы не записаны.")

    # Пропуски рубрик и тегов сохраняем: они не определяют отбор статей.
    for column in ["topic", "tags"]:
        df[column] = df[column].replace("", pd.NA)
    df["article_id"] = df["url"].map(
        lambda url: "lenta_" + hashlib.sha256(url.encode("utf-8")).hexdigest()
    )
    if df["article_id"].duplicated().any():
        raise RuntimeError("Article identifiers are not unique")
    df = df.sort_values(["date", "article_id"]).reset_index(drop=True)
    df = df[["article_id"] + columns]

    report = {
        "schema_version": 1,
        "source": str(args.input),
        "raw_rows": raw_rows,
        "invalid_dates_in_raw": invalid_dates,
        "period": {"start_inclusive": "2018-01-01",
                   "end_exclusive": "2019-01-01"},
        "rows_in_period_before_cleaning": year_rows,
        "empty_fields_in_period": empty_counts,
        "removals_sequential": {
            "empty_url_title_or_text": empty_removed,
            "duplicate_url": url_duplicates_removed,
            "duplicate_normalized_title_and_text": content_duplicates_removed,
        },
        "final_rows": len(df),
        "selection": "all eligible rows; no sampling or topic filtering",
        "text_normalization": "collapse whitespace and strip; preserve case and punctuation",
        "deduplication": "URL first, then exact normalized (title, text); keep first source row",
        "missing_topic": int(df["topic"].isna().sum()),
        "missing_tags": int(df["tags"].isna().sum()),
        "min_date": df["date"].min(),
        "max_date": df["date"].max(),
        "month_counts": {str(k): int(v) for k, v in
                         df.groupby(df["date"].str[:7]).size().items()},
        "topic_counts": {str(k): int(v) for k, v in
                         df["topic"].fillna("Не указано").value_counts().items()},
        "pandas_version": pd.__version__,
    }
    if year_rows != len(df) + sum(report["removals_sequential"].values()):
        raise RuntimeError("Cleaning counts do not reconcile")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    print(f"\nСтатей за 2018 год до очистки: {year_rows:,}")
    print(f"Удалено пустых записей: {empty_removed:,}")
    print(f"Удалено повторных URL: {url_duplicates_removed:,}")
    print(f"Удалено дубликатов содержимого: {content_duplicates_removed:,}")
    print(f"Итоговый корпус: {len(df):,}")
    print(f"Без topic: {report['missing_topic']:,}")
    print(f"Корпус сохранён: {args.output}")
    print(f"Отчёт сохранён: {args.report}")


if __name__ == "__main__":
    main()
