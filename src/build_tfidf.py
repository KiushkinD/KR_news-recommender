"""TF-IDF по тем же фрагментам новостей, которые используются E5.

Словарь и IDF определяются по корпусу кандидатов без оценок релевантности.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path,
                        default=root / "data/processed/news_2018_model_text.csv")
    parser.add_argument("--models-dir", type=Path, default=root / "models")
    parser.add_argument("--embedding-report", type=Path,
                        default=root / "results/embedding_report.json")
    parser.add_argument("--report", type=Path,
                        default=root / "results/tfidf_report.json")
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args()
    if args.max_features < 1 or args.min_df < 1:
        parser.error("--max-features and --min-df must be positive")
    ids_path = args.models_dir / "news_article_ids.json"
    embeddings_path = args.models_dir / "news_embeddings.npy"
    vectorizer_path = args.models_dir / "tfidf_vectorizer.joblib"
    matrix_path = args.models_dir / "news_tfidf.npz"
    inputs = [args.texts, ids_path, embeddings_path, args.embedding_report]
    for path in inputs:
        if not path.is_file():
            parser.error(f"Required file not found: {path}")
    outputs = [vectorizer_path, matrix_path, args.report]
    resolved = [p.resolve() for p in inputs + outputs]
    if len(resolved) != len(set(resolved)):
        parser.error("Input and output paths must be distinct")

    print("Загружаем библиотеки…", flush=True)
    import joblib
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("Читаем общий текст для TF-IDF и E5…", flush=True)
    df = pd.read_csv(args.texts, usecols=["article_id", "model_text"],
                     dtype=str, keep_default_na=False)
    if (
        df.empty
        or df["article_id"].eq("").any()
        or df["article_id"].duplicated().any()
    ):
        parser.error("Expected nonempty corpus with unique article_id values")
    if df["model_text"].str.strip().eq("").any():
        parser.error("Shared model_text must be nonempty")
    article_ids = df["article_id"].tolist()
    texts = df["model_text"].tolist()
    embedding_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if article_ids != embedding_ids:
        parser.error("Article order differs from E5. Rebuild consistent features first.")
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (len(df), 384):
        parser.error(f"Unexpected E5 shape: {embeddings.shape}")
    embedding_report = json.loads(args.embedding_report.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256()
    for article_id, text in zip(article_ids, texts):
        fingerprint.update(json.dumps([article_id, text], ensure_ascii=False).encode("utf-8"))
        fingerprint.update(b"\n")
    corpus_hash = fingerprint.hexdigest()
    if corpus_hash != embedding_report.get("corpus_model_text_sha256"):
        parser.error("Shared text differs from text used for E5. Rebuild embeddings first.")
    print(f"Порядок статей и содержимое совпадают с E5. Статей: {len(df):,}",
          flush=True)

    # Используем общий текст без E5-префикса; лемматизация не применяется.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=0.95,
        max_features=args.max_features,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float32,
    )
    print("Строим словарь и TF-IDF-матрицу…", flush=True)
    started = perf_counter()
    try:
        matrix = vectorizer.fit_transform(texts).tocsr()
    except ValueError as error:
        parser.error(f"TF-IDF fitting failed: {error}")
    fit_seconds = perf_counter() - started
    if not np.isfinite(matrix.data).all():
        raise RuntimeError("TF-IDF contains nonfinite values")
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    nonzero = norms > 0
    if not np.allclose(norms[nonzero], 1.0, atol=1e-4):
        raise RuntimeError("Nonzero TF-IDF rows must be L2-normalized")
    zero_ids = [article_ids[i] for i in np.flatnonzero(~nonzero)]
    memory_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    params = vectorizer.get_params(deep=False)
    params["dtype"] = "float32"
    params["ngram_range"] = list(params["ngram_range"])
    report = {
        "method": "word TF-IDF",
        "corpus_size": len(df),
        "shared_text_path": str(args.texts),
        "corpus_model_text_sha256": corpus_hash,
        "article_ids_path": str(ids_path),
        "vectorizer_path": str(vectorizer_path),
        "matrix_path": str(matrix_path),
        "vectorizer_parameters": params,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "matrix_shape": list(matrix.shape),
        "matrix_format": matrix.format,
        "matrix_dtype": str(matrix.dtype),
        "nonzero_entries": int(matrix.nnz),
        "matrix_density": matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
        "csr_memory_bytes": int(memory_bytes),
        "zero_rows": len(zero_ids),
        "zero_row_article_ids": zero_ids,
        "fit_seconds": fit_seconds,
        "versions": {name: version(name) for name in
                     ["pandas", "numpy", "scikit-learn", "scipy", "joblib"]},
    }
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path, compress=3)
    sparse.save_npz(matrix_path, matrix, compressed=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"\nГотово. Статей: {len(df):,}")
    print(f"Признаков в словаре: {len(vectorizer.vocabulary_):,}")
    print(f"Размер TF-IDF-матрицы: {matrix.shape}")
    print(f"Ненулевых элементов: {matrix.nnz:,}")
    print(f"Пустых векторов статей: {len(zero_ids):,}")
    print(f"Память CSR-матрицы: {memory_bytes / 1024**2:.1f} МиБ")
    print(f"Построение TF-IDF: {fit_seconds:.1f} с")
    print(f"Vectorizer: {vectorizer_path}")
    print(f"TF-IDF-матрица: {matrix_path}")
    print(f"Отчёт: {args.report}")


if __name__ == "__main__":
    main()
