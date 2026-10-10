"""Рекомендации новостей по текущему запросу и истории.

Историю передаёт вызывающий код. Новый запрос можно добавить в неё после выдачи.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


class NewsRecommender:
    def __init__(self, project_root=None, device="auto"):
        self.root = (
            Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        )
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("Unknown device")
        self.device = device
        self.models_dir = self.root / "models"
        self.article_ids = json.loads(
            (self.models_dir / "news_article_ids.json").read_text(encoding="utf-8"))
        if not self.article_ids or len(set(self.article_ids)) != len(self.article_ids):
            raise ValueError("Article identifiers must be nonempty and unique")
        self._ids_array = np.asarray(self.article_ids)
        self.embedding_report = json.loads(
            (self.root / "results/embedding_report.json").read_text(encoding="utf-8"))
        tfidf_report = json.loads(
            (self.root / "results/tfidf_report.json").read_text(encoding="utf-8"))
        fingerprint = self.embedding_report.get("corpus_model_text_sha256")
        if not fingerprint or fingerprint != tfidf_report.get("corpus_model_text_sha256"):
            raise ValueError("TF-IDF and E5 were built from different texts")
        if any(report.get("corpus_size") != len(self.article_ids)
               for report in [self.embedding_report, tfidf_report]):
            raise ValueError("Feature reports disagree about corpus size")
        corpus = pd.read_csv(self.root / "data/processed/news_2018.csv",
                             dtype=str, keep_default_na=False)
        if corpus["article_id"].duplicated().any() or set(corpus["article_id"]) != set(self.article_ids):
            raise ValueError("News corpus and feature identifiers differ")
        # Порядок строк CSV может отличаться от порядка векторов.
        self.articles = corpus.set_index("article_id").loc[self.article_ids].reset_index()
        self._tfidf = self._vectorizer = self._embeddings = self._encoder = None

    def _encode(self, texts, method):
        if method == "tfidf":
            if self._vectorizer is None:
                self._vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.joblib")
                self._tfidf = sparse.load_npz(self.models_dir / "news_tfidf.npz").tocsr()
                expected = (len(self.article_ids), len(self._vectorizer.vocabulary_))
                if self._tfidf.shape != expected:
                    raise ValueError("TF-IDF feature shape is inconsistent")
            # В плотный массив переводим только запросы, а не весь корпус.
            return self._vectorizer.transform(texts).toarray().astype(np.float32, copy=False)

        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer

            mps = getattr(torch.backends, "mps", None)
            mps_available = bool(mps and mps.is_available())
            device = self.device
            if device == "auto":
                device = "mps" if mps_available else (
                    "cuda" if torch.cuda.is_available() else "cpu")
            if device == "mps" and not mps_available:
                raise ValueError("MPS is unavailable; use CPU")
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; use CPU")
            self._embeddings = np.load(self.models_dir / "news_embeddings.npy",
                                       mmap_mode="r", allow_pickle=False)
            if self._embeddings.shape != (len(self.article_ids), 384):
                raise ValueError("E5 feature shape is inconsistent")
            if not np.isfinite(self._embeddings).all():
                raise ValueError("Saved E5 embeddings contain nonfinite values")
            # Проверяем нормы прямым суммированием, как и сходство ниже.
            row_norms = np.sqrt(np.einsum(
                "ij,ij->i", self._embeddings, self._embeddings,
                dtype=np.float64, optimize=False,
            ))
            if not np.allclose(row_norms, 1.0, atol=1e-4):
                raise ValueError("Saved E5 embeddings are not L2-normalized")
            self._encoder = SentenceTransformer(
                self.embedding_report["model"], device=device,
                cache_folder=str(self.models_dir / "hf_cache"),
                revision=self.embedding_report.get("model_revision"),
            )
            self._encoder.max_seq_length = int(self.embedding_report["max_seq_length"])
            self._encoder.default_prompt_name = None
            self._encoder.eval()
        prefix = self.embedding_report["query_prefix"]
        return np.asarray(self._encoder.encode(
            [prefix + text for text in texts], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, batch_size=16,
        ), dtype=np.float32)

    def _scores(self, vector, method):
        matrix = self._tfidf if method == "tfidf" else self._embeddings
        vector = np.asarray(vector)
        if vector.shape != (matrix.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("Invalid vector for ranking")
        if method == "tfidf":
            scores = np.asarray(matrix @ vector).reshape(-1)
        else:
            # Используем float64 и einsum: matmul давал предупреждения на Mac.
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                scores = np.einsum("ij,j->i", matrix, vector,
                                   dtype=np.float64, optimize=False)
        if not np.isfinite(scores).all():
            raise RuntimeError("Nonfinite ranking scores")
        if np.any(np.abs(scores) > 1.0001):
            raise RuntimeError("Similarity exceeds the range of normalized vectors")
        return np.clip(scores, -1.0, 1.0)

    def check_semantic_scores(self, query):
        """Проверяет E5-сходство независимым поэлементным расчётом.

        Старый matmul запускается для диагностики и не влияет на выдачу.
        """
        import warnings

        if not isinstance(query, str) or not query.strip():
            raise ValueError("A nonempty query is required for score verification")
        vector = self._encode([query.strip()], "e5")[0]
        corrected = self._scores(vector, "e5")
        vector64 = vector.astype(np.float64)
        references = []
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for start in range(0, len(self.article_ids), 2048):
                block = np.asarray(self._embeddings[start:start + 2048], dtype=np.float64)
                references.append((block * vector64).sum(axis=1))
        reference = np.concatenate(references)
        delta = float(np.max(np.abs(corrected - reference)))
        if not np.allclose(corrected, reference, rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"Corrected scores disagree with reference: {delta}")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            with np.errstate(over="warn", invalid="warn", divide="warn"):
                legacy = np.asarray(self._embeddings @ vector).reshape(-1)
        legacy_finite = bool(np.isfinite(legacy).all())
        legacy_delta = float(np.max(np.abs(legacy - reference))) if legacy_finite else None
        return {
            "checked_articles": len(reference),
            "max_difference_from_reference": delta,
            "legacy_matmul_warnings": [str(item.message) for item in captured],
            "legacy_matmul_all_finite": legacy_finite,
            "legacy_matmul_max_difference": legacy_delta,
        }

    def recommend(self, query="", history=None, method="e5", alpha=0.8, top_n=10):
        """Возвращает Top-N по запросу, профилю или их смеси.

        Нулевой TF-IDF-запрос даёт пустую выдачу. Нулевые векторы истории
        пропускаются; новости с нулевым TF-IDF-баллом в список не добавляются.
        """
        started = perf_counter()
        if method not in {"tfidf", "e5"}:
            raise ValueError("method must be tfidf or e5")
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if history is None:
            history = []
        if not isinstance(history, (list, tuple)) or any(not isinstance(text, str) for text in history):
            raise ValueError("history must be a sequence of strings")
        query = query.strip()
        recent_history = [text.strip() for text in history if text.strip()][-5:]
        result = {"method": method, "query": query, "history_considered": recent_history,
                  "history_used": [], "history_ignored_oov": [],
                  "alpha_requested": alpha, "effective_alpha": None,
                  "mode": "empty", "status": "ok", "messages": [], "items": []}

        def finish():
            result["elapsed_seconds"] = perf_counter() - started
            return result

        if not query and not recent_history:
            result["status"] = "no_input"
            result["messages"].append("Введите запрос или задайте историю интересов.")
            return finish()
        inputs = ([query] if query else []) + recent_history
        vectors = self._encode(inputs, method)
        if vectors.ndim != 2 or vectors.shape[0] != len(inputs) or not np.isfinite(vectors).all():
            raise RuntimeError("Invalid input vectors")
        query_vector = vectors[0] if query else None
        history_vectors = vectors[1:] if query else vectors
        valid_history = np.linalg.norm(history_vectors, axis=1) > 1e-8
        result["history_used"] = [text for text, keep in zip(recent_history, valid_history) if keep]
        result["history_ignored_oov"] = [text for text, keep in zip(recent_history, valid_history) if not keep]
        if result["history_ignored_oov"]:
            result["messages"].append("Некоторые запросы истории отсутствуют в словаре TF-IDF и пропущены.")
        if query_vector is not None and np.linalg.norm(query_vector) <= 1e-8:
            result["status"] = "query_oov"
            result["messages"].append("Текущий запрос отсутствует в словаре TF-IDF. Переформулируйте запрос или используйте E5.")
            return finish()

        profile = None
        if valid_history.any():
            mean = history_vectors[valid_history].mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 1e-8:
                profile = mean / norm
        query_scores = self._scores(query_vector, method) if query_vector is not None else None
        history_scores = self._scores(profile, method) if profile is not None else None
        if query_scores is not None and history_scores is not None:
            result["mode"], result["effective_alpha"] = "query_and_history", alpha
            scores = alpha * query_scores + (1 - alpha) * history_scores
        elif query_scores is not None:
            result["mode"], result["effective_alpha"] = "query_only", 1.0
            scores = query_scores
        elif history_scores is not None:
            result["mode"], result["effective_alpha"] = "history_only", 0.0
            scores = history_scores
        else:
            result["status"] = "history_oov"
            result["messages"].append("Не удалось представить интересы из истории. Используйте другие запросы или E5.")
            return finish()
        # При одинаковых баллах порядок определяется идентификатором статьи.
        eligible = np.flatnonzero(scores > 0) if method == "tfidf" else np.arange(len(scores))
        order = eligible[np.lexsort((self._ids_array[eligible], -scores[eligible]))][:top_n]
        for rank, index in enumerate(order, 1):
            article = self.articles.iloc[int(index)]
            result["items"].append({
                "rank": rank, "article_id": article["article_id"],
                "title": article["title"], "date": article["date"],
                "topic": article["topic"] or "Не указано", "url": article["url"],
                "score": float(scores[index]),
                "query_score": float(query_scores[index]) if query_scores is not None else None,
                "history_score": float(history_scores[index]) if history_scores is not None else None,
            })
        if not result["items"]:
            result["status"] = "no_positive_scores"
            result["messages"].append("Нет новостей с положительным TF-IDF-сходством.")
        return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["tfidf", "e5"], default="e5")
    parser.add_argument("--query", default="")
    parser.add_argument("--history", action="append", default=[],
                        help="Repeat --history for each previous query in chronological order")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--check-scores", action="store_true",
                        help="Verify E5 scores against an independent float64 reference")
    args = parser.parse_args()
    if args.check_scores and (args.method != "e5" or not args.query.strip()):
        parser.error("--check-scores requires --method e5 and a nonempty --query")
    try:
        print("Загружаем корпус и проверяем согласованность представлений…", flush=True)
        recommender = NewsRecommender(args.project_root, device=args.device)
        if args.check_scores:
            print("Проверяем E5-оценки по всем статьям…", flush=True)
            check = recommender.check_semantic_scores(args.query)
            print(f"Проверено статей: {check['checked_articles']:,}")
            print(f"Максимальное расхождение с независимым расчётом: {check['max_difference_from_reference']:.3e}")
            print(f"Предупреждений старого matmul: {len(check['legacy_matmul_warnings'])}")
            for message in check["legacy_matmul_warnings"]:
                print(f"  {message}")
            print(f"Старый matmul: все оценки конечны — {check['legacy_matmul_all_finite']}")
            print(f"Расхождение старого matmul с эталоном: {check['legacy_matmul_max_difference']}")
            print("Проверка нового расчёта пройдена.", flush=True)
        print(f"Ранжируем новости методом {args.method}…", flush=True)
        result = recommender.recommend(args.query, args.history, args.method,
                                       args.alpha, args.top_n)
    except (FileNotFoundError, ValueError, RuntimeError, FloatingPointError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(f"\nМетод: {result['method']}; режим: {result['mode']}")
    print(f"Текущий запрос: {result['query'] or '(нет)'}")
    print("История в профиле:", "; ".join(result["history_used"]) or "(нет)")
    print(f"Вес текущего запроса: {result['effective_alpha']}")
    print(f"Обработка вызова recommend: {result['elapsed_seconds']:.2f} с")
    for message in result["messages"]:
        print(message)
    for item in result["items"]:
        q = f"{item['query_score']:.4f}" if item["query_score"] is not None else "—"
        h = f"{item['history_score']:.4f}" if item["history_score"] is not None else "—"
        print(f"\n{item['rank']}. {item['title']}")
        print(f"   {item['date']} | {item['topic']}")
        print(f"   score={item['score']:.4f}; query={q}; history={h}")
        print(f"   {item['url']}")


if __name__ == "__main__":
    main()
