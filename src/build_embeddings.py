"""Подготовка общих текстовых фрагментов и E5-векторов новостей.

Фрагменты сохраняются без префикса passage: для дальнейшего использования TF-IDF.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

MODEL_NAME = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "


def synchronize(torch, device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def prepare_fragments(tokenizer, documents, max_length):
    """Обрезает начало статьи с учётом префикса и служебных токенов.

    Границы берём из исходной строки, чтобы не восстанавливать текст из token ID.
    После обрезки повторно проверяем число токенов.
    """
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer with offset mappings is required")
    tokenizer.truncation_side = "right"
    fragments, lengths = [], []
    for start in range(0, len(documents), 512):
        batch = documents[start:start + 512]
        prefixed = [PASSAGE_PREFIX + text for text in batch]
        tokenized = tokenizer(prefixed, truncation=True, max_length=max_length,
                              return_offsets_mapping=True, padding=False,
                              return_attention_mask=False)
        clipped = []
        for text, offsets in zip(batch, tokenized["offset_mapping"]):
            end = max((int(b) for a, b in offsets if b > a), default=0)
            clipped.append(text[:max(0, end - len(PASSAGE_PREFIX))].rstrip())
        verified = tokenizer([PASSAGE_PREFIX + text for text in clipped],
                             truncation=False, padding=False,
                             return_attention_mask=False)["input_ids"]
        for text, ids in zip(clipped, verified):
            # После обрезки последняя часть слова может разбиться на большее число токенов.
            while len(ids) > max_length:
                text = text[:-1].rstrip()
                ids = tokenizer(PASSAGE_PREFIX + text, truncation=False,
                                return_attention_mask=False)["input_ids"]
            if not text:
                raise RuntimeError("Token budget produced an empty model_text")
            fragments.append(text)
            lengths.append(len(ids))
        print(f"Подготовлено фрагментов: {len(fragments):,}/{len(documents):,}",
              flush=True)
    return fragments, lengths


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=root / "data/processed/news_2018.csv")
    parser.add_argument("--texts", type=Path,
                        default=root / "data/processed/news_2018_model_text.csv")
    parser.add_argument("--models-dir", type=Path, default=root / "models")
    parser.add_argument("--report", type=Path,
                        default=root / "results/embedding_report.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"],
                        default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or not 8 <= args.max_length <= 512:
        parser.error("Batch size must be positive; max length must be 8..512")
    if not args.input.is_file():
        parser.error(f"Corpus not found: {args.input}")
    embeddings_path = args.models_dir / "news_embeddings.npy"
    ids_path = args.models_dir / "news_article_ids.json"
    paths = [p.resolve() for p in
             [args.input, args.texts, args.report, embeddings_path, ids_path]]
    if len(paths) != len(set(paths)):
        parser.error("Input and output files must have distinct paths")

    print("Загружаем библиотеки…", flush=True)
    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    mps = getattr(torch.backends, "mps", None)
    mps_available = bool(mps and mps.is_available())
    device = args.device
    if device == "auto":
        device = "mps" if mps_available else (
            "cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps" and not mps_available:
        parser.error("MPS unavailable; use --device cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")

    print("Читаем корпус…", flush=True)
    corpus = pd.read_csv(args.input, usecols=["article_id", "title", "text"],
                         dtype=str, keep_default_na=False)
    if (
        corpus.empty
        or corpus["article_id"].eq("").any()
        or corpus["article_id"].duplicated().any()
    ):
        parser.error("Corpus must have nonempty unique article_id values")
    if corpus[["title", "text"]].apply(lambda s: s.str.strip().eq("")).any().any():
        parser.error("Corpus contains empty titles or texts")
    documents = (corpus["title"] + "\n" + corpus["text"]).tolist()
    article_ids = corpus["article_id"].tolist()
    print(f"Статей: {len(corpus):,}; устройство: {device}", flush=True)

    revision = None
    benchmark_path = root / "results/encoder_benchmark.json"
    if benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark.get("model") == MODEL_NAME:
            revision = benchmark.get("model_revision")
    print(f"Загружаем {MODEL_NAME} из того же кэша, что использовал benchmark…",
          flush=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME, device=device,
                               cache_folder=str(args.models_dir / "hf_cache"),
                               revision=revision)
    model.max_seq_length = args.max_length
    model.default_prompt_name = None
    model.eval()
    tokenizer = model.tokenizer
    print("Готовим общий текст для E5 и TF-IDF…", flush=True)
    started = perf_counter()
    fragments, token_lengths = prepare_fragments(tokenizer, documents, args.max_length)
    preparation_seconds = perf_counter() - started
    shortened = sum(fragment != original.rstrip()
                    for fragment, original in zip(fragments, documents))
    prefixed = [PASSAGE_PREFIX + text for text in fragments]

    print("Прогрев encoder…", flush=True)
    options = dict(batch_size=args.batch_size, normalize_embeddings=True,
                   convert_to_numpy=True)
    model.encode(prefixed[:min(args.batch_size, len(prefixed))],
                 show_progress_bar=False, **options)
    synchronize(torch, device)
    print("Вычисляем embeddings всего корпуса…", flush=True)
    started = perf_counter()
    embeddings = model.encode(prefixed, show_progress_bar=True, **options)
    synchronize(torch, device)
    encode_seconds = perf_counter() - started
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(corpus), 384) or not np.isfinite(embeddings).all():
        raise RuntimeError("Invalid embeddings: wrong shape or nonfinite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError("Embeddings must be L2-normalized")

    config = getattr(getattr(model[0], "auto_model", None), "config", None)
    # Хеш фиксирует и порядок статей, и общий текст для двух представлений.
    fingerprint = hashlib.sha256()
    for article_id, text in zip(article_ids, fragments):
        fingerprint.update(json.dumps([article_id, text], ensure_ascii=False).encode("utf-8"))
        fingerprint.update(b"\n")
    report = {
        "model": MODEL_NAME,
        "model_revision": getattr(config, "_commit_hash", None) or revision,
        "device": device,
        "corpus_size": len(corpus),
        "input_path": str(args.input),
        "shared_text_path": str(args.texts),
        "embeddings_path": str(embeddings_path),
        "article_ids_path": str(ids_path),
        "corpus_model_text_sha256": fingerprint.hexdigest(),
        "document_format": "title + newline + text; original-string prefix fitting E5 token budget",
        "document_prefix": PASSAGE_PREFIX,
        "query_prefix": "query: ",
        "max_seq_length": args.max_length,
        "token_budget_includes": "passage prefix and model special tokens",
        "batch_size": args.batch_size,
        "shortened_articles": int(shortened),
        "max_document_tokens": int(max(token_lengths)),
        "mean_document_tokens": float(np.mean(token_lengths)),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "normalized_embeddings": True,
        "text_preparation_seconds": preparation_seconds,
        "encode_seconds": encode_seconds,
        "articles_per_second": len(corpus) / encode_seconds,
        "versions": {name: version(name) for name in
                     ["pandas", "numpy", "torch", "sentence-transformers", "transformers"]},
    }
    args.texts.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"article_id": article_ids, "model_text": fragments,
                  "e5_token_count": token_lengths}).to_csv(args.texts, index=False, encoding="utf-8")
    np.save(embeddings_path, embeddings, allow_pickle=False)
    ids_path.write_text(json.dumps(article_ids, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"\nГотово. Статей: {len(corpus):,}")
    print(f"Сокращено статей: {shortened:,}")
    print(f"Максимальная длина входа E5: {max(token_lengths)} токенов")
    print(f"Подготовка текстов: {preparation_seconds:.1f} с")
    print(f"Кодирование: {encode_seconds:.1f} с ({encode_seconds / 60:.1f} мин)")
    print(f"Размер embeddings: {embeddings.shape}; dtype: {embeddings.dtype}")
    print(f"Тексты для обеих моделей: {args.texts}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Порядок article_id: {ids_path}")
    print(f"Отчёт: {args.report}")


if __name__ == "__main__":
    main()
