"""Измерение скорости предобученного E5 на фиксированной выборке новостей."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

MODEL_NAME = "intfloat/multilingual-e5-small"


def synchronize(torch, device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=root / "data/processed/news_2018.csv")
    parser.add_argument("--report", type=Path,
                        default=root / "results/encoder_benchmark.json")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"],
                        default="auto")
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error("Sample size and batch size must be positive")
    if not 8 <= args.max_length <= 512:
        parser.error("--max-length must be between 8 and 512")
    if not args.input.is_file():
        parser.error(f"Corpus not found: {args.input}")
    if args.report.resolve() == args.input.resolve():
        parser.error("Report and input must have different paths")

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
        parser.error("MPS is not available; use --device cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")

    print(f"Устройство: {device}", flush=True)
    print("Читаем подготовленный корпус…", flush=True)
    corpus = pd.read_csv(args.input, usecols=["article_id", "title", "text"],
                         dtype=str, keep_default_na=False)
    if corpus.empty:
        parser.error("Corpus is empty")
    if corpus["article_id"].eq("").any() or corpus["article_id"].duplicated().any():
        parser.error("Corpus must have nonempty unique article_id values")
    if corpus[["title", "text"]].apply(lambda s: s.str.strip().eq("")).any().any():
        parser.error("Corpus contains empty titles or texts")
    sample = corpus.sample(n=min(args.samples, len(corpus)), random_state=42)
    documents = ("passage: " + sample["title"] + "\n" + sample["text"]).tolist()
    print(f"Корпус: {len(corpus):,}; выборка: {len(sample):,}", flush=True)

    cache = root / "models/hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Загружаем {MODEL_NAME}…", flush=True)
    print("При первом запуске модель скачивается из интернета; это время измеряется отдельно.",
          flush=True)
    started = perf_counter()
    model = SentenceTransformer(MODEL_NAME, device=device, cache_folder=str(cache))
    model.max_seq_length = args.max_length
    # Inputs already include passage:, so explicitly disable any default prompt.
    model.default_prompt_name = None
    model.eval()
    synchronize(torch, device)
    load_seconds = perf_counter() - started
    print(f"Загрузка модели: {load_seconds:.1f} с", flush=True)

    encode_options = dict(batch_size=args.batch_size, normalize_embeddings=True,
                          convert_to_numpy=True)
    print("Прогрев encoder…", flush=True)
    started = perf_counter()
    model.encode(documents[:min(args.batch_size, len(documents))],
                 show_progress_bar=False, **encode_options)
    synchronize(torch, device)
    warmup_seconds = perf_counter() - started

    print(f"Кодируем {len(documents):,} новостей, максимум {args.max_length} токенов…",
          flush=True)
    synchronize(torch, device)
    started = perf_counter()
    embeddings = model.encode(documents, show_progress_bar=True, **encode_options)
    synchronize(torch, device)
    encode_seconds = perf_counter() - started
    if encode_seconds <= 0:
        raise RuntimeError("Invalid benchmark duration")
    if embeddings.shape != (len(documents), 384):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embeddings contain nonfinite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError("Embeddings are not normalized")

    throughput = len(documents) / encode_seconds
    projected_seconds = len(corpus) / throughput
    config = getattr(getattr(model[0], "auto_model", None), "config", None)
    report = {
        "model": MODEL_NAME,
        "model_revision": getattr(config, "_commit_hash", None),
        "device": device,
        "architecture": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "versions": {name: version(name) for name in
                     ["pandas", "numpy", "torch", "sentence-transformers", "transformers"]},
        "corpus_path": str(args.input),
        "corpus_size": len(corpus),
        "sample_size": len(sample),
        "sample_seed": 42,
        "sample_id_sha256": hashlib.sha256(
            "\n".join(sample["article_id"]).encode("utf-8")).hexdigest(),
        "text_format": "passage: <title>\\n<text>; right truncation by encoder tokenizer",
        "max_seq_length": args.max_length,
        "batch_size": args.batch_size,
        "normalized_embeddings": True,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "encode_seconds": encode_seconds,
        "articles_per_second": throughput,
        "projected_full_encode_seconds": projected_seconds,
        "estimate_note": "Linear extrapolation; excludes loading, corpus I/O and saving. Sustained performance may differ.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"\nЗагрузка модели: {load_seconds:.1f} с")
    print(f"Прогрев: {warmup_seconds:.1f} с")
    print(f"Кодирование {len(documents):,} статей: {encode_seconds:.1f} с")
    print(f"Скорость: {throughput:.2f} статьи/с")
    print(f"Размер embeddings: {embeddings.shape}")
    print(f"Оценка кодирования {len(corpus):,} статей: {projected_seconds / 60:.1f} мин")
    print("Это ориентир: длительный запуск может быть медленнее из-за нагрева Mac.")
    print(f"Отчёт сохранён: {args.report}")


if __name__ == "__main__":
    main()
