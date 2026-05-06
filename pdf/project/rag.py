from pathlib import Path
from collections import defaultdict
import json
import pickle
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import INDEX_DIR, EMBED_MODEL, RAG_EMBEDDINGS, CHUNK_CHARS, CHUNK_OVERLAP, TOP_K, POOL_K
from utils import normalize_text

try:
    import faiss
except Exception:
    faiss = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

GLOBAL_EMBEDDER = None

class TextChunker:
    def __init__(self, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
        self.size = size
        self.overlap = overlap
    def split(self, text):
        text = normalize_text(text)
        if not text:
            return []
        if len(text) <= self.size:
            return [text]
        chunks = []
        i = 0
        while i < len(text):
            j = min(len(text), i + self.size)
            cut = max(text.rfind("\n", i, j), text.rfind(". ", i, j), text.rfind("; ", i, j))
            if cut > i + self.size * 0.55:
                j = cut + 1
            chunks.append(text[i:j].strip())
            if j >= len(text):
                break
            i = max(0, j - self.overlap)
        return [c for c in chunks if len(c) > 50]

class VectorStore:
    def __init__(self, case_id):
        self.case_id = case_id
        self.path = INDEX_DIR / case_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.chunks = []
        self.embedder = None
        self.faiss_index = None
        self.embeddings = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.last_retrieval = {}
    def _load_embedder(self):
        global GLOBAL_EMBEDDER
        if not RAG_EMBEDDINGS or SentenceTransformer is None or faiss is None:
            return None
        if GLOBAL_EMBEDDER is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            GLOBAL_EMBEDDER = SentenceTransformer(EMBED_MODEL, device=device)
        self.embedder = GLOBAL_EMBEDDER
        return self.embedder
    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        try:
            embedder = self._load_embedder()
            if embedder is not None and texts:
                self.embeddings = embedder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False).astype("float32")
                self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.faiss_index.add(self.embeddings)
        except Exception:
            self.faiss_index = None
            self.embeddings = None
        self.tfidf = TfidfVectorizer(max_features=80000, ngram_range=(1, 2), stop_words="english", min_df=1)
        self.tfidf_matrix = self.tfidf.fit_transform(texts) if texts else None
        self.save()
    def save(self):
        (self.path / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(self.path / "tfidf.pkl", "wb") as f:
            pickle.dump({"tfidf": self.tfidf, "matrix": self.tfidf_matrix}, f)
        if self.faiss_index is not None and faiss is not None:
            faiss.write_index(self.faiss_index, str(self.path / "faiss.index"))
            np.save(self.path / "embeddings.npy", self.embeddings)
    def rrf(self, ranked_lists, k=60):
        scores = defaultdict(float)
        for items in ranked_lists:
            for rank, idx in enumerate(items):
                scores[int(idx)] += 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True), scores
    def retrieve(self, query, top_k=TOP_K, pool_k=POOL_K):
        started = time.perf_counter()
        ranked = []
        engines = []
        if self.faiss_index is not None and self.embedder is not None:
            q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
            _, ids = self.faiss_index.search(q, min(pool_k, len(self.chunks)))
            semantic = [int(i) for i in ids[0] if i >= 0]
            ranked.append(semantic)
            engines.append({"name": "faiss", "candidates": len(semantic)})
        if self.tfidf_matrix is not None:
            qx = self.tfidf.transform([query])
            sim = cosine_similarity(qx, self.tfidf_matrix).ravel()
            lexical = [int(i) for i in np.argsort(-sim)[:min(pool_k, len(self.chunks))]]
            ranked.append(lexical)
            engines.append({"name": "tfidf", "candidates": len(lexical)})
        fused, scores = self.rrf(ranked)
        results = []
        for idx in fused[:top_k]:
            row = dict(self.chunks[idx])
            row["rank"] = len(results) + 1
            row["score"] = round(float(scores[idx]), 6)
            results.append(row)
        self.last_retrieval = {
            "query_chars": len(query or ""),
            "chunks_total": len(self.chunks),
            "requested_top_k": top_k,
            "requested_pool_k": pool_k,
            "engines": engines,
            "returned": len(results),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return results

def chunks_from_records(records, case_id):
    chunker = TextChunker()
    chunks = []
    for record in records:
        for i, text in enumerate(chunker.split(record.get("text", ""))):
            chunks.append({"chunk_id": f"{case_id}_C{len(chunks)+1:05d}", "text": text, "source_file": record.get("source_file"), "page": record.get("page"), "element_id": record.get("element_id"), "kind": record.get("kind", "document"), "region_type": record.get("region_type"), "bbox": record.get("bbox"), "text_source": record.get("text_source"), "layout_confidence": record.get("layout_confidence"), "reading_order": record.get("reading_order"), "chunk_index": i})
    return chunks

