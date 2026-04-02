import math
import re
from typing import Iterable, List, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency until installed
    SentenceTransformer = None


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None



def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            )
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model



def embed_texts(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_embedding_model()
    vectors = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)



def cosine_scores(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    if doc_vectors.size == 0:
        return np.array([], dtype=np.float32)
    return np.dot(doc_vectors, query_vector)



def keyword_overlap_score(query: str, text: str) -> float:
    q_tokens = set(re.findall(r"[a-zA-Z0-9+#.-]+", query.lower()))
    t_tokens = set(re.findall(r"[a-zA-Z0-9+#.-]+", text.lower()))
    q_tokens = {t for t in q_tokens if len(t) > 2}
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens) / len(q_tokens)
    return float(overlap)



def retrieve_top_k(query: str, chunks: Sequence[dict], k: int = 5) -> List[dict]:
    if not query or not chunks:
        return []

    texts = [c.get("text", "") for c in chunks]
    try:
        query_vec = embed_texts([query])[0]
        doc_vecs = embed_texts(texts)
        semantic_scores = cosine_scores(query_vec, doc_vecs)
    except Exception:
        semantic_scores = np.array([keyword_overlap_score(query, t) for t in texts], dtype=np.float32)

    rescored = []
    for chunk, semantic_score in zip(chunks, semantic_scores):
        lexical = keyword_overlap_score(query, chunk.get("text", ""))
        final_score = 0.8 * float(semantic_score) + 0.2 * lexical
        enriched = dict(chunk)
        enriched["semantic_score"] = round(float(semantic_score), 4)
        enriched["lexical_score"] = round(float(lexical), 4)
        enriched["score"] = round(float(final_score), 4)
        rescored.append(enriched)

    rescored.sort(key=lambda item: item.get("score", 0), reverse=True)
    return rescored[:k]



def semantic_similarity_from_chunks(query: str, chunks: Sequence[dict], top_k: int = 3) -> float:
    top_chunks = retrieve_top_k(query, chunks, k=top_k)
    if not top_chunks:
        return 0.0
    avg = sum(chunk.get("score", 0) for chunk in top_chunks) / len(top_chunks)
    return round(max(0.0, min(1.0, avg)) * 100, 2)
