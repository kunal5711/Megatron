"""
Vector store for semantic similarity matching using FAISS.

Converts ACTIVE memory records into embeddings for semantic search.
Only ACTIVE records are indexed; DISABLED records are never exposed.
"""
import numpy as np
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer

from backend.models import MemoryRecord
from backend.config import EMBEDDING_MODEL, FAISS_TOP_K

# Global caches
_EMBEDDINGS_MODEL: Optional[SentenceTransformer] = None
_FAISS_INDEX = None
_INDEX_TO_MEMORY: Dict[int, MemoryRecord] = {}
_MEMORY_DOCUMENTS: List[str] = []

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


def initialize_embeddings() -> SentenceTransformer:
    """Load sentence-transformers model."""
    global _EMBEDDINGS_MODEL
    if _EMBEDDINGS_MODEL is None:
        _EMBEDDINGS_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDINGS_MODEL


def build_semantic_documents(memory_records: List[MemoryRecord]) -> List[str]:
    """Convert MemoryRecord objects into text documents for embedding."""
    documents = []
    for record in memory_records:
        doc = (
            f"Source column: {record.source_column}\n"
            f"Target field: {record.target_field}\n"
            f"Context: {record.context.category}, {record.context.tenant_name}"
        )
        documents.append(doc)
    return documents


def build_faiss_index(memory_records: List[MemoryRecord]) -> Optional[object]:
    """
    Generate embeddings and create FAISS index from ACTIVE memory records.
    """
    global _FAISS_INDEX, _INDEX_TO_MEMORY, _MEMORY_DOCUMENTS

    if not FAISS_AVAILABLE:
        print("Warning: FAISS not available. Vector search will be disabled.")
        return None

    if not memory_records:
        print("No memory records to index.")
        _FAISS_INDEX = None
        _INDEX_TO_MEMORY = {}
        _MEMORY_DOCUMENTS = []
        return None

    model = initialize_embeddings()
    documents = build_semantic_documents(memory_records)
    _MEMORY_DOCUMENTS = documents

    print(f"Generating embeddings for {len(documents)} memory records...")
    embeddings = model.encode(documents, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    _INDEX_TO_MEMORY = {i: record for i, record in enumerate(memory_records)}
    _FAISS_INDEX = index

    print(f"FAISS index built with {index.ntotal} vectors (dimension={dimension})")
    return index


def query_memory_vector(query_text: str, top_k: int = None) -> List[Dict]:
    """Search FAISS index for top-K similar ACTIVE memory records."""
    if not FAISS_AVAILABLE or _FAISS_INDEX is None:
        return []

    if top_k is None:
        top_k = FAISS_TOP_K

    top_k = min(top_k, _FAISS_INDEX.ntotal)
    if top_k == 0:
        return []

    model = initialize_embeddings()
    query_embedding = model.encode([query_text], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = _FAISS_INDEX.search(query_embedding, top_k)

    results = []
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        similarity_score = 1.0 / (1.0 + distance)
        results.append({
            'record': _INDEX_TO_MEMORY[idx],
            'score': float(similarity_score),
            'distance': float(distance),
            'rank': i + 1,
        })
    return results


def build_query_from_profile(column_name: str, inferred_dtype: str = "", tenant_name: str = "") -> str:
    """Build a query text from column profile for vector search."""
    return (
        f"Source column: {column_name}\n"
        f"Data type: {inferred_dtype}\n"
        f"Tenant name: {tenant_name}"
    )


def rebuild_index(memory_records: List[MemoryRecord]):
    """Rebuild FAISS index after memory updates."""
    print("Rebuilding FAISS index with updated memory...")
    build_faiss_index(memory_records)


def get_index_stats() -> Dict:
    """Get statistics about the current FAISS index."""
    if not FAISS_AVAILABLE or _FAISS_INDEX is None:
        return {'available': False, 'total_vectors': 0, 'dimension': 0}
    return {
        'available': True,
        'total_vectors': _FAISS_INDEX.ntotal,
        'dimension': _FAISS_INDEX.d,
        'model': EMBEDDING_MODEL,
    }
