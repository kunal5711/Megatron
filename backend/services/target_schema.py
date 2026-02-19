"""
Target schema loading and management.

Loads the target schema definition from JSON file. The target schema defines
all available target fields that source columns can be mapped to.

Also handles:
- Embedding generation for semantic similarity matching
- Required field enforcement
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from sentence_transformers import SentenceTransformer

from backend.models import TargetField, MappingDecision, ColumnProfile
from backend.config import BASE_DIR, EMBEDDING_MODEL, FAISS_TOP_K

# Global variables to cache target embeddings
_TARGET_EMBEDDINGS: Optional[np.ndarray] = None
_TARGET_FIELDS: List[TargetField] = []
_EMBEDDINGS_MODEL: Optional[SentenceTransformer] = None


def load_target_schema(path: Path = None) -> List[TargetField]:
    """
    Load target schema from JSON file.
    
    The target schema defines the destination structure with all available
    target fields. Each field includes:
    - field_name: Name of the target field
    - datatype: Expected data type
    - category: Field category/group
    - description: Human-readable description
    - required: Whether this field is required
    
    Args:
        path: Optional custom path to schema JSON file.
              If not provided, uses default: target_schema.json in project root
    
    Returns:
        List of TargetField objects representing all available target fields
    """
    # Use provided path or default to schema in project root
    schema_path = path or (BASE_DIR / "target_schema.json")
    
    # Read and parse JSON file
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert JSON objects to TargetField Pydantic models
    return [TargetField(**item) for item in data]


def build_target_documents(targets: List[TargetField]) -> List[str]:
    """
    Convert target fields into text documents for embedding.
    
    Creates structured text representations similar to memory documents
    to enable semantic similarity matching.
    
    Args:
        targets: List of TargetField objects
    
    Returns:
        List of text documents suitable for embedding
    """
    documents = []
    
    for target in targets:
        # Build structured document
        doc = f"""Field: {target.field_name}
Category: {target.category or 'N/A'}
Description: {target.description or 'N/A'}
Required: {target.required}"""
        
        documents.append(doc)
    
    return documents


def build_target_embeddings(targets: List[TargetField]) -> np.ndarray:
    """
    Create embeddings for target schema fields once at startup.
    
    Generates embeddings for all target fields and caches them in memory
    for fast similarity search during mapping.
    
    Args:
        targets: List of TargetField objects from target schema
    
    Returns:
        numpy array of embeddings (shape: [num_targets, embedding_dim])
    """
    global _TARGET_EMBEDDINGS, _TARGET_FIELDS, _EMBEDDINGS_MODEL
    
    _TARGET_FIELDS = targets
    # Initialize embeddings model
    if _EMBEDDINGS_MODEL is None:
        _EMBEDDINGS_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    
    # Build documents
    documents = build_target_documents(targets)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(documents)} target fields...")
    embeddings = _EMBEDDINGS_MODEL.encode(documents, show_progress_bar=False)
    _TARGET_EMBEDDINGS = np.array(embeddings).astype('float32')
    
    print(f"Target embeddings generated (shape: {_TARGET_EMBEDDINGS.shape})")
    return _TARGET_EMBEDDINGS


def query_target_similarity(source_column: str, source_samples: List[Any] = None, 
                           inferred_dtype: str = "", top_k: int = None) -> List[Dict]:
    """
    Find top-K target fields by embedding similarity.
    
    Embeds the source column profile and finds the most similar target
    fields using cosine similarity on pre-computed embeddings.
    
    Args:
        source_column: Source column name
        source_samples: Sample values from the source column
        inferred_dtype: Inferred data type of the source column
        top_k: Number of results to return (default: FAISS_TOP_K from config)
    
    Returns:
        List of dicts with 'target', 'score', and 'rank' keys
    """
    global _TARGET_EMBEDDINGS, _TARGET_FIELDS, _EMBEDDINGS_MODEL
    
    if _TARGET_EMBEDDINGS is None or not _TARGET_FIELDS:
        print("Warning: Target embeddings not initialized")
        return []
    
    if top_k is None:
        top_k = FAISS_TOP_K
    
    # Ensure we don't request more results than available
    top_k = min(top_k, len(_TARGET_FIELDS))
    
    # Initialize embeddings model
    if _EMBEDDINGS_MODEL is None:
        _EMBEDDINGS_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    
    query_doc = f"""Source column: {source_column}
Data type: {inferred_dtype}"""
    
    # Embed query
    query_embedding = _EMBEDDINGS_MODEL.encode([query_doc], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Calculate cosine similarities
    # Normalize vectors for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    targets_norm = _TARGET_EMBEDDINGS / np.linalg.norm(_TARGET_EMBEDDINGS, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = np.dot(targets_norm, query_norm.T).flatten()
    
    # Get top-K indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Build results
    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            'target': _TARGET_FIELDS[idx],
            'score': float(similarities[idx]),
            'rank': i + 1
        })
    
    return results


def get_required_fields(targets: List[TargetField]) -> List[TargetField]:
    """
    Filter targets where required=true.
    
    Returns all target fields that are marked as required in the schema.
    These fields must appear in the final mapping output.
    
    Args:
        targets: List of all TargetField objects
    
    Returns:
        List of required TargetField objects
    """
    return [t for t in targets if t.required is True or str(t.required).lower() == 'true']


def enforce_required_fields(mappings: List[MappingDecision], 
                           targets: List[TargetField]) -> List[MappingDecision]:
    """
    Ensure all required fields appear in final output.
    
    Checks that all required target fields are present in the mappings.
    If a required field is missing, adds a mapping decision with:
    - selected_target: the required field
    - decision: "needs_review"
    - confidence: 0.0
    - explanation: "required_without_source"
    
    Args:
        mappings: List of MappingDecision objects from the pipeline
        targets: List of all TargetField objects
    
    Returns:
        Updated list of MappingDecision objects with required fields enforced
    """
    # Get all required fields
    required_fields = get_required_fields(targets)
    
    # Get set of already mapped target fields
    mapped_targets = set()
    for mapping in mappings:
        if mapping.selected_target:
            mapped_targets.add(mapping.selected_target.field_name)
    
    # Find missing required fields
    missing_required = [f for f in required_fields if f.field_name not in mapped_targets]
    
    # Add placeholder mappings for missing required fields
    for field in missing_required:
        placeholder = MappingDecision(
            source_column=f"<unmapped_{field.field_name}>",
            selected_target=field,
            decision="needs_review",
            confidence=0.0,
            explanation="Required field without source mapping. Please assign a source column or provide a default value.",
            candidates=[]
        )
        mappings.append(placeholder)
    
    return mappings


def get_target_by_name(field_name: str, targets: List[TargetField]) -> Optional[TargetField]:
    """
    Find a target field by name (normalized comparison: case-insensitive, spaces as underscores).

    Args:
        field_name: Name of the target field to find
        targets: List of all TargetField objects

    Returns:
        TargetField object if found, None otherwise
    """
    if not field_name:
        return None
    normalized = _normalize_for_match(field_name)
    for target in targets:
        if _normalize_for_match(target.field_name or "") == normalized:
            return target
    return None


def _normalize_for_match(name: str) -> str:
    """Normalize name for exact matching: lower case, spaces to underscores."""
    if not name:
        return ""
    return name.strip().lower().replace(" ", "_")


def normalize_target_schema_file(path: Path = None) -> int:
    """
    Normalize field_name in target_schema.json (lowercase, spaces to underscores).
    Overwrites the file in place.

    Args:
        path: Optional path to schema JSON. Default: BASE_DIR / "target_schema.json"

    Returns:
        Number of fields updated
    """
    schema_path = path or (BASE_DIR / "target_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for item in data:
        orig = item.get("field_name", "")
        if orig:
            norm = _normalize_for_match(orig)
            if norm != orig:
                item["field_name"] = norm
                count += 1

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return count


def exact_match_target_schema(
    source_column: str, targets: List[TargetField]
) -> Optional[Tuple[TargetField, float]]:
    """
    Check if the source column name exactly matches any target field name
    (normalized: case-insensitive, spaces as underscores).
    If found, return (TargetField, confidence) so the pipeline can make an immediate decision.

    Args:
        source_column: Source column name from the CSV
        targets: List of all TargetField objects

    Returns:
        (TargetField, confidence) if exact match found, else None
    """
    if not source_column or not targets:
        return None
    normalized_source = _normalize_for_match(source_column)
    if not normalized_source:
        return None
    for target in targets:
        normalized_target = _normalize_for_match(target.field_name or "")
        if normalized_target and normalized_source == normalized_target:
            return (target, 0.95)
    return None