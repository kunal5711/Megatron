"""
Heuristic scoring functions for matching source columns to target fields.

This module implements the core matching logic that scores how well a source column
matches each potential target field based on multiple criteria:
- Name similarity (fuzzy string matching)
- Data type compatibility
- Length compatibility
- Category alignment

These scores are weighted and combined to produce a final match score.
"""
from typing import List, Dict
from rapidfuzz import fuzz

from backend import config
from backend.models import ColumnProfile, CandidateScore, TargetField
from backend.services.target_schema import _normalize_for_match


def name_similarity(source: str, target: str) -> float:
    """
    Calculate similarity score between source and target field names.

    Normalizes both names (lowercase, spaces to underscores) before comparing
    so that "Customer Name" and "customer_name" are treated as equal.
    Uses token-based fuzzy matching that is order-independent.

    Args:
        source: Source column name (e.g., "customer_name")
        target: Target field name (e.g., "customerName")

    Returns:
        Similarity score between 0.0 (no match) and 1.0 (exact match)
    """
    return fuzz.token_sort_ratio(
        _normalize_for_match(source),
        _normalize_for_match(target),
    ) / 100.0


def datatype_compat(source_dtype: str, target_dtype: str) -> float:
    """
    Calculate compatibility score between source and target data types.
    
    Scores are based on how well the types align:
    - Exact match: 1.0
    - Compatible types (e.g., number ↔ integer): 0.8-0.9
    - String/text (generic): 0.7
    - Incompatible: 0.2
    - Missing target type: 0.5 (neutral)
    
    Args:
        source_dtype: Inferred data type from source column (e.g., "number", "string")
        target_dtype: Expected data type of target field (e.g., "integer", "text")
    
    Returns:
        Compatibility score between 0.0 and 1.0
    """
    # If target type is not specified, return neutral score
    if not target_dtype:
        return 0.5
    
    # Exact type match gets perfect score
    if source_dtype == target_dtype:
        return 1.0
    
    # Numeric types are compatible (number, integer, float, decimal)
    # Bidirectional check handles both source→target and target→source
    if source_dtype in ["number", "integer", "float", "decimal"] and target_dtype in ["number", "integer", "float", "decimal"]:
        return 0.8
    
    # String/text types are compatible but less specific (lower score)
    # Lower score reflects that strings are generic and less meaningful
    if source_dtype in ["string", "text"] and target_dtype in ["string", "text"]:
        return 0.7
    
    # Date/datetime types are compatible
    if source_dtype in ["date", "datetime"] and target_dtype in ["date", "datetime"]:
        return 0.8
    
    # Boolean types are compatible
    if source_dtype in ["boolean", "bool"] and target_dtype in ["boolean", "bool"]:
        return 0.9
    
    # Incompatible types get low score
    return 0.2


def length_compat(avg_len, max_len, target_dtype: str) -> float:
    """
    Calculate compatibility score based on source column length characteristics.
    
    This helps prevent data truncation issues by matching column lengths to 
    appropriate target field types. For string fields, shorter lengths score higher.
    
    Args:
        avg_len: Average length of values in source column
        max_len: Maximum length of values in source column
        target_dtype: Expected data type of target field
    
    Returns:
        Length compatibility score between 0.0 and 1.0
    """
    # If length data is missing, return neutral score
    if avg_len is None or max_len is None:
        return 0.5
    
    # For string/text fields, score based on maximum length
    # Shorter strings are preferred (better fit for VARCHAR fields)
    if target_dtype in ["string", "text"]:
        if max_len <= 50:
            return 1.0  # Perfect for short VARCHAR fields
        if max_len <= 255:
            return 0.8  # Good for standard VARCHAR(255)
        return 0.6  # May need TEXT/CLOB for longer strings
    
    # For numeric fields, length is less critical (consistent score)
    if target_dtype in ["integer", "float", "number", "decimal"]:
        return 0.7
    
    # For other types (date, boolean), return neutral score
    return 0.5


def category_alignment(source_name: str, target_category: str) -> float:
    """
    Calculate alignment score based on category matching.
    
    Checks if the target category appears in the source column name.
    This helps match columns like "customer_email" to category "customer".
    
    Args:
        source_name: Source column name (e.g., "customer_email")
        target_category: Target field category (e.g., "customer")
    
    Returns:
        Alignment score: 1.0 if category found in name, 0.5 otherwise
    """
    # If no category specified, return neutral score
    if not target_category:
        return 0.5
    
    # Check if category appears in source column name (normalized)
    tokens = _normalize_for_match(source_name)
    if _normalize_for_match(target_category) in tokens:
        return 1.0
    
    # No match found
    return 0.5


def score_candidate(source_profile: ColumnProfile, target: TargetField) -> CandidateScore:
    """
    Calculate overall match score for a source column against a target field.
    
    Combines multiple heuristic scores with weighted averaging:
    - Name similarity (50% weight)
    - Data type compatibility (20% weight)
    - Length compatibility (15% weight)
    - Category alignment (15% weight)
    
    Args:
        source_profile: Profile of the source column with inferred characteristics
        target: Target field definition to match against
    
    Returns:
        CandidateScore object with total score and breakdown of individual scores
    """
    # Calculate individual component scores
    name_score = name_similarity(source_profile.name, target.field_name)
    dtype_score = datatype_compat(source_profile.inferred_dtype, target.datatype or "")
    length_score = length_compat(source_profile.avg_length, source_profile.max_length, target.datatype or "")
    category_score = category_alignment(source_profile.name, target.category or "")
    
    # Calculate weighted total score
    # Weights are defined in config.py and sum to 1.0
    total = (
        name_score * config.NAME_WEIGHT
        + dtype_score * config.DATATYPE_WEIGHT
        + length_score * config.LENGTH_WEIGHT
        + category_score * config.CATEGORY_WEIGHT
    )
    
    # Store breakdown for transparency and debugging
    breakdown: Dict[str, float] = {
        "name": name_score,
        "datatype": dtype_score,
        "length": length_score,
        "category": category_score,
    }
    
    return CandidateScore(
        target=target,
        total_score=float(total),
        breakdown=breakdown,
        source_column=source_profile.name,
    )


def rank_candidates(source_profile: ColumnProfile, targets: List[TargetField]) -> List[CandidateScore]:
    """
    Score and rank all target fields against a source column.
    
    Scores each target field and returns them sorted by total score (highest first).
    This is used to identify the best matching candidates for a source column.
    
    Args:
        source_profile: Profile of the source column to match
        targets: List of all available target fields to match against
    
    Returns:
        List of CandidateScore objects sorted by total_score (descending)
    """
    # Score all candidates
    candidates = [score_candidate(source_profile, t) for t in targets]
    
    # Sort by total score (highest first)
    return sorted(candidates, key=lambda c: c.total_score, reverse=True)