"""
Mapping decision logic that combines heuristics, memory, vector search, and LLM.

This module orchestrates the 4-step decision-making pipeline:
1. Heuristic Pass: Exact/fuzzy matching against memory table
2. Memory Vector Search: FAISS-based semantic search of historical mappings
3. Target Schema Similarity: Embedding similarity with target fields
4. LLM Reasoning: OpenRouter API for disambiguation and final selection
"""
from typing import List, Dict, Optional, Any

from backend import config
from backend.models import MappingDecision, CandidateScore, ColumnProfile, TargetField
from backend.services import heuristics, memory, llm_service, vector_store, target_schema
from backend.services.target_schema import _normalize_for_match


def merge_candidates(memory_candidates: List[Dict], schema_candidates: List[Dict]) -> List[Dict]:
    """
    Merge and deduplicate candidates from memory vector search and schema similarity.
    Uses normalized target_field for comparison (case-insensitive, spaces as underscores).

    Args:
        memory_candidates: Candidates from memory vector search
        schema_candidates: Candidates from target schema similarity

    Returns:
        Merged and deduplicated list of candidates, sorted by score
    """
    merged = {}

    for candidate in memory_candidates:
        record = candidate.get('record')
        if isinstance(record, memory.MemoryRecord):
            target_field = record.target_field
        elif isinstance(record, dict):
            target_field = record.get('target_field')
        else:
            continue

        score = candidate.get('score', 0.0)
        key = _normalize_for_match(target_field or "")

        if key and (key not in merged or score > merged[key]['score']):
            merged[key] = {
                'target_field': target_field,
                'score': score,
                'source': 'memory'
            }

    for candidate in schema_candidates:
        target = candidate.get('target')
        if hasattr(target, 'field_name'):
            target_field = target.field_name
            target_obj = target
        elif isinstance(target, dict):
            target_field = target.get('field_name')
            target_obj = target
        else:
            continue

        score = candidate.get('score', 0.0)
        key = _normalize_for_match(target_field or "")

        if key and (key not in merged or score > merged[key]['score']):
            merged[key] = {
                'target_field': target_field,
                'target': target_obj,
                'score': score,
                'source': 'schema'
            }
        elif key and key in merged and 'target' not in merged[key]:
            merged[key]['target'] = target_obj

    result = list(merged.values())
    result.sort(key=lambda x: x['score'], reverse=True)
    return result


def combine_confidence(llm_result: Dict, merged_candidates: List[Dict], 
                      heuristic_match: Optional[Dict] = None) -> float:
    """
    Combine confidence scores from LLM and heuristics.
    
    Logic:
    - If heuristic and LLM agree: add scores (capped at 1.0)
    - If disagree: use LLM score if > threshold, else heuristic score
    
    Args:
        llm_result: Dictionary with LLM result including 'confidence' and 'selected_target'
        merged_candidates: List of merged candidates with scores
        heuristic_match: Optional heuristic match result with confidence
    
    Returns:
        Combined confidence score (0.0 to 1.0)
    """
    llm_confidence = llm_result.get('confidence', 0.0)
    llm_target = llm_result.get('selected_target')
    
    # If no heuristic match, use LLM confidence
    if not heuristic_match:
        return llm_confidence
    
    heuristic_target = heuristic_match.get('target_field')
    heuristic_confidence = heuristic_match.get('confidence', 0.0)

    # Check if they agree (normalized comparison)
    if _normalize_for_match(llm_target or "") == _normalize_for_match(heuristic_target or ""):
        # Agreement: add scores (capped at 1.0)
        combined = min(llm_confidence + heuristic_confidence * 0.5, 1.0)
        return combined
    else:
        # Disagreement: use higher confidence if above threshold
        threshold = config.HEURISTIC_THRESHOLD
        
        if llm_confidence > threshold and llm_confidence > heuristic_confidence:
            return llm_confidence
        elif heuristic_confidence > threshold:
            return heuristic_confidence
        else:
            # Both below threshold, average them
            return (llm_confidence + heuristic_confidence) / 2.0


def create_decision(result: Dict, reason: str, confidence: float, 
                   source_column: str, candidates: List[Dict] = None) -> MappingDecision:
    """
    Create a MappingDecision from a result dictionary.
    
    Args:
        result: Dictionary with mapping result (target_field, etc.)
        reason: Explanation of how the decision was made
        confidence: Confidence score (0.0 to 1.0)
        source_column: Source column name
        candidates: Optional list of all candidates considered
    
    Returns:
        MappingDecision object
    """
    # Ensure confidence is capped at 1.0
    confidence = min(max(confidence, 0.0), 1.0)
    
    # Determine decision status based on confidence
    auto_threshold, review_threshold = config.thresholds()
    
    if confidence >= auto_threshold:
        decision_status = "auto_approved"
    elif confidence >= review_threshold:
        decision_status = "needs_review"
    else:
        decision_status = "reject"
    
    # Extract target field
    target_field = None
    if 'target_field' in result:
        target_name = result['target_field']
        # Find TargetField object
        if 'target' in result:
            target_field = result['target']
        else:
            # Try to find in candidates
            for c in (candidates or []):
                if hasattr(c.get('target'), 'field_name'):
                    if c['target'].field_name == target_name:
                        target_field = c['target']
                        break
    
    # Convert candidates to CandidateScore objects
    candidate_scores = []
    if candidates:
        for c in candidates[:5]:  # Top 5 only
            target = c.get('target')
            if target and hasattr(target, 'field_name'):
                candidate_scores.append(CandidateScore(
                    target=target,
                    total_score=c.get('score', 0.0),
                    breakdown={'merged': c.get('score', 0.0)},
                    source_column=source_column
                ))

    return MappingDecision(
        source_column=source_column,
        selected_target=target_field,
        decision=decision_status,
        confidence=confidence,
        explanation=reason,
        candidates=candidate_scores
    )


def decide_with_pipeline(profile: ColumnProfile, targets: List[TargetField], 
                        tenant_name: str = "") -> MappingDecision:
    """
    Make a mapping decision using the 4-step pipeline.
    
    Pipeline:
    1. Heuristic Pass: Exact/fuzzy matching in memory table
    2. Memory Vector Search: FAISS semantic search
    3. Target Schema Similarity: Embedding similarity with target fields
    4. LLM Reasoning: OpenRouter API for final selection
    
    Args:
        profile: ColumnProfile with source column details
        targets: List of all TargetField objects
        tenant_name: Optional tenant identifier for prioritization
    
    Returns:
        MappingDecision with selected target and confidence
    """
    # STEP 0: Exact match against target schema (same idea as memory exact match)
    target_exact = target_schema.exact_match_target_schema(profile.name, targets)
    if target_exact:
        target_field, confidence = target_exact
        return create_decision(
            {"target_field": target_field.field_name, "target": target_field},
            "Exact match with target schema",
            confidence,
            profile.name,
            [{"target_field": target_field.field_name, "target": target_field, "score": confidence}],
        )

    # STEP 1: Heuristic Pass — if we have an exact match in memory, trust it and return (saved corrections)
    exact = memory.exact_match(profile.name, tenant_name)
    if exact and exact['is_same_tenant']:
        target_name = exact["target_field"]
        target_field_obj = target_schema.get_target_by_name(target_name, targets)
        if target_field_obj:
            return create_decision(
                {"target_field": target_name, "target": target_field_obj},
                exact.get("reason", "Exact match from memory"),
                0.95,
                profile.name,
                [{"target_field": target_name, "target": target_field_obj, "score": 0.95}],
            )
    fuzzy = memory.fuzzy_match(profile.name, tenant_name)

    # STEP 2: Memory Vector Search
    query_text = vector_store.build_query_from_profile(
        profile.name,
        profile.inferred_dtype,
        tenant_name,
    )
    memory_candidates = vector_store.query_memory_vector(query_text, top_k=10)
    
    # STEP 3: Target Schema Similarity
    schema_candidates = target_schema.query_target_similarity(
        profile.name,
        profile.inferred_dtype,
        top_k=10
    )
    
    # Merge candidates from both sources
    merged = merge_candidates(memory_candidates, schema_candidates)

    # Inject heuristic matches into merged candidates.
    # Priority order:
    #   1. exact_match  — highest priority, boosts or dominates the score
    #   2. memory_candidates / schema_candidates — primary signal
    #   3. fuzzy_match  — lowest priority, small supplementary weight
    EXACT_WEIGHT  = 0.8   # exact match is the strongest signal
    VECTOR_WEIGHT = 0.7   # memory / schema candidates are the primary signal
    FUZZY_WEIGHT  = 0.15  # fuzzy match is lowest priority

    exact_target = exact.get('target_field') if exact else None

    for match, weight, label in [
        (exact, EXACT_WEIGHT, 'heuristic_exact'),
        (fuzzy, FUZZY_WEIGHT, 'heuristic_fuzzy'),
    ]:
        if not match:
            continue
        h_target = match.get('target_field')
        h_conf = match.get('confidence', 0.0)
        # If fuzzy points to same target as exact, don't overwrite exact's boost
        if match is fuzzy and _normalize_for_match(h_target or "") == _normalize_for_match(exact_target or ""):
            continue
        h_norm = _normalize_for_match(h_target or "")
        existing = next((c for c in merged if _normalize_for_match(c.get('target_field') or "") == h_norm), None)
        if existing:
            if match is exact:
                # Exact match: boost so it can dominate
                existing['score'] = max(existing['score'], (weight * h_conf) + ((1 - weight) * existing['score']))
            else:
                # Fuzzy match: small additive nudge, never overrides vector/schema
                existing['score'] = (VECTOR_WEIGHT * existing['score']) + (FUZZY_WEIGHT * h_conf)
        else:
            # Add as new candidate with weighted score
            target_field_obj = target_schema.get_target_by_name(h_target, targets)
            merged.append({
                'target_field': h_target,
                'target': target_field_obj,
                'score': weight * h_conf,
                'source': match.get('reason', label)
            })

    merged.sort(key=lambda x: x['score'], reverse=True)

    # If no candidates found at all, return reject decision
    if not merged:
        return MappingDecision(
            source_column=profile.name,
            selected_target=None,
            decision="reject",
            confidence=0.0,
            explanation="No suitable candidates found in memory or schema similarity search.",
            candidates=[]
        )
    
    # STEP 4: LLM Reasoning (only if needed)
    # Use LLM when best score <= 0.95 (no LLM if we already have very high confidence)
    # or when top two are close (score diff < 0.1)
    best_score = merged[0]['score'] if merged else 0.0
    score_diff = (merged[0]['score'] - merged[1]['score']) if len(merged) > 1 else 1.0
    use_llm = best_score <= config.HEURISTIC_THRESHOLD or score_diff < 0.1
    
    # Find corresponding TargetField objects for candidates
    for candidate in merged:
        if 'target' not in candidate:
            target_field = target_schema.get_target_by_name(candidate['target_field'], targets)
            candidate['target'] = target_field
    
    if use_llm:
        # Call LLM for disambiguation
        llm_result = llm_service.llm_select(profile, merged, memory_candidates)
        
        # Find the target in merged candidates
        selected_target_name = llm_result.get('selected_target')
        selected_candidate = None
        
        for candidate in merged:
            if _normalize_for_match(candidate.get('target_field') or "") == _normalize_for_match(selected_target_name or ""):
                selected_candidate = candidate
                break
        
        if not selected_candidate:
            # LLM selected something not in candidates, use top candidate
            selected_candidate = merged[0]
        
        # Combine confidence with heuristic match if available (prefer exact over fuzzy)
        heuristic_match = exact or fuzzy
        final_confidence = combine_confidence(llm_result, merged, heuristic_match)
        
        return create_decision(
            selected_candidate,
            llm_result.get('explanation', 'LLM-based selection'),
            final_confidence,
            profile.name,
            merged
        )
    else:
        # Use top merged candidate without LLM
        top = merged[0]
        return create_decision(
            top,
            f"Top candidate from merged search (source: {top.get('source', 'unknown')})",
            top['score'],
            profile.name,
            merged
        )


def decide(profile: ColumnProfile, targets: List[TargetField], 
          tenant_name: str = "") -> MappingDecision:
    """
    Main entry point for mapping decisions.
    
    Wrapper around decide_with_pipeline for backward compatibility.
    
    Args:
        profile: ColumnProfile with source column details
        targets: List of all TargetField objects
        tenant_name: Optional tenant identifier
    
    Returns:
        MappingDecision with selected target and confidence
    """
    return decide_with_pipeline(profile, targets, tenant_name)


def decide_bulk(profiles: List[ColumnProfile], targets: List[TargetField],
               tenant_name: str = "") -> List[MappingDecision]:
    """
    Make mapping decisions for multiple source columns in batch.
    
    Args:
        profiles: List of source column profiles to map
        targets: All available target fields
        tenant_name: Optional tenant identifier
    
    Returns:
        List of MappingDecision objects, one per source column
    """
    return [decide_with_pipeline(p, targets, tenant_name) for p in profiles]
