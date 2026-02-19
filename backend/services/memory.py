"""
Persistent memory using MongoDB for storing and retrieving mapping history.

All memory operations use MongoDB collections:
- memory: Active and disabled mapping records
- audit_memory: Audit trail for updates and deletes

Key rules:
- All inference-related fetches use { status: "ACTIVE" }
- Memory rows are immutable: updates disable old + insert new
- Deletes set status = DISABLED
- Audit records track all changes
"""
from typing import List, Optional, Dict, Any
from datetime import datetime

import pandas as pd
from bson import ObjectId
from pymongo import MongoClient
from rapidfuzz import fuzz

from backend.config import (
    MONGO_URI, MONGO_DB_NAME,
    MEMORY_COLLECTION, AUDIT_MEMORY_COLLECTION,
)
from backend.models import MemoryRecord, MemoryContext
from backend.services.target_schema import _normalize_for_match

# ============================================================================
# MongoDB connection (lazy singleton)
# ============================================================================
_client: Optional[MongoClient] = None
_db = None


def _get_db():
    """Return the MongoDB database handle, creating the connection if needed."""
    global _client, _db
    if _db is None:
        _client = MongoClient(MONGO_URI)
        _db = _client[MONGO_DB_NAME]
    return _db


def _memory_col():
    """Shortcut to the memory collection."""
    return _get_db()[MEMORY_COLLECTION]


def _audit_col():
    """Shortcut to the audit_memory collection."""
    return _get_db()[AUDIT_MEMORY_COLLECTION]


# ============================================================================
# Cached in-process data for heuristic matching
# ============================================================================
_HEURISTIC_TABLE: Optional[pd.DataFrame] = None


def _doc_to_record(doc: dict) -> MemoryRecord:
    """Convert a MongoDB document to a MemoryRecord model."""
    return MemoryRecord(
        source_column=doc.get("source_column", ""),
        target_field=doc.get("target_field", ""),
        confidence=doc.get("confidence", 1.0),
        context=MemoryContext(
            tenant_name=doc.get("tenant_name", ""),
            category=doc.get("category", ""),
            approval_source=doc.get("memory_source", "system"),
            timestamp=doc.get("created_date", ""),
        ),
        status=doc.get("status", "ACTIVE"),
        usage_count=doc.get("usage_count", 1),
        memory_source=doc.get("memory_source", "system"),
        created_date=doc.get("created_date", ""),
    )


# ============================================================================
# Core CRUD helpers
# ============================================================================

def load_active_memory() -> List[MemoryRecord]:
    """
    Load all ACTIVE memory records from MongoDB.

    This is the ONLY function that should be used for inference/matching.
    It always filters by { status: "ACTIVE" }.

    Returns:
        List of MemoryRecord with status ACTIVE
    """
    docs = _memory_col().find({"status": "ACTIVE"}).sort("created_date", -1)
    return [_doc_to_record(d) for d in docs]


def load_all_memory(status_filter: str = "ACTIVE") -> List[dict]:
    """
    Load memory documents as raw dicts for the management UI.

    Args:
        status_filter: "ACTIVE", "DISABLED", or "ALL"

    Returns:
        List of raw MongoDB documents (with _id converted to string)
    """
    query: dict = {}
    if status_filter != "ALL":
        query["status"] = status_filter

    docs = list(_memory_col().find(query).sort("created_date", -1))
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return docs


def get_all_records() -> List[MemoryRecord]:
    """
    Get all ACTIVE memory records as MemoryRecord objects.
    Used for FAISS index building and heuristic table creation.
    """
    return load_active_memory()


def add_memory_record(
    source_column: str,
    target_field: str,
    confidence: float,
    tenant_name: str = "",
    category: str = "",
    approval_source: str = "system",
) -> str:
    """
    Insert a new ACTIVE memory record into MongoDB.
    Normalizes source_column and target_field before saving.

    Returns:
        The string representation of the inserted _id
    """
    now = datetime.utcnow().isoformat()
    doc = {
        "source_column": _normalize_for_match(source_column or ""),
        "target_field": _normalize_for_match(target_field or ""),
        "confidence": confidence,
        "tenant_name": tenant_name,
        "category": category,
        "memory_source": approval_source,
        "usage_count": 1,
        "status": "ACTIVE",
        "created_date": now,
    }
    result = _memory_col().insert_one(doc)
    return str(result.inserted_id)


def disable_memory_record(memory_id: str) -> bool:
    """
    Set a memory record's status to DISABLED (soft delete).

    Args:
        memory_id: The string _id of the document

    Returns:
        True if a document was modified
    """
    result = _memory_col().update_one(
        {"_id": ObjectId(memory_id)},
        {"$set": {"status": "DISABLED"}},
    )
    return result.modified_count > 0


def insert_audit_record(
    memory_id: str,
    old_target_value: str,
    new_target_value: str,
    audit_source: str = "ui",
) -> str:
    """
    Insert an audit trail record into audit_memory.

    Args:
        memory_id: The _id of the affected memory document
        old_target_value: Previous target_field value
        new_target_value: New target_field value (empty string for deletes)
        audit_source: "ui", "system", "training", etc.

    Returns:
        The string _id of the inserted audit document
    """
    now = datetime.utcnow().isoformat()
    doc = {
        "memory_id": memory_id,
        "old_target_value": old_target_value,
        "new_target_value": new_target_value,
        "audit_source": audit_source,
        "created_date": now,
    }
    result = _audit_col().insert_one(doc)
    return str(result.inserted_id)


# ============================================================================
# Bulk commit (used by the Memory Management UI "Final Commit")
# ============================================================================

def bulk_commit(changes: list) -> dict:
    """
    Apply a list of pending CREATE / UPDATE / DELETE operations using
    MongoDB bulkWrite for atomicity.

    Each change is a dict with:
        { "action": "CREATE" | "UPDATE" | "DELETE", "data": { ... } }

    UPDATE logic:
        1. Disable old record (status = DISABLED)
        2. Insert new record with updated values
        3. Insert audit_memory record

    DELETE logic:
        1. Set status = DISABLED
        2. Insert audit_memory record

    Returns:
        Summary dict with counts
    """
    from pymongo import InsertOne, UpdateOne

    if not changes:
        return {"created": 0, "updated": 0, "deleted": 0}

    memory_ops = []
    audit_docs = []
    created = 0
    updated = 0
    deleted = 0
    now = datetime.utcnow().isoformat()

    for change in changes:
        action = change.get("action", "").upper()
        data = change.get("data", {})

        if action == "CREATE":
            memory_ops.append(InsertOne({
                "source_column": _normalize_for_match(data.get("source_column", "") or ""),
                "target_field": _normalize_for_match(data.get("target_field", "") or ""),
                "confidence": data.get("confidence", 1.0),
                "tenant_name": data.get("tenant_name", ""),
                "category": data.get("category", ""),
                "memory_source": data.get("memory_source", "manual"),
                "usage_count": 1,
                "status": "ACTIVE",
                "created_date": now,
            }))
            created += 1

        elif action == "UPDATE":
            old_id = data.get("memory_id", "")
            if old_id:
                # Fetch old document for audit
                old_doc = _memory_col().find_one({"_id": ObjectId(old_id)})
                old_target = old_doc.get("target_field", "") if old_doc else ""

                # Disable old record
                memory_ops.append(UpdateOne(
                    {"_id": ObjectId(old_id)},
                    {"$set": {"status": "DISABLED"}},
                ))
                # Insert new record
                memory_ops.append(InsertOne({
                    "source_column": _normalize_for_match(data.get("source_column", "") or ""),
                    "target_field": _normalize_for_match(data.get("new_target_field", "") or ""),
                    "confidence": data.get("confidence", 1.0),
                    "tenant_name": data.get("tenant_name", ""),
                    "category": data.get("category", ""),
                    "memory_source": data.get("memory_source", "manual"),
                    "usage_count": 1,
                    "status": "ACTIVE",
                    "created_date": now,
                }))
                # Audit
                audit_docs.append({
                    "memory_id": old_id,
                    "old_target_value": old_target,
                    "new_target_value": data.get("new_target_field", ""),
                    "audit_source": "ui",
                    "created_date": now,
                })
                updated += 1

        elif action == "DELETE":
            del_id = data.get("memory_id", "")
            if del_id:
                old_doc = _memory_col().find_one({"_id": ObjectId(del_id)})
                old_target = old_doc.get("target_field", "") if old_doc else ""

                memory_ops.append(UpdateOne(
                    {"_id": ObjectId(del_id)},
                    {"$set": {"status": "DISABLED"}},
                ))
                audit_docs.append({
                    "memory_id": del_id,
                    "old_target_value": old_target,
                    "new_target_value": "",
                    "audit_source": "ui",
                    "created_date": now,
                })
                deleted += 1

    # Execute bulk write on memory collection
    if memory_ops:
        _memory_col().bulk_write(memory_ops, ordered=False)

    # Insert audit records
    if audit_docs:
        _audit_col().insert_many(audit_docs)

    return {"created": created, "updated": updated, "deleted": deleted}


def normalize_all_memory_records() -> dict:
    """
    Normalize source_column and target_field in all memory documents.
    Uses _normalize_for_match (lowercase, spaces to underscores).

    Returns:
        dict with "updated": count of documents modified
    """
    from pymongo import UpdateOne

    docs = list(_memory_col().find({}))
    ops = []
    for doc in docs:
        doc_id = doc["_id"]
        src = doc.get("source_column", "")
        tgt = doc.get("target_field", "")
        norm_src = _normalize_for_match(src or "")
        norm_tgt = _normalize_for_match(tgt or "")
        if norm_src != src or norm_tgt != tgt:
            ops.append(UpdateOne(
                {"_id": doc_id},
                {"$set": {"source_column": norm_src, "target_field": norm_tgt}},
            ))

    if ops:
        result = _memory_col().bulk_write(ops, ordered=False)
        return {"updated": result.modified_count}
    return {"updated": 0}


# ============================================================================
# Heuristic table (pandas DataFrame for fast matching)
# ============================================================================

def build_heuristic_table(memory_records: List[MemoryRecord]) -> pd.DataFrame:
    """
    Convert ACTIVE memory records into a pandas DataFrame for heuristic matching.
    """
    global _HEURISTIC_TABLE

    if not memory_records:
        _HEURISTIC_TABLE = pd.DataFrame(columns=[
            'source_column', 'target_field', 'usage_frequency',
            'avg_confidence', 'tenant_name', 'category',
        ])
        return _HEURISTIC_TABLE

    rows = []
    for rec in memory_records:
        rows.append({
            'source_column': _normalize_for_match(rec.source_column),
            'target_field': _normalize_for_match(rec.target_field),
            'confidence': rec.confidence,
            'tenant_name': rec.context.tenant_name or "",
            'category': rec.context.category or "",
        })

    df = pd.DataFrame(rows)
    grouped = df.groupby(['source_column', 'target_field', 'tenant_name']).agg({
        'confidence': ['mean', 'count'],
        'category': 'first',
    }).reset_index()
    grouped.columns = [
        'source_column', 'target_field', 'tenant_name',
        'avg_confidence', 'usage_frequency', 'category',
    ]

    _HEURISTIC_TABLE = grouped
    return _HEURISTIC_TABLE


# ============================================================================
# Heuristic matching functions
# ============================================================================

def exact_match(source_column: str, tenant_name: str = "") -> Optional[Dict]:
    """Exact lookup in the heuristic table with tenant priority."""
    if _HEURISTIC_TABLE is None or _HEURISTIC_TABLE.empty:
        return None

    normalized = _normalize_for_match(source_column)
    matches = _HEURISTIC_TABLE[_HEURISTIC_TABLE['source_column'] == normalized]

    if matches.empty:
        return None

    if tenant_name:
        tenant_matches = matches[matches['tenant_name'] == tenant_name]
        if not tenant_matches.empty:
            best = tenant_matches.loc[tenant_matches['usage_frequency'].idxmax()]
            return {
                'target_field': best['target_field'],
                'confidence': 0.95,
                'reason': 'exact_match_same_tenant',
                'usage_frequency': best['usage_frequency'],
                'is_same_tenant': bool(best['tenant_name'] == tenant_name),
            }

    best = matches.loc[matches['usage_frequency'].idxmax()]
    return {
        'target_field': best['target_field'],
        'confidence': best['avg_confidence'],
        'reason': 'exact_match_cross_tenant',
        'usage_frequency': best['usage_frequency'],
        'is_same_tenant': False,
    }


def fuzzy_match(source_column: str, tenant_name: str = "", threshold: float = 0.75) -> Optional[Dict]:
    """Fuzzy string matching with tenant boosting."""
    if _HEURISTIC_TABLE is None or _HEURISTIC_TABLE.empty:
        return None

    normalized = _normalize_for_match(source_column)
    scores = []
    for _, row in _HEURISTIC_TABLE.iterrows():
        similarity = fuzz.token_sort_ratio(normalized, row['source_column']) / 100.0
        if similarity >= threshold:
            boosted = similarity
            if tenant_name and row['tenant_name'] == tenant_name:
                boosted = min(similarity + 0.15, 1.0)
            combined = (boosted * 0.7) + (row['avg_confidence'] * 0.3)
            scores.append({
                'target_field': row['target_field'],
                'confidence': combined,
                'similarity': similarity,
                'usage_frequency': row['usage_frequency'],
                'is_same_tenant': bool(tenant_name and row['tenant_name'] == tenant_name),
            })

    if not scores:
        return None

    best = max(scores, key=lambda x: (x['is_same_tenant'], x['confidence']))
    return {
        'target_field': best['target_field'],
        'confidence': best['confidence'],
        'reason': 'fuzzy_match_same_tenant' if best['is_same_tenant'] else 'fuzzy_match_cross_tenant',
        'similarity': best['similarity'],
        'usage_frequency': best['usage_frequency'],
    }


def get_usage_frequency(target_field: str) -> int:
    """Count occurrences of target_field in ACTIVE memory (normalized for comparison)."""
    if _HEURISTIC_TABLE is None or _HEURISTIC_TABLE.empty:
        return 0
    normalized = _normalize_for_match(target_field or "")
    matches = _HEURISTIC_TABLE[_HEURISTIC_TABLE['target_field'] == normalized]
    return int(matches['usage_frequency'].sum()) if not matches.empty else 0


def find_similar_by_target(target_field: str, limit: int = 5) -> List[MemoryRecord]:
    """Find ACTIVE memory records that map to a specific target field (normalized for comparison)."""
    normalized = _normalize_for_match(target_field or "")
    docs = (
        _memory_col()
        .find({"target_field": normalized, "status": "ACTIVE"})
        .limit(limit)
    )
    return [_doc_to_record(d) for d in docs]
