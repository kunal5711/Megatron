"""
FastAPI application for the Mapper.

Endpoints:
- CSV upload, profiling, mapping generation
- Review workflow (start, submit, finalize)
- Training ingestion
- Memory management (CRUD, bulk commit)
- Saved tasks (save, resume, complete, delete)
"""
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.models import (
    FileUploadResponse,
    ProfileResponse,
    MappingRequest,
    MappingResponse,
    FeedbackRequest,
    HistoryResponse,
    HistoryRecord,
    ReviewAction,
    ReviewSubmission,
    ReviewStartRequest,
    FinalApprovalRequest,
    TrainingIngestionRequest,
    MappingDownloadResponse,
    BulkCommitRequest,
    SaveTaskRequest,
    UpdateTaskStatusRequest,
)
from backend.services import ingestion, profiler, target_schema, decision, memory, vector_store
from backend.config import SAMPLE_ROWS, MONGO_URI, MONGO_DB_NAME, SAVED_TASKS_COLLECTION
import uuid
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient

# ============================================================================
# App init
# ============================================================================
app = FastAPI(title="Mapper Phase 3 - Agentic Schema Mapping", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"[VALIDATION ERROR] URL: {request.url}")
    print(f"[VALIDATION ERROR] Errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body) if hasattr(exc, 'body') else None},
    )


# Global variables
TARGETS = []
MEMORY_TABLE = None
FAISS_INDEX = None
TARGET_EMBEDDINGS = None
REVIEW_SESSIONS = {}

# MongoDB client for saved_tasks
_mongo_client = None
_mongo_db = None


def _get_db():
    global _mongo_client, _mongo_db
    if _mongo_db is None:
        _mongo_client = MongoClient(MONGO_URI)
        _mongo_db = _mongo_client[MONGO_DB_NAME]
    return _mongo_db


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global TARGETS, MEMORY_TABLE, FAISS_INDEX, TARGET_EMBEDDINGS

    print("=" * 60)
    print("Initializing Mapper Phase 3")
    print("=" * 60)

    # 1. Load target schema
    print("\n[1/5] Loading target schema...")
    TARGETS = target_schema.load_target_schema()
    print(f"  Loaded {len(TARGETS)} target fields")

    # 2. Load ACTIVE memory from MongoDB and build heuristic table
    print("\n[2/5] Loading memory from MongoDB...")
    memory_records = memory.load_active_memory()
    MEMORY_TABLE = memory.build_heuristic_table(memory_records)
    print(f"  Loaded {len(memory_records)} ACTIVE memory records")
    print(f"  Heuristic table: {len(MEMORY_TABLE)} unique mappings")

    # 3. Build FAISS index
    print("\n[3/5] Building FAISS vector index...")
    FAISS_INDEX = vector_store.build_faiss_index(memory_records)
    if FAISS_INDEX:
        stats = vector_store.get_index_stats()
        print(f"  FAISS: {stats['total_vectors']} vectors, dim={stats['dimension']}")
    else:
        print("  FAISS index not available")

    # 4. Build target embeddings
    print("\n[4/5] Generating target schema embeddings...")
    TARGET_EMBEDDINGS = target_schema.build_target_embeddings(TARGETS)
    print(f"  Generated embeddings for {len(TARGETS)} target fields")

    print("\n[5/5] Initialization complete!")
    print("=" * 60)


# ============================================================================
# CSV / Profile / Map endpoints
# ============================================================================

@app.post("/upload/", response_model=FileUploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    preview = ingestion.read_csv_preview(file, sample_rows=SAMPLE_ROWS)
    return FileUploadResponse(**preview)


@app.post("/profile/", response_model=ProfileResponse)
async def profile_schema(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    preview = ingestion.read_csv_preview(file, sample_rows=SAMPLE_ROWS)
    profiles = profiler.profile_columns(preview["sample_rows"])
    return ProfileResponse(profiles=profiles)


@app.post("/map/", response_model=MappingResponse)
async def generate_mappings(request: MappingRequest):
    tenant_name = request.tenant_name or ""
    mappings = decision.decide_bulk(request.profiles, TARGETS, tenant_name)
    mappings = target_schema.enforce_required_fields(mappings, TARGETS)
    return MappingResponse(mappings=mappings)


# ============================================================================
# Feedback
# ============================================================================

@app.post("/feedback/")
async def submit_feedback(payload: FeedbackRequest):
    global MEMORY_TABLE, FAISS_INDEX

    target = target_schema.get_target_by_name(payload.approved_target, TARGETS)
    if not target:
        raise HTTPException(status_code=404, detail="Target field not found in schema.")

    memory.add_memory_record(
        source_column=payload.source_column,
        target_field=payload.approved_target,
        confidence=payload.confidence,
        tenant_name=payload.tenant_name,
        category=target.category or "",
    )

    memory_records = memory.get_all_records()
    vector_store.rebuild_index(memory_records)
    MEMORY_TABLE = memory.build_heuristic_table(memory_records)
    stats = vector_store.get_index_stats()

    return {
        "status": "ok",
        "memory_size": len(memory_records),
        "faiss_vectors": stats.get('total_vectors', 0),
    }


# ============================================================================
# History & Stats
# ============================================================================

@app.get("/history/", response_model=HistoryResponse)
async def get_history():
    memory_records = memory.get_all_records()
    records = [
        HistoryRecord(
            source_column=r.source_column,
            approved_target=r.target_field,
            category=r.context.category,
            datatype="",
            description="",
            tenant_name=r.context.tenant_name,
        )
        for r in memory_records
    ]
    return HistoryResponse(records=records)


@app.get("/memory/stats/")
async def get_memory_stats():
    from backend.services import llm_service

    memory_records = memory.get_all_records()
    tenant_counts = {}
    for r in memory_records:
        t = r.context.tenant_name or "unknown"
        tenant_counts[t] = tenant_counts.get(t, 0) + 1

    mapped_targets = set(r.target_field for r in memory_records)
    coverage = len(mapped_targets) / len(TARGETS) if TARGETS else 0
    faiss_stats = vector_store.get_index_stats()
    rate_stats = llm_service.get_rate_limiter_stats()
    cache_stats = llm_service.get_cache_stats()

    return {
        "total_memory_records": len(memory_records),
        "unique_source_columns": len(set(r.source_column for r in memory_records)),
        "unique_target_fields": len(mapped_targets),
        "target_schema_size": len(TARGETS),
        "coverage_ratio": coverage,
        "tenant_distribution": tenant_counts,
        "faiss_index": faiss_stats,
        "heuristic_table_size": len(MEMORY_TABLE) if MEMORY_TABLE is not None else 0,
        "llm_rate_limiter": rate_stats,
        "llm_cache": cache_stats,
    }


@app.post("/llm/clear-cache/")
async def clear_llm_cache():
    from backend.services import llm_service
    llm_service.clear_cache()
    return {"status": "ok", "message": "LLM response cache cleared"}


# ============================================================================
# Review Workflow
# ============================================================================

@app.post("/review/start/")
async def start_review_session(request: ReviewStartRequest):
    session_id = str(uuid.uuid4())
    REVIEW_SESSIONS[session_id] = {
        "tenant_name": request.tenant_name,
        "mappings": request.mappings.mappings,
        "corrections": [],
        "created_at": datetime.now(),
    }
    return {"session_id": session_id, "tenant_name": request.tenant_name}


@app.post("/review/submit/")
async def submit_review(review: ReviewSubmission):
    if review.session_id not in REVIEW_SESSIONS:
        raise HTTPException(404, "Review session not found")

    session = REVIEW_SESSIONS[review.session_id]
    if review.tenant_name != session["tenant_name"]:
        raise HTTPException(400, "Tenant name mismatch")

    session["corrections"] = []
    for action in review.reviews:
        if action.action == "reject":
            if not action.corrected_target:
                raise HTTPException(400, f"Missing corrected target for {action.source_column}")
            session["corrections"].append({
                "source_column": action.source_column,
                "original_target": action.original_target,
                "corrected_target": action.corrected_target,
                "notes": action.notes,
            })

    return {
        "status": "ok",
        "corrections_count": len(session["corrections"]),
        "session_id": review.session_id,
    }


@app.post("/review/finalize/")
async def finalize_review(request: FinalApprovalRequest):
    global MEMORY_TABLE
    session_id = request.session_id

    if session_id not in REVIEW_SESSIONS:
        raise HTTPException(404, "Review session not found")

    session = REVIEW_SESSIONS[session_id]
    corrections = session["corrections"]
    tenant_name = session["tenant_name"]

    if corrections:
        for correction in corrections:
            target = target_schema.get_target_by_name(correction["corrected_target"], TARGETS)
            memory.add_memory_record(
                source_column=correction["source_column"],
                target_field=correction["corrected_target"],
                confidence=1.0,
                tenant_name=tenant_name,
                category=target.category if target else "",
                approval_source="human",
            )

        memory_records = memory.get_all_records()
        vector_store.rebuild_index(memory_records)
        MEMORY_TABLE = memory.build_heuristic_table(memory_records)

    final_mappings = []
    for mapping in session["mappings"]:
        correction = next((c for c in corrections if c["source_column"] == mapping.source_column), None)
        if correction:
            final_mappings.append({
                "source_column": '' if mapping.source_column.startswith('<unmapped_') else mapping.source_column,
                "target_field": correction["corrected_target"],
                "status": "corrected",
                "original_suggestion": correction["original_target"],
                "confidence": 1.0,
            })
        else:
            final_mappings.append({
                "source_column": '' if mapping.source_column.startswith('<unmapped_') else mapping.source_column,
                "target_field": mapping.selected_target.field_name if mapping.selected_target else None,
                "status": "approved",
                "confidence": mapping.confidence,
            })

    del REVIEW_SESSIONS[session_id]

    return {
        "status": "finalized",
        "corrections_saved": len(corrections),
        "mappings": final_mappings,
        "download_ready": True,
        "tenant_name": tenant_name,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Training Ingestion
# ============================================================================

@app.post("/train/ingest/")
async def ingest_training(file: UploadFile = File(...), tenant_name: str = Form(...)):
    global MEMORY_TABLE
    from backend.services import training
    import tempfile
    import os

    if not tenant_name or not tenant_name.strip():
        raise HTTPException(400, "tenant_name is required")

    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(400, "Only Excel and csv files are supported")

    suffix = '.csv' if file.filename.endswith('.csv') else '.xlsx'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = training.ingest_training_data(tmp_path, tenant_name.strip(), TARGETS)
        MEMORY_TABLE = memory.build_heuristic_table(memory.get_all_records())
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Training ingestion failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ============================================================================
# Targets
# ============================================================================

@app.get("/targets/")
async def get_all_targets():
    return {
        "targets": [
            {
                "field_name": t.field_name,
                "description": t.description,
                "category": t.category,
                "required": t.required,
            }
            for t in TARGETS
        ]
    }


@app.get("/")
async def root():
    return {"status": "ok", "message": "Mapper Phase 3 API"}


# ============================================================================
# Memory Management Endpoints
# ============================================================================

@app.get("/memory/list/")
async def list_memory(status: str = "ACTIVE"):
    """
    List memory records for the management UI.
    Accepts status query param: ACTIVE (default), DISABLED, ALL
    """
    allowed = {"ACTIVE", "DISABLED", "ALL"}
    if status.upper() not in allowed:
        raise HTTPException(400, f"status must be one of {allowed}")

    docs = memory.load_all_memory(status_filter=status.upper())
    return {"records": docs, "count": len(docs)}


@app.post("/api/normalize-data/")
async def normalize_data():
    """
    Normalize target_schema.json and all memory records.
    - target_schema.json: field_name values (lowercase, spaces to underscores)
    - memory: source_column and target_field values
    Also reloads schema, embeddings, heuristic table, and FAISS index.
    """
    global TARGETS, MEMORY_TABLE, FAISS_INDEX, TARGET_EMBEDDINGS

    schema_count = target_schema.normalize_target_schema_file()
    memory_result = memory.normalize_all_memory_records()

    # Reload schema and rebuild indexes
    TARGETS = target_schema.load_target_schema()
    memory_records = memory.get_all_records()
    MEMORY_TABLE = memory.build_heuristic_table(memory_records)
    FAISS_INDEX = vector_store.build_faiss_index(memory_records)
    TARGET_EMBEDDINGS = target_schema.build_target_embeddings(TARGETS)

    return {
        "status": "ok",
        "schema_fields_updated": schema_count,
        "memory_records_updated": memory_result.get("updated", 0),
        "message": "Normalization complete. Schema and indexes rebuilt.",
    }


@app.post("/memory/commit/")
async def commit_memory_changes(request: BulkCommitRequest):
    """
    Apply pending CREATE / UPDATE / DELETE changes via bulkWrite.
    This is the ONLY endpoint that modifies memory from the UI.
    """
    global MEMORY_TABLE

    changes = [{"action": c.action, "data": c.data} for c in request.changes]
    result = memory.bulk_commit(changes)

    # Rebuild indexes after bulk commit
    memory_records = memory.get_all_records()
    vector_store.rebuild_index(memory_records)
    MEMORY_TABLE = memory.build_heuristic_table(memory_records)

    return {
        "status": "ok",
        "created": result["created"],
        "updated": result["updated"],
        "deleted": result["deleted"],
        "total_active": len(memory_records),
    }


# ============================================================================
# Saved Tasks Endpoints
# ============================================================================

def _tasks_col():
    return _get_db()[SAVED_TASKS_COLLECTION]


@app.post("/tasks/save/")
async def save_task(request: SaveTaskRequest):
    """Save current mapping review state as a task. Does NOT touch memory."""
    now = datetime.utcnow().isoformat()
    doc = {
        "task_name": request.task_name,
        "tenant_name": request.tenant_name,
        "mapping_data": request.mapping_data,
        "review_status": "SAVED",
        "created_date": now,
        "updated_date": now,
    }
    result = _tasks_col().insert_one(doc)
    return {"status": "ok", "task_id": str(result.inserted_id)}


@app.get("/tasks/list/")
async def list_tasks():
    """List all saved tasks sorted by updated_date descending."""
    docs = list(_tasks_col().find().sort("updated_date", -1))
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return {"tasks": docs, "count": len(docs)}


@app.get("/tasks/{task_id}/")
async def get_task(task_id: str):
    """Get a single saved task by ID (for resuming)."""
    try:
        doc = _tasks_col().find_one({"_id": ObjectId(task_id)})
    except Exception:
        raise HTTPException(400, "Invalid task ID")

    if not doc:
        raise HTTPException(404, "Task not found")

    doc["_id"] = str(doc["_id"])
    return doc


@app.put("/tasks/{task_id}/status/")
async def update_task_status(task_id: str, request: UpdateTaskStatusRequest):
    """Update task review_status (SAVED -> COMPLETED)."""
    allowed = {"SAVED", "COMPLETED"}
    if request.review_status not in allowed:
        raise HTTPException(400, f"review_status must be one of {allowed}")

    now = datetime.utcnow().isoformat()
    result = _tasks_col().update_one(
        {"_id": ObjectId(task_id)},
        {"$set": {"review_status": request.review_status, "updated_date": now}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Task not found")

    return {"status": "ok", "review_status": request.review_status}


@app.put("/tasks/{task_id}/update/")
async def update_task_data(task_id: str, request: SaveTaskRequest):
    """Update task mapping data (when saving progress during resume)."""
    now = datetime.utcnow().isoformat()
    result = _tasks_col().update_one(
        {"_id": ObjectId(task_id)},
        {
            "$set": {
                "task_name": request.task_name,
                "tenant_name": request.tenant_name,
                "mapping_data": request.mapping_data,
                "updated_date": now,
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Task not found")
    return {"status": "ok"}


@app.delete("/tasks/{task_id}/")
async def delete_task(task_id: str):
    """Delete a saved task."""
    try:
        result = _tasks_col().delete_one({"_id": ObjectId(task_id)})
    except Exception:
        raise HTTPException(400, "Invalid task ID")

    if result.deleted_count == 0:
        raise HTTPException(404, "Task not found")

    return {"status": "ok"}
