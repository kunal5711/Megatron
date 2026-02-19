"""
Pydantic models for data structures used throughout the application.

These models define the schema for:
- Source column profiles and characteristics
- Target field definitions
- Mapping decisions and candidate scores
- API request/response models
- MongoDB memory, audit_memory, and saved_tasks documents
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


def _coerce_optional_str(v: Any) -> str:
    """Coerce value to str for optional string fields (JSON may have false/null)."""
    if v is None or isinstance(v, bool):
        return ""
    if isinstance(v, str):
        return v
    return str(v)


class ColumnProfile(BaseModel):
    """Profile of a source column with inferred characteristics."""
    name: str
    inferred_dtype: str
    null_ratio: float
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    uniqueness_ratio: float


class TargetField(BaseModel):
    """Definition of a target field in the destination schema."""
    field_name: str
    description: Optional[str] = ""
    datatype: Optional[str] = ""
    category: Optional[str] = ""
    required: Optional[str] = "false"

    @field_validator("description", "datatype", "category", "required", mode="before")
    @classmethod
    def coerce_str_fields(cls, v: Any) -> str:
        return _coerce_optional_str(v)


class CandidateScore(BaseModel):
    """Score for a source column matched against a target field."""
    target: TargetField
    total_score: float
    breakdown: Dict[str, float]
    source_column: str
    boosted: bool = False


class MappingDecision(BaseModel):
    """Final mapping decision for a source column."""
    source_column: str
    selected_target: Optional[TargetField] = None
    decision: str
    confidence: float
    explanation: Optional[str] = ""
    candidates: List[CandidateScore] = Field(default_factory=list)


# ============================================================================
# API Request/Response Models
# ============================================================================

class FileUploadResponse(BaseModel):
    columns: List[str]
    sample_rows: List[Dict[str, Any]]


class ProfileResponse(BaseModel):
    profiles: List[ColumnProfile]


class MappingRequest(BaseModel):
    profiles: List[ColumnProfile]
    tenant_name: Optional[str] = ""

    @field_validator("tenant_name")
    @classmethod
    def validate_tenant_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tenant_name is required and cannot be empty")
        return v.strip()


class MappingResponse(BaseModel):
    mappings: List[MappingDecision]


class FeedbackRequest(BaseModel):
    source_column: str
    approved_target: str
    notes: Optional[str] = ""
    tenant_name: Optional[str] = ""
    confidence: float = 1.0


class HistoryRecord(BaseModel):
    source_column: str
    approved_target: str
    category: Optional[str] = ""
    datatype: Optional[str] = ""
    description: Optional[str] = ""
    tenant_name: Optional[str] = ""


class HistoryResponse(BaseModel):
    records: List[HistoryRecord]


# ============================================================================
# Review Workflow Models
# ============================================================================

class ReviewAction(BaseModel):
    source_column: str
    original_target: Optional[str] = None
    action: str
    corrected_target: Optional[str] = None
    notes: str = ""


class ReviewSubmission(BaseModel):
    session_id: str
    tenant_name: str
    reviews: List[ReviewAction]

    @field_validator("tenant_name")
    @classmethod
    def validate_tenant_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tenant_name is required and cannot be empty")
        return v.strip()


class ReviewStartRequest(BaseModel):
    mappings: MappingResponse
    tenant_name: str

    @field_validator("tenant_name")
    @classmethod
    def validate_tenant_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tenant_name is required and cannot be empty")
        return v.strip()


class FinalApprovalRequest(BaseModel):
    session_id: str


class TrainingIngestionRequest(BaseModel):
    tenant_name: str
    category: Optional[str] = ""

    @field_validator("tenant_name")
    @classmethod
    def validate_tenant_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tenant_name is required and cannot be empty")
        return v.strip()


class MappingDownloadResponse(BaseModel):
    mappings: List[Dict[str, Any]]
    tenant_name: str
    timestamp: str


# ============================================================================
# MongoDB Memory Models
# ============================================================================

class MemoryContext(BaseModel):
    """Context metadata for a memory record."""
    tenant_name: Optional[str] = ""
    category: Optional[str] = ""
    approval_source: Optional[str] = "system"
    timestamp: Optional[str] = ""


class MemoryRecord(BaseModel):
    """
    In-memory representation of a MongoDB memory document.
    Used for heuristic matching and vector search.
    """
    source_column: str
    target_field: str
    confidence: float
    context: MemoryContext
    status: str = "ACTIVE"
    usage_count: int = 1
    memory_source: str = "system"
    created_date: Optional[str] = ""


# ============================================================================
# Memory Management API Models
# ============================================================================

class MemoryCreateRequest(BaseModel):
    """Request to create a new memory record."""
    source_column: str
    target_field: str
    tenant_name: str = ""
    category: str = ""
    memory_source: str = "manual"


class MemoryUpdateRequest(BaseModel):
    """Request to update a memory record (disable old + insert new)."""
    memory_id: str
    source_column: str
    new_target_field: str
    tenant_name: str = ""
    category: str = ""


class MemoryDeleteRequest(BaseModel):
    """Request to delete (disable) a memory record."""
    memory_id: str


class PendingChange(BaseModel):
    """A single pending change in the bulk commit queue."""
    action: str  # "CREATE", "UPDATE", "DELETE"
    data: Dict[str, Any]


class BulkCommitRequest(BaseModel):
    """Request to commit all pending changes in bulk."""
    changes: List[PendingChange]


# ============================================================================
# Saved Tasks Models
# ============================================================================

class SaveTaskRequest(BaseModel):
    """Request to save a mapping review task."""
    task_name: str
    tenant_name: str
    mapping_data: List[Dict[str, Any]]

    @field_validator("tenant_name")
    @classmethod
    def validate_tenant_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tenant_name is required")
        return v.strip()

    @field_validator("task_name")
    @classmethod
    def validate_task_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("task_name is required")
        return v.strip()


class UpdateTaskStatusRequest(BaseModel):
    """Request to update a saved task status."""
    task_id: str
    review_status: str  # "SAVED", "COMPLETED"
