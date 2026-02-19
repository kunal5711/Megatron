"""
Configuration constants and settings for the Mapper application.

This module centralizes all configuration values including:
- MongoDB database settings
- Heuristic scoring weights
- Confidence thresholds for decision classification
- Type inference parameters
"""
import os
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Base paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================================
# MongoDB configuration
# ============================================================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "mapper_db")

# Collection names
MEMORY_COLLECTION = "memory"
AUDIT_MEMORY_COLLECTION = "audit_memory"
SAVED_TASKS_COLLECTION = "saved_tasks"

# ============================================================================
# LLM integration configuration (OpenRouter API)
# https://openrouter.ai/docs
# ============================================================================
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Model ID from OpenRouter (e.g. openai/gpt-4o-mini, google/gemini-flash-1.5, anthropic/claude-3-haiku)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Legacy / generic LLM config (used by llm_service)
LLM_API_URL = OPENROUTER_API_URL
LLM_API_KEY = OPENROUTER_API_KEY
LLM_MODEL = OPENROUTER_MODEL

# ============================================================================
# Heuristic scoring weights
# ============================================================================
NAME_WEIGHT = 0.5
DATATYPE_WEIGHT = 0.2
LENGTH_WEIGHT = 0.15
CATEGORY_WEIGHT = 0.15

# ============================================================================
# Decision classification thresholds
# ============================================================================
AUTO_APPROVE_THRESHOLD = 0.95
NEEDS_REVIEW_THRESHOLD = 0.8
HEURISTIC_THRESHOLD = 0.95

# ============================================================================
# Candidate shortlisting
# ============================================================================
TOP_K = 5

# ============================================================================
# CSV sampling configuration
# ============================================================================
SAMPLE_ROWS = 50

# ============================================================================
# Type inference thresholds
# ============================================================================
NUMERIC_RATIO_THRESHOLD = 0.9
DATE_RATIO_THRESHOLD = 0.6
BOOLEAN_RATIO_THRESHOLD = 0.8

# ============================================================================
# Vector store configuration
# ============================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_HUB_DISABLE_SSL_VERIFY = os.getenv("HF_HUB_DISABLE_SSL_VERIFY", "1").lower() in ("1", "true", "yes")
DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "1").lower() in ("1", "true", "yes")
FAISS_TOP_K = 5

# ============================================================================
# LLM Rate Limiting Configuration
# ============================================================================
LLM_MAX_RPM = int(os.getenv("LLM_MAX_RPM", "30"))
LLM_MAX_RPD = int(os.getenv("LLM_MAX_RPD", "14400"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "2.0"))
LLM_MAX_WAIT_TIME = float(os.getenv("LLM_MAX_WAIT_TIME", "120.0"))
ENABLE_LLM_CACHE = os.getenv("ENABLE_LLM_CACHE", "1").lower() in ("1", "true", "yes")


def thresholds() -> Tuple[float, float]:
    """
    Get decision classification thresholds.

    Returns:
        Tuple of (AUTO_APPROVE_THRESHOLD, NEEDS_REVIEW_THRESHOLD)
    """
    return AUTO_APPROVE_THRESHOLD, NEEDS_REVIEW_THRESHOLD
