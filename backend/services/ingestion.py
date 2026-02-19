"""
CSV file ingestion and parsing.

Handles reading and parsing uploaded CSV files, extracting column headers
and sample rows for preview and profiling.
"""
import csv
from io import StringIO
from typing import List, Dict, Any

from fastapi import UploadFile


def read_csv_preview(file: UploadFile, sample_rows: int = 50) -> Dict[str, Any]:
    """
    Read uploaded CSV file and extract headers plus sample rows.
    
    This function:
    - Decodes the file content (handles encoding issues gracefully)
    - Parses CSV format using DictReader (column name → value mapping)
    - Extracts only a sample of rows to avoid loading large files into memory
    - Resets file pointer so the file can be re-read if needed
    
    Args:
        file: FastAPI UploadFile object containing the CSV file
        sample_rows: Number of rows to sample (default: 50)
    
    Returns:
        Dictionary with:
        - "columns": List of column names (headers)
        - "sample_rows": List of dictionaries, each representing a row
    """
    # Read file content and decode as UTF-8
    # errors="replace" handles encoding issues by replacing invalid characters
    # This makes the function tolerant of "dirty" CSV files
    content = file.file.read().decode("utf-8", errors="replace")
    
    # Parse CSV using DictReader (creates dict per row: column_name → value)
    reader = csv.DictReader(StringIO(content))
    headers = reader.fieldnames or []
    headers = [header.lower().replace(' ', '_') for header in headers]
    
    # Extract sample rows (limit to avoid memory issues with large files)
    rows: List[Dict[str, Any]] = []
    for _, row in zip(range(sample_rows), reader):
        rows.append(row)
    
    # Reset file pointer to beginning so file can be re-read if needed
    # (e.g., for profiling after preview)
    file.file.seek(0)
    
    return {"columns": headers, "sample_rows": rows}