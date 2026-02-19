"""
Column profiling and data type inference.

This module analyzes uploaded CSV data to infer column characteristics:
- Data types (number, string, date, boolean)
- Statistical properties (null ratio, length, uniqueness)
- Sample values for preview

These profiles are used by the matching system to score compatibility with target fields.
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from backend import config
from backend.models import ColumnProfile


def infer_dtype(series: pd.Series) -> str:
    """
    Infer the data type of a pandas Series by analyzing its values.
    
    Uses a hierarchical approach with configurable thresholds:
    1. Check if values are numeric (90% threshold)
    2. Check if values are dates (60% threshold)
    3. Check if values are boolean-like (80% threshold)
    4. Default to string if none match
    
    Args:
        series: Pandas Series containing column data
    
    Returns:
        Inferred data type: "number", "date", "boolean", or "string"
    """
    # Step 1: Attempt to convert to numeric
    # errors="coerce" converts non-numeric values to NaN
    numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
    if numeric_ratio >= config.NUMERIC_RATIO_THRESHOLD:
        return "number"
    
    # Step 2: Attempt to parse as datetime
    # Lower threshold (0.6) because date formats vary widely
    datetime_ratio = pd.to_datetime(series, errors="coerce").notna().mean()
    if datetime_ratio >= config.DATE_RATIO_THRESHOLD:
        return "date"
    
    # Step 3: Check for boolean-like values
    # Convert to lowercase strings and check for common boolean patterns
    lowered = series.dropna().astype(str).str.lower()
    bool_ratio = lowered.isin(["true", "false", "yes", "no", "0", "1"]).mean() if not lowered.empty else 0
    if bool_ratio >= config.BOOLEAN_RATIO_THRESHOLD:
        return "boolean"
    
    # Step 4: Default to string for everything else
    return "string"


def profile_columns(rows: List[Dict[str, Any]]) -> List[ColumnProfile]:
    """
    Analyze CSV rows and create profiles for each column.
    
    For each column, calculates:
    - Inferred data type
    - Null ratio (proportion of missing values)
    - Average and maximum string lengths
    - Uniqueness ratio (proportion of unique values)
    - Sample values (first 5 rows)
    
    Args:
        rows: List of dictionaries representing CSV rows (column name → value)
    
    Returns:
        List of ColumnProfile objects, one per column
    """
    # Handle empty input
    if not rows:
        return []
    
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(rows)
    profiles: List[ColumnProfile] = []
    
    # Profile each column
    for col in df.columns:
        series = df[col]
        
        # Calculate null ratio (proportion of missing values)
        null_ratio = series.isna().mean()
        
        # Infer the data type
        inferred = infer_dtype(series)
        
        # Calculate length statistics (for string columns)
        # Convert all values to strings to measure length
        lengths = series.dropna().astype(str).str.len()
        avg_len = float(lengths.mean()) if not lengths.empty else None
        max_len = int(lengths.max()) if not lengths.empty else None
        
        # Calculate uniqueness ratio (proportion of unique values)
        # High ratio = mostly unique (e.g., IDs), low ratio = many duplicates
        uniqueness_ratio = series.nunique(dropna=True) / len(series) if len(series) else 0
        
        # Extract sample values (first 5 rows) for preview
        
        # Create profile object
        profiles.append(
            ColumnProfile(
                name=col,
                inferred_dtype=inferred,
                null_ratio=float(null_ratio),
                avg_length=avg_len,
                max_length=max_len,
                uniqueness_ratio=float(uniqueness_ratio)
            )
        )
    
    return profiles