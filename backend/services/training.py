"""
Training ingestion pipeline for learning from pre-mapped Excel files.

Compares agent-generated mappings with human-provided mappings,
saves only mismatches to memory for learning.
"""
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

from backend.models import ColumnProfile
from backend.services import decision, memory, target_schema
from backend.services.target_schema import _normalize_for_match


def load_training_excel_or_csv(file_path: str) -> pd.DataFrame:
    """
    Load training Excel with source/target mappings.
    
    Args:
        file_path: Path to Excel or CSV file
    
    Returns:
        DataFrame with training data
    
    Raises:
        ValueError: If required columns are missing
    """
    print(f"[TRAINING] Loading file: {file_path}")
    if file_path.endswith('.csv'):
        # Try multiple encodings — CSV files from Windows often use cp1252/latin-1
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"[TRAINING] CSV loaded successfully (encoding: {encoding})")
                break
            except UnicodeDecodeError:
                print(f"[TRAINING] Encoding {encoding} failed, trying next...")
        else:
            # Final fallback: latin-1 never raises UnicodeDecodeError
            df = pd.read_csv(file_path, encoding="latin-1")
            print(f"[TRAINING] CSV loaded with latin-1 fallback")
    else:
        df = pd.read_excel(file_path)
        print(f"[TRAINING] Excel loaded successfully")
        
    
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    print(f"[TRAINING] Shape: {df.shape} (rows, columns)")
    print(f"[TRAINING] Columns found: {list(df.columns)}")
    
    # Expected columns: source_field_name, target_field_name, section (optional)
    required = ["source_field_name", "target_field_name"]
    
    if not all(col in df.columns for col in required):
        print(f"[TRAINING ERROR] Missing required columns!")
        print(f"[TRAINING ERROR] Required: {required}")
        print(f"[TRAINING ERROR] Found: {list(df.columns)}")
        raise ValueError(f"Excel must contain columns: {required}")
    
    print(f"[TRAINING] Required columns validated: {required}")
    return df


def filter_valid_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid rows:
    - source_field_name is NULL (required-only targets)
    - target_field_name is NULL
    
    Args:
        df: DataFrame with training data
    
    Returns:
        Filtered DataFrame with only valid mappings
    """
    print(f"[TRAINING] Filtering valid mappings...")
    initial_count = len(df)
    print(f"[TRAINING] Initial row count: {initial_count}")
    
    df = df.dropna(subset=["source_field_name", "target_field_name"])
    print(f"[TRAINING] After dropping NULLs: {len(df)} rows")
    
    df = df[df["source_field_name"].str.strip() != ""]
    print(f"[TRAINING] After removing empty source: {len(df)} rows")
    
    df = df[df["target_field_name"].str.strip() != ""]
    print(f"[TRAINING] After removing empty target: {len(df)} rows")
    
    removed = initial_count - len(df)
    print(f"[TRAINING] Filtered out {removed} invalid rows")
    
    return df


def generate_agent_mappings(source_columns: List[str], targets, tenant_name: str) -> Dict[str, str]:
    """
    Run Mapper agent on source columns to get predictions.
    
    Args:
        source_columns: List of source column names
        targets: List of target fields
        tenant_name: Tenant identifier
    
    Returns:
        Dictionary mapping source_column to predicted_target_field
    """
    print(f"[TRAINING] Generating agent mappings for {len(source_columns)} columns...")
    agent_mappings = {}
    
    for idx, source_col in enumerate(source_columns, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"[TRAINING] Processing column {idx}/{len(source_columns)}: {source_col}")
        
        # Create minimal profile for agent
        profile = ColumnProfile(
            name=source_col,
            inferred_dtype="string",  # Default, could be enhanced
            null_ratio=0.0,
            avg_length=None,
            max_length=None,
            uniqueness_ratio=0.5
        )
        
        # Run decision pipeline
        decision_result = decision.decide(profile, targets, tenant_name)
        
        predicted_target = None
        if decision_result.selected_target:
            predicted_target = decision_result.selected_target.field_name
        
        agent_mappings[source_col] = predicted_target
        
        if idx % 10 == 0 or idx == 1:
            print(f"[TRAINING]   -> Predicted: {predicted_target}")
    
    print(f"[TRAINING] Agent mapping complete: {len(agent_mappings)} predictions")
    return agent_mappings


def find_mismatches(df: pd.DataFrame, agent_mappings: Dict[str, str]) -> List[Dict]:
    """
    Compare agent predictions with Excel ground truth.
    Return only mismatches.
    
    Args:
        df: DataFrame with ground truth mappings
        agent_mappings: Dictionary of agent predictions
    
    Returns:
        List of mismatch dictionaries
    """
    print(f"[TRAINING] Finding mismatches...")
    mismatches = []
    
    for idx, row in df.iterrows():
        source = row["source_field_name"]
        correct_target = row["target_field_name"]
        predicted_target = agent_mappings.get(source)
        
        # Check for mismatch (normalized comparison)
        if _normalize_for_match(str(predicted_target or "")) != _normalize_for_match(str(correct_target or "")):
            # Get category and handle NaN values from pandas
            category = row.get("section", "") if "section" in row else ""
            # Convert pandas NaN to empty string
            if pd.isna(category):
                category = ""
            else:
                category = str(category).strip()
            
            mismatches.append({
                "source_column": source,
                "correct_target": correct_target,
                "predicted_target": predicted_target,
                "category": category
            })
            
            # Log first few mismatches
            if len(mismatches) <= 5:
                print(f"[TRAINING] Mismatch #{len(mismatches)}: {source}")
                print(f"[TRAINING]   Expected: {correct_target}")
                print(f"[TRAINING]   Predicted: {predicted_target}")
                print(f"[TRAINING]   Category: '{category}'")
    
    print(f"[TRAINING] Total mismatches found: {len(mismatches)}")
    return mismatches


def ingest_training_data(file_path: str, tenant_name: str, targets=None) -> Dict[str, Any]:
    """
    Main training ingestion pipeline.
    
    Steps:
    1. Load Excel
    2. Filter valid mappings
    3. Run agent on source columns
    4. Compare and find mismatches
    5. Save mismatches to memory (BULK)
    
    Args:
        file_path: Path to training Excel file
        tenant_name: Tenant identifier
        targets: List of target fields
    
    Returns:
        Dictionary with ingestion statistics
    
    Raises:
        ValueError: If tenant_name is empty
    """
    print("\n" + "="*60)
    print("[TRAINING INGESTION] Starting pipeline")
    print("="*60)
    print(f"[TRAINING] File path: {file_path}")
    print(f"[TRAINING] Tenant name: {tenant_name}")
    print(f"[TRAINING] Targets provided: {targets is not None}")
    if targets:
        print(f"[TRAINING] Number of target fields: {len(targets)}")
    
    if not tenant_name:
        print("[TRAINING ERROR] tenant_name is required")
        raise ValueError("tenant_name is required for training ingestion")
    
    # Load and validate
    print("\n[TRAINING STEP 1] Loading Excel file...")
    df = load_training_excel_or_csv(file_path)
    
    print("\n[TRAINING STEP 2] Filtering valid mappings...")
    df = filter_valid_mappings(df)
    
    total_rows = len(df)
    source_columns = df["source_field_name"].tolist()
    print(f"[TRAINING] Valid mappings: {len(source_columns)}")
    
    # Generate agent predictions
    print("\n[TRAINING STEP 3] Generating agent predictions...")
    print(f"[TRAINING] Processing {len(source_columns)} source columns...")
    agent_mappings = generate_agent_mappings(source_columns, targets, tenant_name)
    
    # Find mismatches
    print("\n[TRAINING STEP 4] Comparing predictions with ground truth...")
    mismatches = find_mismatches(df, agent_mappings)
    
    print(f"\n[TRAINING] Results:")
    print(f"[TRAINING]   Total valid mappings: {total_rows}")
    print(f"[TRAINING]   Agent correct: {total_rows - len(mismatches)}")
    print(f"[TRAINING]   Mismatches: {len(mismatches)}")
    accuracy = (total_rows - len(mismatches)) / total_rows if total_rows > 0 else 0
    print(f"[TRAINING]   Accuracy: {accuracy*100:.1f}%")
    
    # Save mismatches to memory (BULK)
    if mismatches:
        print(f"\n[TRAINING STEP 5] Saving {len(mismatches)} mismatches to memory...")
        for idx, mismatch in enumerate(mismatches, 1):
            
            if idx <= 3 or idx == len(mismatches):
                print(f"[TRAINING] Saving mismatch {idx}/{len(mismatches)}: {mismatch['source_column']} -> {mismatch['correct_target']} (category: '{mismatch['category']}')")
            
            memory.add_memory_record(
                source_column=mismatch["source_column"],
                target_field=mismatch["correct_target"],
                confidence=1.0,  # Training data = 100% confidence
                tenant_name=tenant_name,
                category=mismatch["category"],
                approval_source="training_ingestion"
            )
        
        print(f"[TRAINING] All {len(mismatches)} mismatches saved to memory")
        
        # Rebuild indexes if mismatches were added
        print("[TRAINING] Rebuilding FAISS index...")
        memory_records = memory.get_all_records()
        from backend.services import vector_store
        vector_store.rebuild_index(memory_records)
        print("[TRAINING] FAISS index rebuilt")
    else:
        print("\n[TRAINING] No mismatches to save - agent was 100% accurate!")
    
    result = {
        "total_rows": total_rows,
        "valid_mappings": len(source_columns),
        "agent_correct": total_rows - len(mismatches),
        "mismatches_saved": len(mismatches),
        "accuracy": accuracy,
        "tenant_name": tenant_name,
        "mismatches": mismatches  # Include details for review
    }
    
    print("\n[TRAINING] Ingestion pipeline complete!")
    print("="*60)
    
    return result
