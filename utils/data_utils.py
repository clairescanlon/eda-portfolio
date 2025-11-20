import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify numeric columns in the DataFrame.
    
    Identifies columns with numeric dtypes (int, float) for numeric analysis.
    This is purely informational—no data modification.
    
    Args:
        df: Input DataFrame to analyze
        
    Returns:
        List of column names with numeric dtypes
        
    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'income': [50000.0, 60000.0, 70000.0],
        ...     'name': ['Alice', 'Bob', 'Charlie']
        ... })
        >>> get_numeric_columns(df)
        ['age', 'income']
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify categorical columns in the DataFrame.
    
    Identifies columns with object, category, or bool dtypes for categorical
    analysis. This includes string columns and explicitly categorical columns.
    This is purely informational—no data modification.
    
    Args:
        df: Input DataFrame to analyze
        
    Returns:
        List of column names with categorical dtypes
        
    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'city': ['NYC', 'LA', 'Chicago'],
        ...     'employed': [True, True, False]
        ... })
        >>> get_categorical_columns(df)
        ['city', 'employed']
    """
    return df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify datetime columns in the DataFrame.
    
    Identifies columns with datetime dtypes for temporal analysis.
    This is purely informational—no data modification.
    
    Args:
        df: Input DataFrame to analyze
        
    Returns:
        List of column names with datetime dtypes
        
    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.to_datetime(['2024-01-01', '2024-02-01']),
        ...     'value': [100, 200]
        ... })
        >>> get_datetime_columns(df)
        ['date']
    """
    return df.select_dtypes(include=['datetime64']).columns.tolist()


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Categorize all columns by their data types.
    
    Returns a dictionary mapping each column name to its type category:
    'numeric', 'categorical', or 'datetime'. This helps you understand
    which columns are which type for analysis purposes.
    This is purely informational—no data modification.
    
    Args:
        df: Input DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to type strings
        
    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30],
        ...     'city': ['NYC', 'LA'],
        ...     'date': pd.to_datetime(['2024-01-01', '2024-02-01'])
        ... })
        >>> detect_column_types(df)
        {'age': 'numeric', 'city': 'categorical', 'date': 'datetime'}
    """
    column_types = {}
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = 'datetime'
        else:
            column_types[col] = 'categorical'
    
    return column_types


def check_missing_values(df: pd.DataFrame) -> Dict[str, Dict[str, any]]:
    """
    Report missing value statistics for each column.
    
    Calculates and reports how many values are missing in each column
    and what percentage they represent. This is purely REPORTING—we do
    not fill, drop, or modify any data.
    
    Args:
        df: Input DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to their missing value statistics:
            - 'count': Number of missing values
            - 'percent': Percentage of missing values (0-100)
        Only columns with missing values are included.
        
    Example:
        >>> df = pd.DataFrame({
        ...     'a': [1, 2, np.nan, 4],
        ...     'b': [5, 6, 7, 8],
        ...     'c': [np.nan, np.nan, 9, 10]
        ... })
        >>> check_missing_values(df)
        {
            'a': {'count': 1, 'percent': 25.0},
            'c': {'count': 2, 'percent': 50.0}
        }
    """
    missing_stats = {}
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_stats[col] = {
                'count': int(missing_count),
                'percent': round((missing_count / len(df)) * 100, 2)
            }
    
    return missing_stats


def get_dataframe_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Generate a high-level summary of DataFrame characteristics.
    
    Provides quick overview statistics useful for understanding the
    structure and content of your data. This is purely informational
    reporting—no data modification.
    
    Args:
        df: Input DataFrame to analyze
        
    Returns:
        Dictionary containing:
            - 'n_rows': Number of rows
            - 'n_cols': Number of columns
            - 'n_numeric': Number of numeric columns
            - 'n_categorical': Number of categorical columns
            - 'n_datetime': Number of datetime columns
            - 'total_missing': Total count of missing values across entire DataFrame
            - 'missing_percent': Overall percentage of missing values
            - 'memory_usage_mb': Approximate memory usage in MB
            
    Example:
        >>> df = pd.DataFrame({
        ...     'a': [1, 2, np.nan],
        ...     'b': ['x', 'y', 'z']
        ... })
        >>> summary = get_dataframe_summary(df)
        >>> summary['n_rows']
        3
        >>> summary['n_numeric']
        1
        >>> summary['total_missing']
        1
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    
    return {
        'n_rows': df.shape[0],
        'n_cols': df.shape[1],
        'n_numeric': len(get_numeric_columns(df)),
        'n_categorical': len(get_categorical_columns(df)),
        'n_datetime': len(get_datetime_columns(df)),
        'total_missing': int(total_missing),
        'missing_percent': round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024**2), 2)
    }


def get_column_info(df: pd.DataFrame, col: str) -> Dict[str, any]:
    """
    Get detailed information about a specific column.
    
    Provides comprehensive metadata and statistics about a single column
    to help you understand what that column contains. This is purely
    informational analysis—no modification to the data.
    
    Args:
        df: Input DataFrame
        col: Column name to analyze
        
    Returns:
        Dictionary containing:
            - 'dtype': Data type of the column
            - 'type_category': 'numeric', 'categorical', or 'datetime'
            - 'n_unique': Number of unique values
            - 'n_missing': Number of missing values
            - 'missing_percent': Percentage missing (0-100)
            - 'is_constant': True if all non-null values are identical
            
    Raises:
        KeyError: If column doesn't exist in DataFrame
        
    Example:
        >>> df = pd.DataFrame({'age': [25, 30, 25, np.nan]})
        >>> info = get_column_info(df, 'age')
        >>> info['n_unique']
        2
        >>> info['type_category']
        'numeric'
        >>> info['missing_percent']
        25.0
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    
    col_data = df[col]
    n_missing = col_data.isnull().sum()
    n_unique = col_data.nunique()
    
    # Determine type category
    if pd.api.types.is_numeric_dtype(col_data):
        type_category = 'numeric'
    elif pd.api.types.is_datetime64_any_dtype(col_data):
        type_category = 'datetime'
    else:
        type_category = 'categorical'
    
    # Check if constant (all non-null values are same)
    is_constant = n_unique <= 1
    
    return {
        'dtype': str(col_data.dtype),
        'type_category': type_category,
        'n_unique': int(n_unique),
        'n_missing': int(n_missing),
        'missing_percent': round((n_missing / len(df)) * 100, 2),
        'is_constant': is_constant
    }


def validate_dataframe(
    df: pd.DataFrame,
    min_rows: int = 1,
    min_cols: int = 1,
    required_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate that a DataFrame meets minimum requirements for analysis.
    
    Checks if your DataFrame is valid for analysis—has enough rows,
    columns, and required columns. This is purely validation checking
    to ensure data quality for analysis—no modification of data.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum required number of rows (default: 1)
        min_cols: Minimum required number of columns (default: 1)
        required_columns: List of column names that must exist (optional)
        
    Returns:
        True if all validations pass
        
    Raises:
        ValueError: If DataFrame is None, empty, or doesn't meet minimum 
                   row/column requirements
        KeyError: If required columns are missing
        
    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> validate_dataframe(df, min_rows=2, required_columns=['a', 'b'])
        True
        
        >>> validate_dataframe(df, min_rows=5)
        ValueError: DataFrame has 2 rows, minimum required: 5
    """
    if df is None:
        raise ValueError("DataFrame is None")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) < min_rows:
        raise ValueError(
            f"DataFrame has {len(df)} rows, minimum required: {min_rows}"
        )
    
    if len(df.columns) < min_cols:
        raise ValueError(
            f"DataFrame has {len(df.columns)} columns, minimum required: {min_cols}"
        )
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise KeyError(
                f"Required columns missing from DataFrame: {sorted(missing_cols)}"
            )
    
    return True
