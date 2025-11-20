"""
Utility modules for EDA analysis scripts.

This package provides shared utilities for consistent data analysis:
- constants: Configuration values and thresholds
- data_utils: Data inspection and reporting functions
- io_utils: File I/O operations (future)
- logging_config: Logging setup (future)

Philosophy: All utilities analyze and report - they never modify data.

Usage:
    from utils.constants import RANDOM_STATE, LOGGING_FORMAT
    from utils.data_utils import get_numeric_columns, check_missing_values
"""

__version__ = '0.1.0'
__all__ = [
    'constants',
    'data_utils',
]
