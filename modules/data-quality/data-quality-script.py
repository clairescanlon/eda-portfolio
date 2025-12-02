import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import logging
from enum import Enum
import json

from utils.constants import (
    MIN_UNIQUE_RATIO,
    MAX_UNIQUE_RATIO,
    JSON_INDENT,
    LOGGING_FORMAT,
    LOGGING_LEVEL
)

from utils.data_utils import (
    get_categorical_columns,
    check_missing_values,
    get_dataframe_summary,
    detect_column_types
)


# Configure logging using standardized format from utils
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format=LOGGING_FORMAT
)
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Enum for validation status outcomes."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class DataQualityAnalyzer:
    """
    Analyzes data quality across multiple dimensions.
    
    Follows Single Responsibility Principle by handling all quality checks
    in a modular, reusable manner. Integrates completeness, validity,
    consistency, and accuracy checks.
    """

    def __init__(self, data: pd.DataFrame, name: str = "dataset"):
        """
        Initialize the analyzer with data and configuration.
        
        Args:
            data: pandas DataFrame to analyze
            name: descriptive name for the dataset
        """
        self.data = data
        self.name = name
        self.report = {}
        logger.info("Initialized DataQualityAnalyzer for %s", name)

    def validate_field(self, field_name: str, field_type: str,
                       required: bool = True) -> Dict[str, Any]:
        """
        Generalized field validation using parameterized approach.
        
        Validates a single field against type and completeness criteria.
        Replaces multiple specific validators with one flexible function.
        
        Args:
            field_name: column name to validate
            field_type: expected data type ('int', 'float', 'str', 'datetime')
            required: whether null values are acceptable
            
        Returns:
            Dictionary containing validation results
        """
        result = {
            "field": field_name,
            "type_expected": field_type,
            "status": ValidationStatus.PASS.value,
            "issues": []
        }

        if field_name not in self.data.columns:
            result["status"] = ValidationStatus.FAIL.value
            result["issues"].append(f"Column '{field_name}' not found")
            return result

        column = self.data[field_name]

        # Check completeness
        null_count = column.isnull().sum()
        null_pct = (null_count / len(self.data)) * 100

        if required and null_count > 0:
            result["status"] = ValidationStatus.WARN.value
            result["issues"].append(
                f"Missing values: {null_count} ({null_pct:.2f}%)"
            )

        # Type validation (skip nulls for type checking)
        non_null_data = column.dropna()
        if len(non_null_data) > 0:
            type_mismatch = self._check_type_compliance(non_null_data, field_type)
            if type_mismatch > 0:
                result["status"] = ValidationStatus.WARN.value
                result["issues"].append(
                    f"Type mismatch: {type_mismatch} values"
                )

        result["null_count"] = null_count
        result["null_percentage"] = null_pct
        result["non_null_count"] = len(non_null_data)

        return result

    def _check_type_compliance(self, series: pd.Series, expected_type: str) -> int:
        """
        Check type compliance for a series against expected type.
        
        Args:
            series: pandas Series to check
            expected_type: expected type string
            
        Returns:
            Count of non-compliant values
        """
        mismatch_count = 0

        if expected_type == 'int':
            mismatch_count = sum(1 for x in series if not isinstance(x, (int, np.integer)))
        elif expected_type == 'float':
            mismatch_count = sum(1 for x in series if not isinstance(x, (float, np.floating, int, np.integer)))
        elif expected_type == 'str':
            mismatch_count = sum(1 for x in series if not isinstance(x, str))
        elif expected_type == 'datetime':
            try:
                pd.to_datetime(series)
            except Exception:
                mismatch_count = len(series)

        return mismatch_count

    def assess_completeness(self) -> Dict[str, Any]:
        """
        Assess data completeness across all fields.
        
        Uses utils.data_utils.check_missing_values for consistent reporting.
        
        Returns:
            Dictionary with completeness metrics
        """
        # Use utility function for missing value analysis
        missing_stats = check_missing_values(self.data)
        
        completeness = {}
        for column in self.data.columns:
            if column in missing_stats:
                null_count = missing_stats[column]['count']
                null_pct = missing_stats[column]['percent']
            else:
                null_count = 0
                null_pct = 0.0
            
            completeness[column] = {
                "null_count": null_count,
                "null_percentage": null_pct,
                "complete_count": len(self.data) - null_count
            }

        overall_null_pct = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        completeness["overall_completeness"] = 100 - overall_null_pct

        self.report["completeness"] = completeness
        return completeness

    def assess_consistency(self) -> Dict[str, Any]:
        """
        Assess consistency of data format and values across fields.
        
        Returns:
            Dictionary with consistency metrics
        """
        consistency = {}

        for column in self.data.columns:
            unique_count = self.data[column].nunique()
            duplicate_count = len(self.data) - unique_count
            consistency[column] = {
                "unique_values": unique_count,
                "duplicate_count": duplicate_count,
                "duplicate_percentage": (duplicate_count / len(self.data)) * 100
            }

        self.report["consistency"] = consistency
        return consistency

    def assess_validity(self, rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess data validity against defined rules and standards.
        
        Args:
            rules: Dictionary defining validation rules per field
                   Example: {"age": {"min": 0, "max": 150}}
                   
        Returns:
            Dictionary with validity assessment
        """
        validity = {}
        rules = rules or {}

        for column in self.data.columns:
            validity[column] = {"valid_count": len(self.data), "invalid_count": 0, "issues": []}

            if column in rules:
                rule = rules[column]

                if "min" in rule:
                    invalid = (self.data[column] < rule["min"]).sum()
                    validity[column]["invalid_count"] += invalid
                    if invalid > 0:
                        validity[column]["issues"].append(
                            f"{invalid} values below minimum {rule['min']}"
                        )

                if "max" in rule:
                    invalid = (self.data[column] > rule["max"]).sum()
                    validity[column]["invalid_count"] += invalid
                    if invalid > 0:
                        validity[column]["issues"].append(
                            f"{invalid} values above maximum {rule['max']}"
                        )

        self.report["validity"] = validity
        return validity

    def assess_data_summary(self) -> Dict[str, Any]:
        """
        Generate high-level data summary using utils.
        
        Returns:
            Dictionary with data summary metrics
        """
        summary = get_dataframe_summary(self.data)
        column_types = detect_column_types(self.data)
        
        summary['column_types'] = column_types
        self.report["data_summary"] = summary
        return summary

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Returns:
            Complete quality assessment report
        """
        self.report["dataset_name"] = self.name
        self.report["total_rows"] = len(self.data)
        self.report["total_columns"] = len(self.data.columns)
        self.report["timestamp"] = pd.Timestamp.now().isoformat()
        
        # Add data summary from utils
        self.assess_data_summary()

        logger.info(f"Generated quality report for {self.name}")
        return self.report

    def export_report(self, filepath: str) -> None:
        """
        Export quality report to JSON file.
        
        Uses JSON_INDENT constant for consistent formatting.
        
        Args:
            filepath: relative path for output (security best practice)
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=JSON_INDENT, default=str)

        logger.info("Report exported to %s", filepath)


def analyze_data_quality(input_file: str, output_file: str,
                        validation_rules: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main entry point for data quality analysis.
    
    Orchestrates the complete quality analysis workflow.
    
    Args:
        input_file: relative path to input CSV (security best practice)
        output_file: relative path for output report
        validation_rules: optional rules for field validation
        
    Returns:
        Complete quality assessment report
    """
    try:
        # Use relative paths for security and portability
        data_path = Path(input_file)

        if not data_path.exists():
            logger.error("Input file not found: %s", input_file)
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Load data
        logger.info(f"Loading data from {input_file}")
        data = pd.read_csv(data_path)

        # Initialize analyzer
        analyzer = DataQualityAnalyzer(data, name=data_path.stem)

        # Execute quality checks
        analyzer.assess_completeness()
        analyzer.assess_consistency()
        analyzer.assess_validity(validation_rules)

        # Generate and export report
        report = analyzer.generate_report()
        analyzer.export_report(output_file)

        return report

    except Exception as e:
        logger.error("Error during data quality analysis: %s", str(e))
        raise


# Example usage demonstrating the analyzer
if __name__ == "__main__":
    # Sample data for demonstration
    sample_data = {
        "user_id": [1, 2, 3, 4, 5, None],
        "user_name": ["Alice", "Bob", "Charlie", "David", None, "Frank"],
        "user_email": ["alice@example.com", "bob@example.com", None, "david@example.com",
                       "eve@example.com", "frank@example.com"],
        "age": [25, 30, 28, 35, 29, 32]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv("data/sample_data.csv", index=False)

    # Define validation rules
    rules = {
        "age": {"min": 0, "max": 150}
    }

    # Run analysis
    report = analyze_data_quality(
        input_file="data/sample_data.csv",
        output_file="output/quality_report.json",
        validation_rules=rules
    )

    print("\n=== Data Quality Report ===")
    print(f"Dataset: {report['dataset_name']}")
    print(f"Total Rows: {report['total_rows']}")
    print(f"Total Columns: {report['total_columns']}")
    print(f"Overall Completeness: {report['completeness']['overall_completeness']:.2f}%")
