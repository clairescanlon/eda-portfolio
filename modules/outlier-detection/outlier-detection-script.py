import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import logging
import json
from collections import Counter

# Import from utils
from utils.constants import (
    ZSCORE_THRESHOLD,
    MAD_THRESHOLD,
    IQR_MULTIPLIER,
    JSON_INDENT,
    LOGGING_FORMAT,
    LOGGING_LEVEL,
    DEFAULT_OUTPUT_DIR
)
from utils.data_utils import (
    get_numeric_columns,
    check_missing_values
)


# Configure logging using standardized format from utils
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format=LOGGING_FORMAT
)
logger = logging.getLogger(__name__)


class OutlierDetector(ABC):
    """Base class for outlier detection methods."""

    @abstractmethod
    def detect(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers in a series."""
        pass


class IQRDetector(OutlierDetector):
    """
    Detects outliers using Interquartile Range (IQR) method.
    Values beyond Q1 - IQR_MULTIPLIER*IQR or Q3 + IQR_MULTIPLIER*IQR are flagged.
    Most robust for distributions with extreme values.
    """

    def __init__(self, multiplier: float = IQR_MULTIPLIER):
        """
        Initialize IQR detector.
        
        Args:
            multiplier: IQR multiplier (from constants, default 1.5)
        """
        self.multiplier = multiplier

    def detect(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect outliers using IQR method.
        
        Uses IQR_MULTIPLIER constant for threshold calculation.
        
        Args:
            series: numeric pandas Series
            
        Returns:
            Dictionary with outlier indices and statistics
        """
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {
                "method": "IQR",
                "multiplier": self.multiplier,
                "outlier_indices": [],
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "bounds": {"lower": None, "upper": None}
            }
        
        q1 = series_clean.quantile(0.25)
        q3 = series_clean.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr
        
        outlier_mask = (series_clean < lower_bound) | (series_clean > upper_bound)
        outlier_indices = series_clean[outlier_mask].index.tolist()
        
        return {
            "method": "IQR",
            "multiplier": self.multiplier,
            "outlier_indices": outlier_indices,
            "outlier_count": len(outlier_indices),
            "outlier_percentage": (len(outlier_indices) / len(series_clean)) * 100,
            "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr)
        }


class ZScoreDetector(OutlierDetector):
    """
    Detects outliers using Z-Score method.
    Values with |z-score| > threshold are considered outliers.
    Best for normally distributed data.
    """

    def __init__(self, threshold: float = ZSCORE_THRESHOLD):
        """
        Initialize Z-Score detector.
        
        Args:
            threshold: Z-score threshold (from constants, default 3.0)
        """
        self.threshold = threshold

    def detect(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect outliers using Z-Score method.
        
        Uses ZSCORE_THRESHOLD constant.
        
        Args:
            series: numeric pandas Series
            
        Returns:
            Dictionary with outlier indices and statistics
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 2:
            return {
                "method": "ZScore",
                "threshold": self.threshold,
                "outlier_indices": [],
                "outlier_count": 0,
                "outlier_percentage": 0.0
            }
        
        mean = series_clean.mean()
        std = series_clean.std()
        
        if std == 0:
            return {
                "method": "ZScore",
                "threshold": self.threshold,
                "outlier_indices": [],
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "note": "No variance in data (all values identical)"
            }
        
        z_scores = np.abs((series_clean - mean) / std)
        outlier_mask = z_scores > self.threshold
        outlier_indices = series_clean[outlier_mask].index.tolist()
        
        return {
            "method": "ZScore",
            "threshold": self.threshold,
            "outlier_indices": outlier_indices,
            "outlier_count": len(outlier_indices),
            "outlier_percentage": (len(outlier_indices) / len(series_clean)) * 100,
            "mean": float(mean),
            "std": float(std)
        }


class MADDetector(OutlierDetector):
    """
    Detects outliers using Median Absolute Deviation (MAD) method.
    More robust than Z-Score for non-normal distributions.
    Modified Z-Score > threshold indicates outlier.
    """

    def __init__(self, threshold: float = MAD_THRESHOLD):
        """
        Initialize MAD detector.
        
        Args:
            threshold: MAD threshold (from constants, default 3.5)
        """
        self.threshold = threshold

    def detect(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect outliers using MAD method.
        
        Uses MAD_THRESHOLD constant.
        
        Args:
            series: numeric pandas Series
            
        Returns:
            Dictionary with outlier indices and statistics
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 2:
            return {
                "method": "MAD",
                "threshold": self.threshold,
                "outlier_indices": [],
                "outlier_count": 0,
                "outlier_percentage": 0.0
            }
        
        median = series_clean.median()
        mad = np.median(np.abs(series_clean - median))
        
        if mad == 0:
            return {
                "method": "MAD",
                "threshold": self.threshold,
                "outlier_indices": [],
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "note": "No deviation from median (insufficient variation)"
            }
        
        modified_z_scores = 0.6745 * (series_clean - median) / mad
        outlier_mask = np.abs(modified_z_scores) > self.threshold
        outlier_indices = series_clean[outlier_mask].index.tolist()
        
        return {
            "method": "MAD",
            "threshold": self.threshold,
            "outlier_indices": outlier_indices,
            "outlier_count": len(outlier_indices),
            "outlier_percentage": (len(outlier_indices) / len(series_clean)) * 100,
            "median": float(median),
            "mad": float(mad)
        }


class OutlierAnalyzer:
    """
    Orchestrates outlier detection across multiple columns and methods.
    Applies multiple detection algorithms for comprehensive analysis.
    """

    def __init__(self, data: pd.DataFrame, name: str = "dataset"):
        """
        Initialize outlier analyzer.
        
        Uses get_numeric_columns() from data_utils.
        
        Args:
            data: pandas DataFrame to analyze
            name: descriptive name for dataset
        """
        self.data = data
        self.name = name
        self.report = {}
        
        # Use utility function for column detection
        self.numeric_columns = get_numeric_columns(data)
        
        logger.info(f"Initialized OutlierAnalyzer for {name}")
        logger.info(f"Identified {len(self.numeric_columns)} numeric columns")

    def detect_outliers(self, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers using specified methods.
        
        Args:
            methods: List of methods ('iqr', 'zscore', 'mad')
                    Defaults to all methods
                    
        Returns:
            Dictionary with outlier detection results
        """
        methods = methods or ['iqr', 'zscore', 'mad']
        results = {}
        
        for col in self.numeric_columns:
            logger.info(f"Analyzing column: {col}")
            results[col] = self._analyze_column(col, methods)
        
        self.report["outliers"] = results
        return results

    def _analyze_column(self, col: str, methods: List[str]) -> Dict[str, Any]:
        """
        Apply detection methods to a single column.
        
        Args:
            col: column name
            methods: list of detection methods
            
        Returns:
            Detection results for the column
        """
        series = self.data[col]
        
        col_report = {
            "column": col,
            "dtype": str(series.dtype),
            "total_values": len(series),
            "null_count": int(series.isnull().sum()),
            "min": float(series.min()) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
            "mean": float(series.mean()) if not series.empty else None,
            "median": float(series.median()) if not series.empty else None,
            "std": float(series.std()) if not series.empty else None,
            "detectors": {}
        }
        
        # Initialize detectors with constants
        detectors = {
            'iqr': IQRDetector(multiplier=IQR_MULTIPLIER),
            'zscore': ZScoreDetector(threshold=ZSCORE_THRESHOLD),
            'mad': MADDetector(threshold=MAD_THRESHOLD)
        }
        
        for method in methods:
            if method in detectors:
                detector = detectors[method]
                col_report["detectors"][method] = detector.detect(series)
        
        return col_report

    def consensus_outliers(self) -> Dict[str, Any]:
        """
        Identify values flagged by multiple detection methods.
        Consensus outliers (flagged by 2+ methods) are more reliable.
        
        Returns:
            Dictionary with consensus outlier information
        """
        consensus = {}
        
        for col, col_data in self.report.get("outliers", {}).items():
            all_outlier_indices = []
            
            for detector, results in col_data.get("detectors", {}).items():
                all_outlier_indices.extend(results.get("outlier_indices", []))
            
            # Count occurrences
            counts = Counter(all_outlier_indices)
            
            # Consensus: flagged by 2+ methods
            consensus_indices = [idx for idx, count in counts.items() if count >= 2]
            
            if consensus_indices:
                consensus[col] = {
                    "consensus_outlier_indices": consensus_indices,
                    "consensus_count": len(consensus_indices),
                    "consensus_values": self.data.loc[consensus_indices, col].tolist()
                }
        
        self.report["consensus"] = consensus
        logger.info(f"Identified consensus outliers in {len(consensus)} columns")
        return consensus

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics for outlier analysis.
        
        Returns:
            Dictionary with analysis summary
        """
        summary = {
            "dataset_name": self.name,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "numeric_columns_analyzed": len(self.numeric_columns),
            "total_outliers_by_method": {},
            "thresholds_used": {
                "iqr_multiplier": IQR_MULTIPLIER,
                "zscore_threshold": ZSCORE_THRESHOLD,
                "mad_threshold": MAD_THRESHOLD
            }
        }
        
        for col, col_data in self.report.get("outliers", {}).items():
            for method, results in col_data.get("detectors", {}).items():
                if method not in summary["total_outliers_by_method"]:
                    summary["total_outliers_by_method"][method] = 0
                summary["total_outliers_by_method"][method] += results.get("outlier_count", 0)
        
        summary["consensus_outliers_found"] = len(self.report.get("consensus", {}))
        
        self.report["summary"] = summary
        logger.info(f"Generated summary: {summary['total_outliers_by_method']}")
        return summary

    def export_report(self, filepath: str) -> None:
        """
        Export outlier analysis report to JSON.
        
        Uses JSON_INDENT constant for consistent formatting.
        
        Args:
            filepath: relative path for output
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=JSON_INDENT, default=str)
        
        logger.info(f"Outlier report exported to {filepath}")


def analyze_outliers(input_file: str, output_file: str,
                    methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main entry point for outlier detection analysis.
    
    Orchestrates complete outlier detection workflow.
    
    Args:
        input_file: relative path to input CSV
        output_file: relative path for output report
        methods: detection methods to apply
        
    Returns:
        Complete outlier analysis report
    """
    try:
        data_path = Path(input_file)
        
        if not data_path.exists():
            logger.error(f"Input file not found: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Loading data from {input_file}")
        data = pd.read_csv(data_path)
        
        analyzer = OutlierAnalyzer(data, name=data_path.stem)
        analyzer.detect_outliers(methods=methods)
        analyzer.consensus_outliers()
        analyzer.generate_summary()
        
        report = analyzer.report
        analyzer.export_report(output_file)
        
        return report
        
    except Exception as e:
        logger.error(f"Error during outlier analysis: {str(e)}")
        raise


if __name__ == "__main__":
    # Sample data for demonstration
    sample_data = {
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "transaction_amount": [100, 150, 120, 5000, 110, 125, 130, 999, 115, 105],
        "account_age_days": [30, 45, 60, 75, 90, 180, 365, 500, 50, 100],
        "login_attempts": [1, 2, 1, 50, 1, 2, 3, 1, 2, 1]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("data/transactions.csv", index=False)
    
    report = analyze_outliers(
        input_file="data/transactions.csv",
        output_file="output/outlier_report.json",
        methods=['iqr', 'zscore', 'mad']
    )
    
    print("\n=== Outlier Analysis Report ===")
    print(f"Dataset: {report['summary']['dataset_name']}")
    print(f"Columns Analyzed: {report['summary']['numeric_columns_analyzed']}")
    print(f"Total Outliers by Method: {report['summary']['total_outliers_by_method']}")
    print(f"Consensus Outliers Found: {report['summary']['consensus_outliers_found']}")
    print(f"\nThresholds Used:")
    print(f"  IQR Multiplier: {report['summary']['thresholds_used']['iqr_multiplier']}")
    print(f"  Z-Score Threshold: {report['summary']['thresholds_used']['zscore_threshold']}")
    print(f"  MAD Threshold: {report['summary']['thresholds_used']['mad_threshold']}")
