import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

# Import from utils
from utils.constants import (
    LOGGING_FORMAT,
    LOGGING_LEVEL,
    JSON_INDENT,
    FIGURE_SIZE,
    FIGURE_DPI,
    DEFAULT_OUTPUT_DIR
)
from utils.data_utils import (
    get_numeric_columns,
    get_categorical_columns,
    check_missing_values
)


# Configure logging using standardized format from utils
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format=LOGGING_FORMAT
)
logger = logging.getLogger(__name__)


class DistributionAnalyzer:
    """Analyze and visualize variable distributions, without altering data."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        """
        Initialize the distribution analyzer.
        
        Args:
            df: pandas DataFrame to analyze
            name: descriptive name for the dataset
        """
        self.df = df
        self.name = name
        
        # Use utility functions for column type detection
        self.numeric = get_numeric_columns(df)
        self.categorical = get_categorical_columns(df)
        
        # Use constant for output directory
        self.output_dir = Path(DEFAULT_OUTPUT_DIR) / "distributions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary = {}
        
        logger.info("Initialized DistributionAnalyzer for %s", name)

    def summarize(self) -> Dict[str, Any]:
        """
        Generate summary statistics for numeric and categorical variables.
        
        Uses utils.data_utils.check_missing_values for consistent missing
        value reporting.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {"numeric": {}, "categorical": {}}
        
        # Get missing values using utility function
        missing_stats = check_missing_values(self.df)

        # Numeric summaries
        for col in self.numeric:
            summary["numeric"][col] = self.df[col].describe(include='all').to_dict()
            
            # Use utility function result for missing values
            if col in missing_stats:
                summary["numeric"][col]["missing"] = missing_stats[col]['count']
                summary["numeric"][col]["missing_percent"] = missing_stats[col]['percent']
            else:
                summary["numeric"][col]["missing"] = 0
                summary["numeric"][col]["missing_percent"] = 0.0

        # Categorical summaries
        for col in self.categorical:
            vc = self.df[col].value_counts(dropna=False)
            summary["categorical"][col] = {
                "frequency": vc.to_dict(),
                "top": vc.index[0] if not vc.empty else None,
                "top_count": int(vc.iloc[0]) if not vc.empty else 0
            }
            
            # Use utility function result for missing values
            if col in missing_stats:
                summary["categorical"][col]["missing"] = missing_stats[col]['count']
                summary["categorical"][col]["missing_percent"] = missing_stats[col]['percent']
            else:
                summary["categorical"][col]["missing"] = 0
                summary["categorical"][col]["missing_percent"] = 0.0

        self.summary = summary
        logger.info("Generated summary for %s numeric and %s categorical columns", len(self.numeric), len(self.categorical))
        return summary

    def visualize(self):
        """
        Generate visualizations for all variables.
        
        Uses FIGURE_SIZE and FIGURE_DPI constants for consistent styling.
        """
        # Numeric: histogram, kde, boxplot
        for col in self.numeric:
            # Histogram with KDE
            plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
            sns.histplot(self.df[col], kde=True, bins="auto", color="skyblue")
            plt.title(f"{col} - Histogram & Density")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{col}_hist_density.png")
            plt.close()

            # Boxplot
            plt.figure(figsize=(7, 2), dpi=FIGURE_DPI)
            sns.boxplot(x=self.df[col], color="orange")
            plt.title(f"{col} - Boxplot")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{col}_boxplot.png")
            plt.close()

        # Categorical: bar plot
        for col in self.categorical:
            plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
            freq = self.df[col].value_counts(dropna=False)
            freq.plot(kind="bar", color="lightgreen")
            plt.title(f"{col} - Frequency Count")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{col}_barplot.png")
            plt.close()

        logger.info("Generated visualizations for %s columns", len(self.numeric) + len(self.categorical))

    def export_summary(self, file_path: str):
        """
        Export summary statistics to JSON file.
        
        Uses JSON_INDENT constant for consistent formatting.
        
        Args:
            file_path: relative path for output file
        """
        file = Path(file_path)
        file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file, "w") as f:
            json.dump(self.summary, f, indent=JSON_INDENT, default=str)
        
        logger.info("Summary exported to %s", file_path)


def analyze_distributions(input_file="data/dataset.csv"):
    """
    Main entry point for distribution analysis.
    
    Orchestrates the complete distribution analysis workflow.
    
    Args:
        input_file: relative path to input CSV file
    """
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        analyzer = DistributionAnalyzer(df)
        summary = analyzer.summarize()
        analyzer.visualize()
        analyzer.export_summary("output/distributions/summary.json")
        
        print("Summary statistics and distribution plots generated in output/distributions/")
        
        # Print quick stats to console as a sample
        print(json.dumps(summary, indent=JSON_INDENT))
        
    except Exception as e:
        logger.error(f"Distribution analysis failed: {e}")
        raise


if __name__ == "__main__":
    analyze_distributions()
