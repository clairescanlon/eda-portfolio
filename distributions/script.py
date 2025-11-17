import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class DistributionAnalyzer:
    """Analyze and visualize variable distributions, without altering data."""
    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.df = df
        self.name = name
        self.numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.output_dir = Path("output/distributions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary = {}

    def summarize(self) -> Dict[str, Any]:
        summary = {"numeric": {}, "categorical": {}}
        for col in self.numeric:
            summary["numeric"][col] = self.df[col].describe(include='all').to_dict()
            summary["numeric"][col]["missing"] = int(self.df[col].isnull().sum())
        for col in self.categorical:
            vc = self.df[col].value_counts(dropna=False)
            summary["categorical"][col] = {
                "frequency": vc.to_dict(),
                "top": vc.index[0] if not vc.empty else None,
                "top_count": int(vc.iloc[0]) if not vc.empty else 0,
                "missing": int(self.df[col].isnull().sum())
            }
        self.summary = summary
        return summary
    
    def visualize(self):
        # Numeric: histogram, kde, boxplot
        for col in self.numeric:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], kde=True, bins="auto", color="skyblue")
            plt.title(f"{col} - Histogram & Density")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{col}_hist_density.png")
            plt.close()
            
            plt.figure(figsize=(7, 2))
            sns.boxplot(x=self.df[col], color="orange")
            plt.title(f"{col} - Boxplot")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{col}_boxplot.png")
            plt.close()

        # Categorical: bar plot
        for col in self.categorical:
            plt.figure(figsize=(8, 4))
            freq = self.df[col].value_counts(dropna=False)
            freq.plot(kind="bar", color="lightgreen")
            plt.title(f"{col} - Frequency Count")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{col}_barplot.png")
            plt.close()

    def export_summary(self, file_path: str):
        file = Path(file_path)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as f:
            json.dump(self.summary, f, indent=2, default=str)
        logger.info(f"Summary exported to {file_path}")

def main(input_file="data/dataset.csv"):
    try:
        df = pd.read_csv(input_file)
        analyzer = DistributionAnalyzer(df)
        summary = analyzer.summarize()
        analyzer.visualize()
        analyzer.export_summary("output/distributions/summary.json")
        print("Summary statistics and distribution plots generated in output/distributions/")

        # Print quick stats to console as a sample
        print(json.dumps(summary, indent=2))
    except Exception as e:
        logger.error(f"Distribution analysis failed: {e}")

if __name__ == "__main__":
    main()
