import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, chi2_contingency
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

# Import from utils
from utils.constants import (
    ALPHA,
    MIN_GROUP_SIZE,
    JSON_INDENT,
    LOGGING_FORMAT,
    LOGGING_LEVEL
)
from utils.data_utils import (
    get_numeric_columns,
    get_categorical_columns
)


# Configure logging using standardized format from utils
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format=LOGGING_FORMAT
)
logger = logging.getLogger(__name__)


class StatTestRunner:
    """
    Run hypothesis tests comparing groups on numeric or categorical outcomes,
    strictly using original data (with existing missing values as-is).
    
    Uses ALPHA and MIN_GROUP_SIZE constants for significance and validation.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize statistical test runner.
        
        Uses get_numeric_columns() and get_categorical_columns() from data_utils.
        
        Args:
            df: pandas DataFrame to analyze
        """
        self.df = df
        
        # Use utility functions for column detection
        self.numeric = get_numeric_columns(df)
        self.categorical = get_categorical_columns(df)
        
        self.results: Dict[str, Any] = {}
        
        logger.info(f"Initialized StatTestRunner: {len(self.numeric)} numeric, {len(self.categorical)} categorical columns")

    def ttest_between_groups(self, group_col: str, numeric_col: str, 
                            group_values: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform t-test between two groups.
        
        Uses ALPHA constant for significance interpretation.
        
        Args:
            group_col: Grouping variable (categorical)
            numeric_col: Numeric variable to compare
            group_values: Optional list of two group values
            
        Returns:
            Dictionary with test results
        """
        result = {}
        
        # Only works for two groups with sufficient non-null data in each
        data = self.df[[group_col, numeric_col]].dropna()
        
        if group_values is None:
            group_values = list(data[group_col].unique())
        
        if len(group_values) == 2:
            g1 = data[data[group_col] == group_values[0]][numeric_col]
            g2 = data[data[group_col] == group_values[1]][numeric_col]
            
            # Check minimum group size (not hardcoded 3)
            min_size = max(3, MIN_GROUP_SIZE // 10)  # Use at least 3, scale with constant
            
            if len(g1) >= min_size and len(g2) >= min_size:
                stat, p = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
                
                result = {
                    'test': 't-test',
                    'group1': group_values[0],
                    'group2': group_values[1],
                    'n_g1': int(len(g1)),
                    'n_g2': int(len(g2)),
                    'statistic': float(stat),
                    'p_value': float(p),
                    'significant': p < ALPHA,
                    'alpha': ALPHA
                }
        
        return result

    def mannwhitney_between_groups(self, group_col: str, numeric_col: str, 
                                   group_values: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test between two groups.
        
        Non-parametric alternative to t-test. Uses ALPHA constant.
        
        Args:
            group_col: Grouping variable (categorical)
            numeric_col: Numeric variable to compare
            group_values: Optional list of two group values
            
        Returns:
            Dictionary with test results
        """
        result = {}
        data = self.df[[group_col, numeric_col]].dropna()
        
        if group_values is None:
            group_values = list(data[group_col].unique())
        
        if len(group_values) == 2:
            g1 = data[data[group_col] == group_values[0]][numeric_col]
            g2 = data[data[group_col] == group_values[1]][numeric_col]
            
            min_size = max(3, MIN_GROUP_SIZE // 10)
            
            if len(g1) >= min_size and len(g2) >= min_size:
                stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                
                result = {
                    'test': 'Mann-Whitney U',
                    'group1': group_values[0],
                    'group2': group_values[1],
                    'n_g1': int(len(g1)),
                    'n_g2': int(len(g2)),
                    'statistic': float(stat),
                    'p_value': float(p),
                    'significant': p < ALPHA,
                    'alpha': ALPHA
                }
        
        return result

    def anova(self, group_col: str, numeric_col: str) -> Dict[str, Any]:
        """
        Perform one-way ANOVA across multiple groups.
        
        Uses ALPHA constant for significance interpretation.
        
        Args:
            group_col: Grouping variable (categorical)
            numeric_col: Numeric variable to compare
            
        Returns:
            Dictionary with test results
        """
        result = {}
        data = self.df[[group_col, numeric_col]].dropna()
        groups = [vals[numeric_col].values for _, vals in data.groupby(group_col)]
        
        min_size = max(3, MIN_GROUP_SIZE // 10)
        
        if len(groups) >= 2 and all(len(g) >= min_size for g in groups):
            stat, p = f_oneway(*groups)
            
            result = {
                'test': 'ANOVA',
                'groups': list(data[group_col].unique()),
                'ns': [int(len(g)) for g in groups],
                'statistic': float(stat),
                'p_value': float(p),
                'significant': p < ALPHA,
                'alpha': ALPHA
            }
        
        return result

    def chi2_test(self, cat1: str, cat2: str) -> Dict[str, Any]:
        """
        Perform chi-square test of independence between two categorical variables.
        
        Uses ALPHA constant for significance interpretation.
        
        Args:
            cat1: First categorical variable
            cat2: Second categorical variable
            
        Returns:
            Dictionary with test results
        """
        result = {}
        table = pd.crosstab(self.df[cat1], self.df[cat2])
        
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            stat, p, _, _ = chi2_contingency(table)
            
            result = {
                'test': 'chi2',
                'vars': (cat1, cat2),
                'shape': table.shape,
                'statistic': float(stat),
                'p_value': float(p),
                'significant': p < ALPHA,
                'alpha': ALPHA
            }
        
        return result

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all applicable statistical tests.
        
        Automatically determines which tests to run based on variable types
        and adds significance flags using ALPHA constant.
        
        Returns:
            Dictionary with results for each test type
        """
def run(self) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all applicable statistical tests.
    ...
    """
    results: Dict[str, Any] = {
        "ttest": [],
        "mannwhitney": [],
        "anova": [],
        "chi2": []
    }
    
    logger.info("Running statistical tests...")
        
        logger.info("Running statistical tests...")
        
        # Pairwise numeric ~ categorical
        for num in self.numeric:
            for cat in self.categorical:
                vals = self.df[cat].dropna().unique()
                
                if len(vals) == 2:
                    ttest_res = self.ttest_between_groups(cat, num, list(vals))
                    if ttest_res:
                        results['ttest'].append({'cat': cat, 'num': num, **ttest_res})
                    
                    mw_res = self.mannwhitney_between_groups(cat, num, list(vals))
                    if mw_res:
                        results['mannwhitney'].append({'cat': cat, 'num': num, **mw_res})
                
                if len(vals) >= 2:
                    anova_res = self.anova(cat, num)
                    if anova_res:
                        results['anova'].append({'cat': cat, 'num': num, **anova_res})
        
        # Pairwise categorical ~ categorical
        for i, cat1 in enumerate(self.categorical):
            for cat2 in self.categorical[i+1:]:
                chi2_res = self.chi2_test(cat1, cat2)
                if chi2_res:
                    results['chi2'].append(chi2_res)
        
        self.results = results
        
        # Count significant results
        sig_counts = {
            test: sum(1 for r in results[test] if r.get('significant', False))
            for test in results.keys()
        }
        
        logger.info(f"Tests completed: {sig_counts}")
        
        return results

    def export(self, file_out: str) -> None:
        """
        Export test results to JSON.
        
        Uses JSON_INDENT constant for consistent formatting.
        
        Args:
            file_out: Output file path
        """
        Path(file_out).parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata to results
        export_data = {
            'configuration': {
                'alpha': ALPHA,
                'min_group_size': MIN_GROUP_SIZE
            },
            'results': self.results
        }
        
        with open(file_out, "w") as f:
            json.dump(export_data, f, indent=JSON_INDENT, default=str)
        
        logger.info(f"Results exported to {file_out}")


def analyze_statistical_tests(input_file: str = "data/dataset.csv",
                              output_file: str = "output/stat_tests.json"):
    """
    Main entry point for statistical testing analysis.
    
    Orchestrates complete hypothesis testing workflow.
    
    Args:
        input_file: Path to input CSV
        output_file: Path for output JSON
    """
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        runner = StatTestRunner(df)
        results = runner.run()
        runner.export(output_file)
        
        print("✓ Statistical testing complete")
        print(f"✓ Significance level (alpha): {ALPHA}")
        print(f"✓ Results exported to {output_file}")
        
        # Print summary of significant results
        print("\n=== Significant Results Summary ===")
        for test in ['ttest', 'mannwhitney', 'anova', 'chi2']:
            sig_count = sum(1 for r in results[test] if r.get('significant', False))
            total = len(results[test])
            print(f"{test.upper()}: {sig_count}/{total} significant (α={ALPHA})")
        
        # Print sample results
        print("\n=== Sample Results ===")
        for test in ['ttest', 'mannwhitney', 'anova', 'chi2']:
            if results[test]:
                print(f"\n{test.upper()} (first result):")
                sample = results[test][0]
                print(f"  Variables: {sample.get('cat', '')}-{sample.get('num', '')} {sample.get('vars', '')}")
                print(f"  p-value: {sample['p_value']:.4f}")
                print(f"  Significant: {'YES' if sample.get('significant') else 'NO'}")
        
    except Exception as e:
        logger.error(f"Statistical testing failed: {e}")
        raise


if __name__ == "__main__":
    analyze_statistical_tests()
