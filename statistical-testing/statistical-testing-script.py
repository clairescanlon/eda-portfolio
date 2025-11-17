import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, chi2_contingency
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class StatTestRunner:
    """
    Run hypothesis tests comparing groups on numeric or categorical outcomes,
    strictly using original data (with existing missing values as-is).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.results = {}

    def ttest_between_groups(self, group_col: str, numeric_col: str, group_values: Optional[List[str]] = None):
        result = {}
        # Only works for two groups with sufficient non-null data in each
        data = self.df[[group_col, numeric_col]].dropna()
        if group_values is None:
            group_values = list(data[group_col].unique())
        if len(group_values) == 2:
            g1 = data[data[group_col] == group_values[0]][numeric_col]
            g2 = data[data[group_col] == group_values[1]][numeric_col]
            if len(g1) >= 3 and len(g2) >= 3:
                stat, p = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
                result = {
                    'test': 't-test',
                    'group1': group_values[0],
                    'group2': group_values[1],
                    'n_g1': int(len(g1)),
                    'n_g2': int(len(g2)),
                    'statistic': float(stat),
                    'p_value': float(p)
                }
        return result

    def mannwhitney_between_groups(self, group_col: str, numeric_col: str, group_values: Optional[List[str]] = None):
        result = {}
        data = self.df[[group_col, numeric_col]].dropna()
        if group_values is None:
            group_values = list(data[group_col].unique())
        if len(group_values) == 2:
            g1 = data[data[group_col] == group_values[0]][numeric_col]
            g2 = data[data[group_col] == group_values[1]][numeric_col]
            if len(g1) >= 3 and len(g2) >= 3:
                stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                result = {
                    'test': 'Mann-Whitney U',
                    'group1': group_values[0],
                    'group2': group_values[1],
                    'n_g1': int(len(g1)),
                    'n_g2': int(len(g2)),
                    'statistic': float(stat),
                    'p_value': float(p)
                }
        return result

    def anova(self, group_col: str, numeric_col: str):
        result = {}
        data = self.df[[group_col, numeric_col]].dropna()
        groups = [vals[numeric_col].values for _, vals in data.groupby(group_col)]
        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            stat, p = f_oneway(*groups)
            result = {
                'test': 'ANOVA',
                'groups': list(data[group_col].unique()),
                'ns': [int(len(g)) for g in groups],
                'statistic': float(stat),
                'p_value': float(p)
            }
        return result

    def chi2_test(self, cat1: str, cat2: str):
        result = {}
        table = pd.crosstab(self.df[cat1], self.df[cat2])
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            stat, p, _, _ = chi2_contingency(table)
            result = {
                'test': 'chi2',
                'vars': (cat1, cat2),
                'shape': table.shape,
                'statistic': float(stat),
                'p_value': float(p)
            }
        return result

    def run(self):
        results = {'ttest': [], 'mannwhitney': [], 'anova': [], 'chi2': []}
        # Pairwise numeric ~ categorical
        for num in self.numeric:
            for cat in self.categorical:
                vals = self.df[cat].dropna().unique()
                if len(vals) == 2:
                    ttest_res = self.ttest_between_groups(cat, num, list(vals))
                    if ttest_res: results['ttest'].append({'cat': cat, 'num': num, **ttest_res})
                    mw_res = self.mannwhitney_between_groups(cat, num, list(vals))
                    if mw_res: results['mannwhitney'].append({'cat': cat, 'num': num, **mw_res})
                if len(vals) >= 2:
                    anova_res = self.anova(cat, num)
                    if anova_res: results['anova'].append({'cat': cat, 'num': num, **anova_res})
        # Pairwise categorical ~ categorical
        for i, cat1 in enumerate(self.categorical):
            for cat2 in self.categorical[i+1:]:
                chi2_res = self.chi2_test(cat1, cat2)
                if chi2_res: results['chi2'].append(chi2_res)
        self.results = results
        return results

    def export(self, file_out: str):
        Path(file_out).parent.mkdir(parents=True, exist_ok=True)
        with open(file_out, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results exported to {file_out}")

def main(input_file="data/dataset.csv", output_file="output/stat_tests.json"):
    try:
        df = pd.read_csv(input_file)
        runner = StatTestRunner(df)
        results = runner.run()
        runner.export(output_file)
        print(f"Results exported to {output_file}")
        # Print sample results
        for test in ['ttest', 'mannwhitney', 'anova', 'chi2']:
            print(f"\n== {test.upper()} (sample): ==")
            for rec in results[test][:2]:
                print(rec)
    except Exception as e:
        logger.error(f"Statistical testing failed: {e}")

if __name__ == "__main__":
    main()
