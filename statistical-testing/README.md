
# Statistical Testing Module 

## Overview
This code performs rigorous statistical hypothesis tests to determine if observed differences between groups or variables are statistically significant. Results are calculated on the raw data with all missing values, outliers, and data types left exactly as-is.

## Core Question: 
**Are observed differences (between groups, categories, or variables) in my data statistically significant, or likely due to random chance?**

This code:</br>
* Compares numeric variables between groups (e.g. t-test, ANOVA, Mann-Whitney U)</br>
* Tests categorical associations (chi-square test)</br>
* Provides exact p-values/statistics per comparison</br>
* Flags which differences are likely meaningful for further investigation</br>

## Key Features: </br>
* Detects statistical differences without cleaning or dropping data
* Runs parametric (t-test, ANOVA) and non-parametric (Mann-Whitney, chi-square) tests automatically
* Handles numeric and categorical variables for all possible combinations
* Outputs structured JSON with all results, ready for inclusion in team reports or dashboards
* Modular, readable, and PEP8-compliant codebase
* Clearly reports n values (group sizes) and p-values (statistical significance)
* Logs full workflow for reproducibility and future audits
 Results inform strategy—filtering, splits, modeling, or root cause investigation
* No manual parameter tuning—fully automated and robust for new data

> [!TIP]
> Always interpret statistically significant results (p < 0.05) in the context of your project. Statistical significance does **not** always mean practical significance, especially with large datasets.
