# Feature Importance Module

## Overview
The feature importance module identifies and ranks which variables have the greatest influence on outcome or target variables. Using multiple statistical and machine learning methods, it quantifies how much each feature contributes to predicting or explaining your outcome of interest.

*Purpose:* Determine which variables matter most to inform feature selection, guide data collection priorities, and focus engineering efforts on high-impact variables before creating a project strategy.

## Core Question Answered: 
**Which variables matter most?**

This code:
* Calculates feature importance using multiple methods (correlation, mutual information, permutation importance, tree-based importance)
* Ranks features by their predictive power or association strength with the target variable
* Provides confidence intervals and statistical significance for importance estimates
* Visualizes feature rankings to help stakeholders quickly understand which variables drive outcomes

## Key Features:
* Analyzes raw, uncleaned data as-is—no dropping or filtering applied before importance calculation
* Supports both regression (continuous targets) and classification (categorical targets) problems
* Applies multiple importance methods and compares results for robustness and consensus
* Reports normalized importance scores (0-1) for easy comparison across features
* Generates clear rankings and visualizations for team communication and documentation
* Modular, extensible, and PEP8-compliant codebase ready for production workflows
* Full logging of method selection, calculations, and convergence for reproducibility and audit trails
* Results directly inform feature engineering priorities, data collection focus, and modeling strategy

> [!TIP]
> High feature importance does **not** automatically mean causation. Always validate results with domain experts and consider multicollinearity and confounding relationships before making strategic decisions.

> [!WARNING]
> Feature importance scores can be unstable with highly correlated features or small sample sizes. Use multiple methods and domain knowledge together when interpreting results.
