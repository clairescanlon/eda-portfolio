# Distributions Module

## Overview
The distributions module explores and visualizes how individual variables are distributed across your dataset. It generates summary statistics, histograms, density plots, boxplots, and frequency tables for each variable **without modifying, cleaning, or transforming the data**—showing you the natural shape and characteristics of your variables exactly as they exist.

*Purpose:* Understand the distribution, range, central tendency, and spread of each variable at project start to inform data transformation needs, identify skewness or bimodality, and guide downstream analysis strategies before any data cleaning occurs.

## Core Question Answered: 
**What do individual variables look like?**

This code:
* Calculates comprehensive summary statistics (mean, median, std dev, min/max, quartiles) for numeric variables
* Generates histograms and kernel density estimates (KDE) to visualize numeric distributions and identify skewness
* Creates boxplots to show numeric outliers, quartile ranges, and distribution shape at a glance
* Computes frequency tables and bar plots for categorical variables with value counts and percentages
* Reports missing value counts per variable for completeness assessment
* Identifies top categories (mode) and their frequencies for categorical data

## Key Features:
* Visualizes and summarizes data **exactly as-is**—no cleaning, dropping, or imputation of missing values
* Generates publication-quality plots (histograms, KDE, boxplots, bar charts) for all variables
* Exports both visual plots (PNG) and structured summary statistics (JSON) for reporting
* Handles numeric and categorical variables automatically with appropriate visualizations
* Supports variables with missing/NaN values—includes missing count in all summaries
* Outputs JSON summary with full statistics for integration into dashboards or reports
* Modular, extensible, and PEP8-compliant codebase ready for production use
* Full logging of analysis steps and export paths for reproducibility and troubleshooting
* Results inform skewness corrections, outlier handling, binning strategies, and feature engineering before modeling

> [!TIP]
> Highly skewed distributions may require log transformation or other normalization before modeling. Review histograms carefully to identify candidates for transformation.

> [!WARNING]
> Bimodal or multimodal distributions may indicate data quality issues (e.g., data from two different sources combined), or reveal meaningful natural groups. Investigate unexpected distribution shapes with domain experts before deciding on handling strategy.

