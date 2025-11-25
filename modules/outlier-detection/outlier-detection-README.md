# Outlier Detection Module

## Overview
The outlier detection module identifies anomalies and unusual values in numeric datasets using three statistical methods. It does not modify or remove data—only identifies and reports potential outliers for investigation.

*Purpose:* Detect data anomalies before they corrupt downstream analytics, dashboards, or machine learning models.

## Core Question Answered: 
**Are there anomalies or outliers in my data?**

This code:
* Identifies which numeric values are statistically unusual using industry-standard algorithms (e.g., IQR, Z-Score, MAD)
* Counts the number of outliers per variable using multiple methods
* Flags high-confidence outliers (consensus: those detected by two or more methods)
* Localizes and reports the specific records/indices that are considered outliers for investigation

## Key Features:
* Detects outliers using Interquartile Range (IQR), Z-Score, and Median Absolute Deviation (MAD) methods—no data cleaning or filtering
* Reports outlier indices and values for reproducibility and audit trail
* Shows consensus (agreement) across methods for high-confidence anomaly detection
* Handles both univariate and multivariate data, outputting comprehensive reports per variable
* Outputs structured JSON with all results for further review, visualization, or downstream use
* Fast, modular, and PEP8-compliant implementation—ready to use in any portfolio or pipeline
* Visualizes distributions and outlier locations if desired (boxplots, histograms)
* No manual parameter tuning required—works robustly across many datasets
* Clear logging for all analysis steps and results

> [!WARNING]
> Outlier detection flags values that are unusual statistically—but further domain investigation is needed before removing or altering any data. Outliers may represent data entry errors or meaningful rare events. Review with context!
