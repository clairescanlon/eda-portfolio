# Exploratory Data Analysis (EDA)
A systematic framework for understanding your data **before** any cleaning, modeling, or strategic decisions. This portfolio provides modular, reusable Python tools to answer eight core questions about data structure, quality, relationships, and patterns.

> [!NOTE]
> **Philosophy**: EDA tools **analyze and report**—they never modify, clean, or transform data. By understanding data exactly as it exists, you make informed decisions about your project strategy.

> [!WARNING]
> This repository is under construction.

---

## Why EDA First?

Before you build pipelines, train models, or make decisions, you need to understand your data:

- **Avoid wasted effort** — Identify data quality issues early, not after weeks of cleaning
- **Make informed decisions** — Know your data's true structure before choosing strategies
- **Build trust** — Show stakeholders exactly what data you're working with
- **Reduce surprises** — Understand patterns, outliers, and gaps before analysis

---

## Core Questions

Each folder contains a standalone Python script, README documentation, and sample outputs demonstrating how to answer one critical question. Click any question to jump to that module.

| Question | Module | What You'll Learn |
|----------|--------|------------------|
| **Is my data clean and complete?** | [`data-quality/`](./data-quality/) | Missing values • Duplicates • Invalid entries • Type mismatches • Overall completeness |
| **What do individual variables look like?** | [`distributions/`](./distributions/) | Distributions • Histograms • Density plots • Summary statistics • Frequency tables |
| **Which variables are related?** | [`correlations/`](./correlations/) | Pearson/Spearman correlations • Categorical associations • Variable relationships |
| **Are there anomalies or outliers?** | [`outlier-detection/`](./outlier-detection/) | Outlier identification • Consensus detection across methods • Anomaly locations |
| **Do natural groups exist?** | [`segmentation/`](./segmentation/) | Clustering analysis • Natural groupings • Segment characteristics |
| **Are there patterns over time?** | [`temporal-analysis/`](./temporal-analysis/) | Trends • Seasonality • Stationarity • Changepoint detection |
| **Are differences statistically significant?** | [`statistical-testing/`](./statistical-testing/) | Hypothesis tests • T-tests • ANOVA • Chi-square • P-values |
| **Which variables matter most?** | [`feature-importance/`](./feature-importance/) | Feature rankings • Importance scores • Predictive power analysis |

---

## What's in Each Module?

Every folder follows the same structure for consistency and ease of use:

- **`README.md`** — Detailed explanation of the question, methods, and how to interpret results
- **`script.py`** — Production-ready Python code following PEP8 and best practices
- **`output/`** — Sample outputs (JSON reports, visualizations, statistics)

---

## Key Principles

✅ **No data modification** — Tools analyze and report only; no cleaning, imputation, or transformation  
✅ **Raw data insights** — Understand your data exactly as it exists, including issues  
✅ **Multiple methods** — Most analyses use several approaches for robust, consensus results  
✅ **Modular design** — Each script runs independently; use any or all modules  
✅ **Production-ready** — PEP8-compliant, fully logged, error-handled Python code  
✅ **Structured output** — JSON reports and visualizations ready for documentation or downstream use  

---

## Getting Started

### Quick Start

1. **Choose a question** from the Core Questions table above
2. **Navigate to the module folder** (e.g., `data-quality/`)
3. **Read the README.md** to understand the analysis
4. **Run the script** on your CSV file:
