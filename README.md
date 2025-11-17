# Exploratory Data Analysis (EDA) 
Exploratory Data Analysis (EDA) is prioritized at the start of every data project. Understanding the data throughly enables efficient strategies and better project outcomes. 

This repository is organized by the specific questions asked to understand data better. 

Each folder contains a README.md, Python script, and the output(s) of the script demonstrating how to answer one of these core questions:
| Core Question | Folder | Includes |
|--------|----------|----------|
| Is my data clean and complete? | [`data-quality/`](./data-quality/) | Detect missing values (nulls), duplicates, invalid entries, type mismatches, and overall completeness.
| What do individual variables look like? | [`distributions/`](./distributions/) | Visualize distributions, frequency counts, histograms, density plots, and summary statistics of each variable.
| Which variables are related? |  [`correlations/`](./correlations/)|Statistical correlation and association analysis to determine relationships between variables (Pearson, Spearman, Cramér's V). 
| Are there anomalies or outliers? | [`outlier-detection/`](./outlier-detection/) | Identification of statistically unusual values, number of outliers per variable, consensus outliers from different methods, and their locations.
| Do natural groups exist? | `segmentation/` | Clustering and segmentation methods to find natural groups or clusters in the data. This is done before data is cleaned. 
| Are there patterns over time? | `temporal-analysis/` | Time series analysis and temporal trend detection in the data. This is done before data is cleaned. 
| Are differences statistically significant? | `statistical-testing/` | Hypothesis testing methods to determine if observed differences are statistically significant. This is done before data is cleaned. 
| Which variables matter most? | `feature-importance/` | Feature selection and importance analysis to identify variables that have greatest influence on outcome variables.  

