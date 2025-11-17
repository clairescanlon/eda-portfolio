# Exploratory Data Analysis (EDA) 
Exploratory Data Analysis (EDA) is prioritized at the start of every data project. Understanding the data throughly enables efficient strategies and better project outcomes. 

This repository is organized by the specific questions asked to understand data better. 

Each folder contains a README.md, Python script, and the output(s) of the script demonstrating how to answer one of these core questions:
| Core Question | Folder | Includes |
|--------|----------|----------|
| Is my data clean and complete? | [`data-quality/`](./data-quality/) | Are there missing values (nulls) that break analysis?
| What do individual variables look like? | `distributions/` |
| Which variables are related? |  [`correlations/`](./correlations/)|
| Are there anomalies or outliers? | [`outlier-detection/`](./outlier-detection/) | Which numeric values are statistically unusual? How many outliers exist per column?  Which values are flagged by multiple detection methods (consensus)? Where are these anomalous values located? 
| Do natural groups exist? | `segmentation/` |
| Are there patterns over time? | `temporal-analysis/` |
| Are differences statistically significant? | `statistical-testing/` |
| Which variables matter most? | `feature-importance/` |

