# Exploratory Data Analysis (EDA)
A systematic framework for understanding your data before any cleaning, modeling, or strategic decisions. This repository provides modular, reusable Python tools to answer core questions about data structure, quality, relationships, and patterns.


> [!NOTE]
> **Philosophy**: These EDA tools analyze and report. They never modify, clean, or transform data. By understanding data exactly as it exists, you make informed decisions about your project strategy.


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
## 🛠️ What Each Module Teaches
* **data-quality**: Missing values, duplicates, data validation, completeness assessment, data profiling, quality metrics, anomaly detection in data integrity
* **distributions**: Distribution analysis, shape characteristics (skewness, kurtosis), normality testing, visual exploration, histogram interpretation, probability distributions, tail analysis
* **feature-importance**: Feature ranking, variable influence on outcomes, correlation analysis, feature selection methods, predictive power assessment, dominance patterns
* **outlier-detection**: Anomaly identification, detection methods (statistical, clustering-based), outlier handling strategies, impact quantification, IQR analysis, z-score methods, isolation techniques
* **segmentation**: Clustering and grouping, customer/data segmentation, pattern identification, cluster characteristics, segmentation strategies, group profiling, market segmentation
* **statistical-testing**: Hypothesis testing, p-values, significance levels, t-tests, chi-square tests, ANOVA, test selection criteria, statistical validation of relationships
* **temporal-analysis**: Time series exploration, trend analysis, seasonality detection, autocorrelation, time-based patterns, date-based aggregations, time-window analysis

These 7 modules represent a comprehensive, systematic approach to exploratory data analysis that covers all the foundational questions data professionals ask when beginning a new project: Is the data clean? How is it distributed? What features matter? What are the outliers? How does it segment? Are patterns statistically significant? And what changes over time?

---

## 📁 Repository Structure
eda-portfolio/
├── README.md                      # Main documentation
├── requirements.txt               # Dependencies
├── data-quality/                  # Data quality module
│   ├── README.md
│   └── data-quality-script.py
├── distributions/                 # Distribution analysis module
│   ├── README.md
│   └── distributions-script.py
├── feature-importance/            # Feature importance module
│   ├── README.md
│   └── feature-importance-script.py
├── outlier-detection/             # Anomaly detection module
│   ├── README.md
│   └── outlier-detection-script.py
├── segmentation/                  # Clustering & grouping module
│   ├── README.md
│   └── segmentation-script.py
├── statistical-testing/           # Hypothesis testing module
│   ├── README.md
│   └── statistical-testing-script.py
├── temporal-analysis/             # Time series analysis module
│   ├── README.md
│   └── temporal-analysis-script.py
├── src/
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       └── helpers.py
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_*.py
└── docs/                          # Additional documentation
    └── METHODOLOGY.md

> [!WARNING]
> This repository is under construction.


---
## 📚 Documentation
For detailed documentation on each module, see:
* data-quality/README.md — Data quality methodology
* distributions/README.md — Distribution analysis
* correlations/README.md — Correlation analysis
* outlier-detection/README.md — Anomaly detection
* segmentation/README.md — Clustering analysis
* temporal-analysis/README.md — Time series analysis
* statistical-testing/README.md — Hypothesis testing
* feature-importance/README.md — Feature ranking

---
## 👤 Author
Claire Scanlon 
* AWS Certified Cloud Practitioner
* Full stack data professional


## 🔗 Connect
* GitHub: [Claire Scanlon GitHub](https://github.com/clairescanlon/)
* LinkedIn: [Claire Scanlon](https://www.linkedin.com/in/clairescanlon/)
* Portfolio: [Website](https://www.claire-scanlon.com)
* Medium: [Insights, Tutorials, and Thought Leadership](https://medium.com/@clairehelenscanlon)


