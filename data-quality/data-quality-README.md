# Data Quality Module

## Overview
The data quality module systematically evaluates dataset integrity across multiple quality dimensions. It diagnoses and reports data quality issues (missing values, duplicates, invalid entries, type mismatches) **without modifying, cleaning, or transforming the data**—enabling you to understand data structure exactly as it exists before creating a project strategy.

*Purpose:* Detect data quality issues before they corrupt downstream analytics, dashboards, or machine learning models. Identify issues with the data before creating a strategy for the data in the project.

## Core Question Answered: 
**Is my data clean and complete?**

This code:
* Assesses completeness by measuring null/missing values per column and overall dataset fill rate
* Evaluates consistency by identifying duplicate records and unique value patterns
* Validates data against business rules (min/max constraints, allowed values, type requirements)
* Reports type compliance—ensuring each column contains the expected data type
* Flags specific columns and records with quality issues for investigation and remediation

## Key Features:
* Diagnoses data quality issues **without cleaning, dropping, or modifying any data**—all analysis on raw input
* Assesses four critical dimensions: Completeness, Consistency, Validity, and Type Compliance
* Supports parameterized validation rules—define min/max bounds, allowed values, and required fields per project
* Reports structured metrics: null counts, percentages, violation details, and issue indices
* Outputs comprehensive JSON report with per-field and per-issue granularity for easy troubleshooting
* Modular, extensible, and PEP8-compliant codebase ready for production use
* Full logging of all analysis steps and findings for reproducibility and audit trails
* Results directly inform data cleaning strategy, resource allocation, and project feasibility before work begins

> [!TIP]
> Data quality issues are often symptoms of upstream problems—data collection errors, system integration issues, or schema changes. Always investigate root causes in source systems before spending time cleaning data downstream.

> [!WARNING]
> High missing-data percentages (>20%) in critical fields may require source system intervention or entirely new data collection. Assess feasibility and timelines before committing to data cleaning.
