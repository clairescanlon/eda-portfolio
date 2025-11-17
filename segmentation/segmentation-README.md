# Segmentation Module

## Overview
The segmentation module discovers natural groups ("clusters" or "segments") within your dataset by applying unsupervised machine learning techniques. 
This reveals hidden patterns, distinct customer types, or structure in the data without relying on predefined categories.

*Purpose:* Identify, visualize, and quantify distinct groups within your data to inform project strategy.

## Core Question Answered
**Do natural groups exist?**

This code:
* Applies multiple unsupervised clustering algorithms (K-Means, Hierarchical Clustering, DBSCAN) to discover natural groupings
* Evaluates cluster quality using metrics like silhouette score, inertia, and davies-bouldin index to find optimal number of clusters
* Assigns each data point to a cluster and provides descriptive statistics for each segment (size, composition, centroids)
* Visualizes clusters in reduced dimensions (PCA, t-SNE) and outputs segment assignments and summaries for interpretation

## Key Features
* Discovers segments in fully raw, uncleaned data—all missing values and outliers retained for transparent analysis
* Supports multiple clustering algorithms and automatically compares their results for robustness
* Provides comprehensive cluster quality metrics to assess whether groupings are meaningful or artefactual
* Outputs cluster assignments with per-cluster statistics (size, mean values, variance) for easy understanding
* Generates clear visualizations of clusters for stakeholder communication and team alignment
* Modular, extensible, and PEP8-compliant codebase ready for production use
* Full logging of algorithm choices, cluster metrics, and convergence information for reproducibility
* Results directly feed into downstream analyses, targeted modeling, or strategic business decisions

> [!TIP]
> Review discovered segments with domain experts to confirm they make business sense. Strong clustering often reveals actionable patterns or data collection issues worth investigating further.
