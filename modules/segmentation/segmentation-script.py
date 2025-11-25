import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
import logging
import json
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Import from utils
from utils.constants import (
    RANDOM_STATE,
    DEFAULT_N_CLUSTERS,
    PCA_N_COMPONENTS,
    SILHOUETTE_MIN_SCORE,
    JSON_INDENT,
    LOGGING_FORMAT,
    LOGGING_LEVEL
)
from utils.data_utils import (
    get_numeric_columns,
    check_missing_values
)


# Configure logging using standardized format from utils
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format=LOGGING_FORMAT
)
logger = logging.getLogger(__name__)


class ClusteringMethod(ABC):
    """Base class for clustering algorithms."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit clustering model and return cluster labels."""
        pass


class KMeansClusterer(ClusteringMethod):
    """K-Means clustering with automatic elbow detection."""

    def __init__(self, n_clusters: int = DEFAULT_N_CLUSTERS):
        """
        Initialize K-Means clusterer.
        
        Args:
            n_clusters: Number of clusters (from constants, default 3)
        """
        self.n_clusters = n_clusters
        self.model = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit K-Means and return labels.
        
        Uses RANDOM_STATE constant for reproducibility.
        """
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=RANDOM_STATE,
            n_init=10
        )
        return self.model.fit_predict(X)

    def get_inertia(self) -> float:
        """Return within-cluster sum of squares."""
        return self.model.inertia_ if self.model else np.inf


class HierarchicalClusterer(ClusteringMethod):
    """Hierarchical (agglomerative) clustering."""

    def __init__(self, n_clusters: int = DEFAULT_N_CLUSTERS):
        """
        Initialize hierarchical clusterer.
        
        Args:
            n_clusters: Number of clusters (from constants, default 3)
        """
        self.n_clusters = n_clusters
        self.model = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit hierarchical clustering and return labels."""
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        return self.model.fit_predict(X)


class DBSCANClusterer(ClusteringMethod):
    """Density-based clustering (DBSCAN)."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize DBSCAN clusterer.
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit DBSCAN and return labels."""
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return self.model.fit_predict(X)


class SegmentationAnalyzer:
    """Performs clustering analysis with multiple algorithms and quality metrics."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        """
        Initialize segmentation analyzer.
        
        Uses get_numeric_columns() from data_utils.
        
        Args:
            df: pandas DataFrame to analyze
            name: descriptive name for dataset
        """
        self.df = df
        self.name = name
        
        # Use utility function for column detection
        self.numeric_cols = get_numeric_columns(df)
        
        self.X_scaled = None
        self.results = {}
        
        logger.info("Initialized SegmentationAnalyzer for %s with %s numeric columns", name, len(self.numeric_cols))

    def preprocess(self) -> np.ndarray:
        """
        Prepare data for clustering: select numeric columns, handle NaN.
        
        Note: Imputes missing values for clustering calculation only.
        Does not modify original data (aligns with EDA philosophy).
        
        Returns:
            Scaled feature matrix
        """
        X = self.df[self.numeric_cols].copy()
        
        # Use utility function to check missing values
        missing_stats = check_missing_values(self.df[self.numeric_cols])
        
        if missing_stats:
            logger.info("Missing values found: %s", missing_stats)
        
        # Impute with mean for clustering (calculation only, doesn't modify original)
        X = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)
        
        logger.info("Data preprocessed: shape %s", self.X_scaled.shape)
        return self.X_scaled

    def find_optimal_k(self, max_k: int = 10) -> int:
        """
        Use elbow method to suggest optimal number of clusters.
        
        Uses RANDOM_STATE constant for reproducibility.
        
        Args:
            max_k: Maximum number of clusters to test
            
        Returns:
            Suggested k value
        """
        inertias = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeansClusterer(n_clusters=k)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.get_inertia())
        
        # Simple elbow detection: largest drop in inertia
        differences = np.diff(inertias)
        optimal_k = np.argmax(np.diff(differences)) + 2
        
        logger.info("Suggested optimal k: %s", optimal_k)
        return optimal_k

    def cluster(self, n_clusters: int = DEFAULT_N_CLUSTERS) -> Dict[str, Any]:
        """
        Apply multiple clustering methods and evaluate quality.
        
        Uses DEFAULT_N_CLUSTERS and SILHOUETTE_MIN_SCORE constants.
        
        Args:
            n_clusters: Number of clusters (uses constant by default)
            
        Returns:
            Results with labels and metrics per method
        """
        methods = {
            'kmeans': KMeansClusterer(n_clusters=n_clusters),
            'hierarchical': HierarchicalClusterer(n_clusters=n_clusters),
            'dbscan': DBSCANClusterer(eps=0.5, min_samples=5)
        }
        
        results = {}
        
        for method_name, clusterer in methods.items():
            labels = clusterer.fit(self.X_scaled)
            
            # Skip quality metrics if only 1 cluster or noise labels (DBSCAN)
            if len(np.unique(labels)) < 2:
                results[method_name] = {
                    'labels': labels.tolist(),
                    'n_clusters': len(np.unique(labels)),
                    'silhouette': None,
                    'davies_bouldin': None,
                    'quality_acceptable': False
                }
            else:
                silhouette = silhouette_score(self.X_scaled, labels)
                davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
                
                # Check if silhouette meets minimum threshold
                quality_acceptable = silhouette >= SILHOUETTE_MIN_SCORE
                
                results[method_name] = {
                    'labels': labels.tolist(),
                    'n_clusters': len(np.unique(labels)),
                    'silhouette': float(silhouette),
                    'davies_bouldin': float(davies_bouldin),
                    'quality_acceptable': quality_acceptable
                }
                
                quality_msg = "GOOD" if quality_acceptable else "POOR (< %s)" % 
                SILHOUETTE_MIN_SCORE
                logger.info("%s: %s clusters, silhouette=%.4f [%s]",
                method_name, len(np.unique(labels)), silhouette, quality_msg)
        
        self.results['clustering'] = results
        return results

    def cluster_summary(self, labels: List[int]) -> Dict[str, Any]:
        """
        Compute per-cluster statistics (size, composition, centroids).
        
        Args:
            labels: Cluster labels for each data point
            
        Returns:
            Dictionary with statistics for each cluster
        """
        cluster_data = pd.DataFrame({
            'cluster': labels,
            **{col: self.df[col].values for col in self.numeric_cols}
        })
        
        summary = {}
        
        for cluster_id in np.unique(labels):
            cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
            summary[int(cluster_id)] = {
                'size': int(len(cluster_subset)),
                'percentage': float(len(cluster_subset) / len(self.df) * 100),
                'centroid': cluster_subset[self.numeric_cols].mean().to_dict(),
                'variance': cluster_subset[self.numeric_cols].var().to_dict()
            }
        
        return summary

    def reduce_dimensions(self, labels: List[int]) -> Dict[str, Any]:
        """
        Reduce to 2D using PCA for visualization.
        
        Uses PCA_N_COMPONENTS constant.
        
        Args:
            labels: Cluster labels for each data point
            
        Returns:
            Dictionary with PCA coordinates and variance explained
        """
        pca = PCA(n_components=PCA_N_COMPONENTS)
        X_pca = pca.fit_transform(self.X_scaled)
        
        return {
            'pca_coords': X_pca.tolist(),
            'cluster_labels': labels,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'n_components': PCA_N_COMPONENTS
        }

    def generate_report(self) -> Dict[str, Any]:
        """
        Compile full segmentation report.
        
        Includes configuration parameters used.
        
        Returns:
            Complete segmentation report
        """
        report = {
            'dataset_name': self.name,
            'total_rows': len(self.df),
            'numeric_features': len(self.numeric_cols),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'configuration': {
                'default_n_clusters': DEFAULT_N_CLUSTERS,
                'pca_n_components': PCA_N_COMPONENTS,
                'silhouette_min_score': SILHOUETTE_MIN_SCORE,
                'random_state': RANDOM_STATE
            },
            'clustering': self.results.get('clustering', {}),
            'summary': self.results.get('summary', {})
        }
        
        logger.info("Report generated")
        return report

    def export(self, file_path: str) -> None:
        """
        Export results to JSON.
        
        Uses JSON_INDENT constant for consistent formatting.
        
        Args:
            file_path: relative path for output
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=JSON_INDENT, default=str)
        
        logger.info("Results exported to %s", file_path)


def analyze_segments(input_file: str, output_file: str, 
                     n_clusters: int = DEFAULT_N_CLUSTERS):
    """
    Main entry point: load data, cluster, and export results.
    
    Uses DEFAULT_N_CLUSTERS constant by default.
    
    Args:
        input_file: Path to input CSV
        output_file: Path for output JSON
        n_clusters: Number of clusters (uses constant by default)
    """
    try:
        logger.info("Loading data from %s", input_file)
        df = pd.read_csv(input_file)
        
        analyzer = SegmentationAnalyzer(df)
        
        # Preprocess and detect optimal k
        analyzer.preprocess()
        optimal_k = analyzer.find_optimal_k()
        
        logger.info("Using n_clusters=%s (optimal suggested: %s)", n_clusters, optimal_k)
        
        # Cluster with specified or default k
        cluster_results = analyzer.cluster(n_clusters=n_clusters)
        
        # Get summary for primary method (K-Means)
        kmeans_labels = cluster_results['kmeans']['labels']
        summary = analyzer.cluster_summary(kmeans_labels)
        analyzer.results['summary'] = summary
        
        # Add PCA dimensionality reduction
        pca_results = analyzer.reduce_dimensions(kmeans_labels)
        analyzer.results['pca'] = pca_results
        
        # Generate and export
        report = analyzer.generate_report()
        analyzer.export(output_file)
        
        # Print summary
        print(f"✓ Segmentation analysis complete")
        print(f"✓ Results: {len(np.unique(kmeans_labels))} clusters found")
        print(f"✓ Silhouette score: {cluster_results['kmeans']['silhouette']:.4f}")
        print(f"✓ Quality: {'GOOD' if cluster_results['kmeans']['quality_acceptable'] else 'POOR'}")
        print(f"✓ Exported to {output_file}")
        
    except Exception as e:
        logger.error("Segmentation analysis failed: %s", e)
        raise


if __name__ == "__main__":
    analyze_segments(
        input_file="data/dataset.csv",
        output_file="output/segmentation_results.json",
        n_clusters=DEFAULT_N_CLUSTERS
    )
