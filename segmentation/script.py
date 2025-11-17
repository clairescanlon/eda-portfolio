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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ClusteringMethod(ABC):
    """Base class for clustering algorithms."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit clustering model and return cluster labels."""
        pass


class KMeansClusterer(ClusteringMethod):
    """K-Means clustering with automatic elbow detection."""
    
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.model = None
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit K-Means and return labels."""
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        return self.model.fit_predict(X)
    
    def get_inertia(self) -> float:
        """Return within-cluster sum of squares."""
        return self.model.inertia_ if self.model else np.inf


class HierarchicalClusterer(ClusteringMethod):
    """Hierarchical (agglomerative) clustering."""
    
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.model = None
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit hierarchical clustering and return labels."""
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        return self.model.fit_predict(X)


class DBSCANClusterer(ClusteringMethod):
    """Density-based clustering (DBSCAN)."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
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
        self.df = df
        self.name = name
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.X_scaled = None
        self.results = {}
        logger.info(f"Initialized SegmentationAnalyzer for {name} with {len(self.numeric_cols)} numeric columns")
    
    def preprocess(self) -> np.ndarray:
        """
        Prepare data for clustering: select numeric columns, handle NaN.
        Returns scaled feature matrix.
        """
        X = self.df[self.numeric_cols].copy()
        
        # Log missing data info
        missing_per_col = X.isnull().sum()
        if missing_per_col.sum() > 0:
            logger.info(f"Missing values found: {missing_per_col[missing_per_col > 0].to_dict()}")
        
        # Forward-fill NaN for clustering (preserves structure)
        X = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)
        logger.info(f"Data preprocessed: shape {self.X_scaled.shape}")
        
        return self.X_scaled
    
    def find_optimal_k(self, max_k: int = 10) -> int:
        """
        Use elbow method to suggest optimal number of clusters.
        Returns suggested k value.
        """
        inertias = []
        for k in range(2, max_k + 1):
            kmeans = KMeansClusterer(n_clusters=k)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.get_inertia())
        
        # Simple elbow detection: largest drop in inertia
        differences = np.diff(inertias)
        optimal_k = np.argmax(np.diff(differences)) + 2
        logger.info(f"Suggested optimal k: {optimal_k}")
        
        return optimal_k
    
    def cluster(self, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Apply multiple clustering methods and evaluate quality.
        Returns results with labels and metrics per method.
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
                    'davies_bouldin': None
                }
            else:
                silhouette = silhouette_score(self.X_scaled, labels)
                davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
                
                results[method_name] = {
                    'labels': labels.tolist(),
                    'n_clusters': len(np.unique(labels)),
                    'silhouette': float(silhouette),
                    'davies_bouldin': float(davies_bouldin)
                }
            
            logger.info(f"{method_name}: {len(np.unique(labels))} clusters, silhouette={results[method_name]['silhouette']}")
        
        self.results['clustering'] = results
        return results
    
    def cluster_summary(self, labels: List[int]) -> Dict[str, Any]:
        """
        Compute per-cluster statistics (size, composition, centroids).
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
        """
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        return {
            'pca_coords': X_pca.tolist(),
            'cluster_labels': labels,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Compile full segmentation report."""
        report = {
            'dataset_name': self.name,
            'total_rows': len(self.df),
            'numeric_features': len(self.numeric_cols),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'clustering': self.results.get('clustering', {}),
            'summary': self.results.get('summary', {})
        }
        logger.info("Report generated")
        return report
    
    def export(self, file_path: str) -> None:
        """Export results to JSON."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results exported to {file_path}")


def analyze_segments(input_file: str, output_file: str, n_clusters: int = 3):
    """
    Main entry point: load data, cluster, and export results.
    """
    try:
        df = pd.read_csv(input_file)
        analyzer = SegmentationAnalyzer(df)
        
        # Preprocess and detect optimal k
        analyzer.preprocess()
        optimal_k = analyzer.find_optimal_k()
        
        # Cluster with suggested k
        cluster_results = analyzer.cluster(n_clusters=n_clusters)
        
        # Get summary for primary method (K-Means)
        kmeans_labels = cluster_results['kmeans']['labels']
        summary = analyzer.cluster_summary(kmeans_labels)
        analyzer.results['summary'] = summary
        
        # Generate and export
        report = analyzer.generate_report()
        analyzer.export(output_file)
        
        print(f"Segmentation analysis complete. Results: {len(np.unique(kmeans_labels))} clusters found.")
        print(f"Exported to {output_file}")
        
    except Exception as e:
        logger.error(f"Segmentation analysis failed: {e}")
        raise


if __name__ == "__main__":
    analyze_segments(
        input_file="data/dataset.csv",
        output_file="output/segmentation_results.json",
        n_clusters=3
    )
