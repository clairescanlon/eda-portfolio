import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging
import json
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, entropy, chi2_contingency
from scipy.special import mutual_info_classif, mutual_info_regression

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ImportanceMethod(ABC):
    """Base class for feature importance calculation methods."""
    
    @abstractmethod
    def calculate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance scores."""
        pass


class CorrelationImportance(ImportanceMethod):
    """Calculate importance via correlation/association with target."""
    
    def calculate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                  problem_type: str = 'regression') -> Dict[str, float]:
        """
        Calculate Pearson/Spearman correlation importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Column names
            problem_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with normalized importance scores
        """
        importances = {}
        
        for i, name in enumerate(feature_names):
            feature = X[:, i]
            
            # Skip if insufficient variance
            if np.std(feature) == 0:
                importances[name] = 0.0
                continue
            
            try:
                if problem_type == 'regression':
                    corr, _ = spearmanr(feature, y)
                    importances[name] = float(abs(corr))
                else:
                    # For classification, use correlation of encoded target
                    if isinstance(y[0], str):
                        y_encoded = LabelEncoder().fit_transform(y)
                    else:
                        y_encoded = y
                    corr, _ = spearmanr(feature, y_encoded)
                    importances[name] = float(abs(corr))
            except Exception as e:
                logger.warning(f"Correlation failed for {name}: {e}")
                importances[name] = 0.0
        
        # Normalize to 0-1
        max_val = max(importances.values()) if importances.values() else 1.0
        normalized = {k: v / max_val if max_val > 0 else 0 for k, v in importances.items()}
        
        return normalized


class MutualInformationImportance(ImportanceMethod):
    """Calculate importance via mutual information with target."""
    
    def calculate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                  problem_type: str = 'regression') -> Dict[str, float]:
        """
        Calculate mutual information importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Column names
            problem_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with normalized importance scores
        """
        try:
            if problem_type == 'regression':
                # Discretize continuous y for mutual information
                y_binned = pd.qcut(y, q=10, duplicates='drop')
                mi_scores = mutual_info_classif(X, y_binned, random_state=42)
            else:
                mi_scores = mutual_info_classif(X, y, random_state=42)
            
            importances = {name: float(score) for name, score in zip(feature_names, mi_scores)}
            
            # Normalize to 0-1
            max_val = max(mi_scores) if len(mi_scores) > 0 else 1.0
            normalized = {k: v / max_val if max_val > 0 else 0 for k, v in importances.items()}
            
            return normalized
        except Exception as e:
            logger.warning(f"Mutual information failed: {e}")
            return {name: 0.0 for name in feature_names}


class TreeImportance(ImportanceMethod):
    """Calculate importance using tree-based models (Random Forest)."""
    
    def calculate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                  problem_type: str = 'regression') -> Dict[str, float]:
        """
        Calculate tree-based feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Column names
            problem_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with normalized importance scores
        """
        try:
            if problem_type == 'regression':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            importances = {name: float(score) for name, score in zip(feature_names, model.feature_importances_)}
            
            # Already normalized by sklearn (sums to 1)
            return importances
        except Exception as e:
            logger.warning(f"Tree importance failed: {e}")
            return {name: 0.0 for name in feature_names}


class PermutationImportance(ImportanceMethod):
    """Calculate importance via permutation method."""
    
    def calculate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                  problem_type: str = 'regression') -> Dict[str, float]:
        """
        Calculate permutation-based feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Column names
            problem_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with normalized importance scores
        """
        try:
            if problem_type == 'regression':
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            
            # Calculate permutation importance
            perm_imp = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
            importances = {name: float(score) for name, score in zip(feature_names, perm_imp.importances_mean)}
            
            # Normalize to 0-1
            max_val = max(importances.values()) if importances.values() else 1.0
            normalized = {k: v / max_val if max_val > 0 else 0 for k, v in importances.items()}
            
            return normalized
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            return {name: 0.0 for name in feature_names}


class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analysis with multiple methods."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, target_name: str = "target"):
        """
        Initialize analyzer.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            target_name: Name of target variable
        """
        self.X = X.copy()
        self.y = y.copy()
        self.target_name = target_name
        self.feature_names = X.columns.tolist()
        self.problem_type = self._infer_problem_type()
        self.results = {}
        logger.info(f"Initialized FeatureImportanceAnalyzer: {self.problem_type}, {len(self.feature_names)} features")
    
    def _infer_problem_type(self) -> str:
        """Infer whether problem is regression or classification."""
        if self.y.dtype in ['object', 'category', 'bool']:
            return 'classification'
        elif len(np.unique(self.y)) <= 20:
            return 'classification'
        else:
            return 'regression'
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for analysis, handling missing values gracefully."""
        # Drop rows where target is missing
        valid_idx = ~self.y.isnull()
        X_clean = self.X[valid_idx].copy()
        y_clean = self.y[valid_idx].copy()
        
        # For features, impute with median/mode (for importance calc only, doesn't modify original)
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                if X_clean[col].dtype in ['object', 'category']:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].mode()[0] if not X_clean[col].mode().empty else 'missing')
                else:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        # Encode categorical features
        X_encoded = X_clean.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype in ['object', 'category']:
                X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
        
        # Encode target if classification
        if self.problem_type == 'classification' and self.y.dtype in ['object', 'category']:
            y_clean = LabelEncoder().fit_transform(y_clean.astype(str))
        
        return X_encoded.values, y_clean.values
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run feature importance analysis with all methods.
        
        Returns:
            Dictionary with results from all methods
        """
        logger.info(f"Starting feature importance analysis ({self.problem_type} problem)...")
        
        X, y = self._prepare_data()
        
        # Initialize methods
        methods = {
            'correlation': CorrelationImportance(),
            'mutual_information': MutualInformationImportance(),
            'tree_based': TreeImportance(),
            'permutation': PermutationImportance()
        }
        
        # Calculate importance for each method
        method_results = {}
        for method_name, method_obj in methods.items():
            logger.info(f"Calculating {method_name}...")
            scores = method_obj.calculate(X, y, self.feature_names, self.problem_type)
            method_results[method_name] = scores
        
        # Calculate consensus importance (average across methods)
        consensus = {}
        for feature in self.feature_names:
            scores = [method_results[m].get(feature, 0) for m in methods.keys()]
            consensus[feature] = float(np.mean(scores))
        
        # Rank features by consensus
        ranked = sorted(consensus.items(), key=lambda x: x[1], reverse=True)
        
        self.results = {
            'problem_type': self.problem_type,
            'target_variable': self.target_name,
            'n_features': len(self.feature_names),
            'by_method': method_results,
            'consensus': consensus,
            'ranked_features': [{'feature': f, 'importance': s} for f, s in ranked]
        }
        
        return self.results
    
    def export(self, file_path: str) -> None:
        """Export results to JSON."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Feature importance exported to {file_path}")


def analyze_feature_importance(input_file: str, target_col: str, output_file: str):
    """
    Main entry point for feature importance analysis.
    
    Args:
        input_file: Path to CSV
        target_col: Target column name
        output_file: Path for JSON output
    """
    try:
        df = pd.read_csv(input_file)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        analyzer = FeatureImportanceAnalyzer(X, y, target_name=target_col)
        results = analyzer.analyze()
        analyzer.export(output_file)
        
        print(f"✓ Feature importance analysis complete")
        print(f"✓ Problem type: {results['problem_type']}")
        print(f"✓ Analyzed {results['n_features']} features")
        print(f"✓ Exported to {output_file}")
        print("\nTop 5 Most Important Features:")
        for i, item in enumerate(results['ranked_features'][:5], 1):
            print(f"  {i}. {item['feature']}: {item['importance']:.4f}")
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        raise


if __name__ == "__main__":
    analyze_feature_importance(
        input_file="data/dataset.csv",
        target_col="target",
        output_file="output/feature_importance.json"
    )
