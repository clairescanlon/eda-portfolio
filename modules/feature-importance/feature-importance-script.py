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
from scipy.stats import spearmanr
from scipy.special import mutual_info_classif

# Import from utils
from utils.constants import (
    RANDOM_STATE,
    N_ESTIMATORS,
    PERMUTATION_N_REPEATS,
    MIN_IMPORTANCE_THRESHOLD,
    JSON_INDENT,
    LOGGING_FORMAT,
    LOGGING_LEVEL
)
from utils.data_utils import (
    get_numeric_columns,
    get_categorical_columns,
    check_missing_values
)


warnings.filterwarnings('ignore')

# Configure logging using standardized format from utils
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format=LOGGING_FORMAT
)
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
                logger.warning("Correlation failed for %s: %s", name, e)
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
        
        Uses RANDOM_STATE constant for reproducibility.
        
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
                mi_scores = mutual_info_classif(X, y_binned, random_state=RANDOM_STATE)
            else:
                mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
            
            importances = {name: float(score) for name, score in zip(feature_names, mi_scores)}
            
            # Normalize to 0-1
            max_val = max(mi_scores) if len(mi_scores) > 0 else 1.0
            normalized = {k: v / max_val if max_val > 0 else 0 for k, v in importances.items()}
            return normalized
            
        except Exception as e:
            logger.warning("Mutual information failed: %s", e)
            return {name: 0.0 for name in feature_names}


class TreeImportance(ImportanceMethod):
    """Calculate importance using tree-based models (Random Forest)."""

    def calculate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                  problem_type: str = 'regression') -> Dict[str, float]:
        """
        Calculate tree-based feature importance.
        
        Uses N_ESTIMATORS and RANDOM_STATE constants.
        
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
                model = RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            
            model.fit(X, y)
            importances = {name: float(score) for name, score in zip(feature_names, model.feature_importances_)}
            
            # Already normalized by sklearn (sums to 1)
            return importances
            
        except Exception as e:
            logger.warning("Tree importance failed: %s", e)
            return {name: 0.0 for name in feature_names}


class PermutationImportance(ImportanceMethod):
    """Calculate importance via permutation method."""

    def calculate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                  problem_type: str = 'regression') -> Dict[str, float]:
        """
        Calculate permutation-based feature importance.
        
        Uses PERMUTATION_N_REPEATS and RANDOM_STATE constants.
        
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
                model = RandomForestRegressor(
                    n_estimators=50,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=50,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            
            model.fit(X, y)
            
            # Calculate permutation importance using constant
            perm_imp = permutation_importance(
                model, X, y,
                n_repeats=PERMUTATION_N_REPEATS,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            
            importances = {name: float(score) for name, score in zip(feature_names, perm_imp.importances_mean)}
            
            # Normalize to 0-1
            max_val = max(importances.values()) if importances.values() else 1.0
            normalized = {k: v / max_val if max_val > 0 else 0 for k, v in importances.items()}
            return normalized
            
        except Exception as e:
            logger.warning("Permutation importance failed: %s", e)
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
        self.X = X
        self.y = y
        self.target_name = target_name
        self.feature_names = X.columns.tolist()
        self.problem_type = self._infer_problem_type()
        self.results = {}
        
        logger.info("Initialized FeatureImportanceAnalyzer: %s, %s features", self.problem_type, len(self.feature_names))

    def _infer_problem_type(self) -> str:
        """Infer whether problem is regression or classification."""
        if self.y.dtype in ['object', 'category', 'bool']:
            return 'classification'
        elif len(np.unique(self.y)) <= 20:
            return 'classification'
        else:
            return 'regression'

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for analysis, handling missing values gracefully.
        
        Note: This method imputes for internal calculation only - does not
        modify original data (aligns with EDA philosophy).
        """
        # Drop rows where target is missing
        valid_idx = ~self.y.isnull()
        X_clean = self.X[valid_idx].copy()
        y_clean = self.y[valid_idx].copy()
        
        # For features, impute with median/mode (for importance calc only)
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                if X_clean[col].dtype in ['object', 'category']:
                    X_clean[col] = X_clean[col].fillna(
                        X_clean[col].mode()[0] if not X_clean[col].mode().empty else 'missing'
                    )
                else:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        # Encode categorical features
        X_encoded = X_clean.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype in ['object', 'category']:
                X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
        
        # Encode target if classification
        if self.problem_type == 'classification' and y_clean.dtype in ['object', 'category']:
            y_clean = LabelEncoder().fit_transform(y_clean.astype(str))
        
        return X_encoded.values, y_clean.values

    def analyze(self) -> Dict[str, Any]:
        """
        Run feature importance analysis with all methods.
        
        Returns:
            Dictionary with results from all methods
        """
        logger.info("Starting feature importance analysis (%s problem)...", self.problem_type)
        
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
            logger.info("Calculating %s...", method_name)
            scores = method_obj.calculate(X, y, self.feature_names, self.problem_type)
            method_results[method_name] = scores
        
        # Calculate consensus importance (average across methods)
        consensus = {}
        for feature in self.feature_names:
            scores = [method_results[m].get(feature, 0) for m in methods.keys()]
            consensus[feature] = float(np.mean(scores))
        
        # Rank features by consensus
        ranked = sorted(consensus.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by MIN_IMPORTANCE_THRESHOLD
        ranked_filtered = [(f, s) for f, s in ranked if s >= MIN_IMPORTANCE_THRESHOLD]
        
        self.results = {
            'problem_type': self.problem_type,
            'target_variable': self.target_name,
            'n_features': len(self.feature_names),
            'by_method': method_results,
            'consensus': consensus,
            'ranked_features': [{'feature': f, 'importance': s} for f, s in ranked],
            'significant_features': [{'feature': f, 'importance': s} for f, s in ranked_filtered]
        }
        
        logger.info("Analysis complete: %s significant features (threshold: %s)", len(ranked_filtered), MIN_IMPORTANCE_THRESHOLD)
        
        return self.results

    def export(self, file_path: str) -> None:
        """
        Export results to JSON.
        
        Uses JSON_INDENT constant for consistent formatting.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=JSON_INDENT, default=str)
        
        logger.info("Feature importance exported to %s", file_path)


def analyze_feature_importance(input_file: str, target_col: str, output_file: str):
    """
    Main entry point for feature importance analysis.
    
    Args:
        input_file: Path to CSV
        target_col: Target column name
        output_file: Path for JSON output
    """
    try:
        logger.info(f"Loading data from {input_file}")
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
        print(f"✓ Significant features (>{MIN_IMPORTANCE_THRESHOLD}): {len(results['significant_features'])}")
        print(f"✓ Exported to {output_file}")
        
        print("\nTop 5 Most Important Features:")
        for i, item in enumerate(results['ranked_features'][:5], 1):
            print(f"  {i}. {item['feature']}: {item['importance']:.4f}")
        
    except Exception as e:
        logger.error("Feature importance analysis failed: %s", e)
        raise


if __name__ == "__main__":
    analyze_feature_importance(
        input_file="data/dataset.csv",
        target_col="target",
        output_file="output/feature_importance.json"
    )
