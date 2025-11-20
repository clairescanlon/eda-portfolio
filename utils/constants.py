# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_STATE = 42
"""Random seed for all stochastic operations to ensure reproducibility."""

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'
"""Standardized logging format across all modules (no dashes separator)."""

LOGGING_LEVEL = 'INFO'
"""Default logging level for all modules."""

# ============================================================================
# FILE PATHS
# ============================================================================
DEFAULT_INPUT_PATH = "data/dataset.csv"
"""Default input data path."""

DEFAULT_OUTPUT_DIR = "output"
"""Default output directory for all results."""

# ============================================================================
# JSON/EXPORT SETTINGS
# ============================================================================
JSON_INDENT = 2
"""Indentation level for JSON exports."""

FLOAT_PRECISION = 4
"""Decimal precision for floating point exports."""

# ============================================================================
# CLUSTERING PARAMETERS (segmentation-script.py)
# ============================================================================
DEFAULT_N_CLUSTERS = 3
"""Default number of clusters for K-Means."""

PCA_N_COMPONENTS = 2
"""Number of principal components for visualization."""

SILHOUETTE_MIN_SCORE = 0.5
"""Minimum acceptable silhouette score for cluster quality."""

# ============================================================================
# OUTLIER DETECTION THRESHOLDS (outlier-detection-script.py)
# ============================================================================
ZSCORE_THRESHOLD = 3.0
"""Z-score threshold for outlier detection (3 standard deviations)."""

MAD_THRESHOLD = 3.5
"""Modified Z-score threshold using Median Absolute Deviation."""

IQR_MULTIPLIER = 1.5
"""IQR multiplier for outlier detection (standard Tukey method)."""

ISOLATION_CONTAMINATION = 0.1
"""Expected proportion of outliers for Isolation Forest (10%)."""

# ============================================================================
# TIME SERIES PARAMETERS (temporal-analysis-script.py)
# ============================================================================
DEFAULT_SEASONAL_PERIOD = 12
"""Default seasonal period for time series analysis (12 for monthly data)."""

ACF_MAX_LAGS = 40
"""Maximum number of lags for autocorrelation analysis."""

KPSS_REGRESSION = 'c'
"""KPSS test regression type ('c' for constant, 'ct' for constant+trend)."""

ADF_AUTOLAG = 'AIC'
"""ADF test autolag selection criterion."""

# ============================================================================
# STATISTICAL TESTING (statistical-testing-script.py)
# ============================================================================
ALPHA = 0.05
"""Significance level for hypothesis tests (5%)."""

MIN_GROUP_SIZE = 30
"""Minimum group size for statistical tests."""

# ============================================================================
# FEATURE IMPORTANCE (feature-importance-script.py)
# ============================================================================
N_ESTIMATORS = 100
"""Number of trees for Random Forest models."""

MIN_IMPORTANCE_THRESHOLD = 0.01
"""Minimum feature importance to report (1%)."""

PERMUTATION_N_REPEATS = 10
"""Number of permutations for permutation importance."""

# ============================================================================
# DATA QUALITY THRESHOLDS (data-quality-script.py)
# ============================================================================
MAX_MISSING_RATE = 0.3
"""Maximum acceptable missing value rate per column (30%)."""

MIN_UNIQUE_RATIO = 0.01
"""Minimum unique value ratio for valid categorical (1%)."""

MAX_UNIQUE_RATIO = 0.95
"""Maximum unique value ratio for valid categorical (95%)."""

# ============================================================================
# VISUALIZATION
# ============================================================================
FIGURE_DPI = 100
"""DPI for saved figures."""

FIGURE_SIZE = (10, 6)
"""Default figure size (width, height) in inches."""
