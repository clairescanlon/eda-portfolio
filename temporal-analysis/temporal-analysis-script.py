import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging
import json
import warnings
from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TimeSeriesDecomposer(ABC):
    """Base class for time series decomposition methods."""
    
    @abstractmethod
    def decompose(self, series: pd.Series, period: int) -> Dict[str, Any]:
        """Decompose time series into components."""
        pass


class AdditiveDecomposer(TimeSeriesDecomposer):
    """Additive decomposition: Y = Trend + Seasonal + Residual."""
    
    def decompose(self, series: pd.Series, period: int) -> Dict[str, Any]:
        """
        Perform additive decomposition.
        
        Args:
            series: Time series to decompose
            period: Seasonal period
            
        Returns:
            Dictionary with decomposition components and metrics
        """
        try:
            result = seasonal_decompose(series, model='additive', period=period, extrapolate='fill_mean')
            
            trend_var = np.var(result.trend.dropna())
            seasonal_var = np.var(result.seasonal)
            residual_var = np.var(result.resid)
            total_var = trend_var + seasonal_var + residual_var
            
            return {
                'method': 'additive',
                'trend': result.trend.tolist(),
                'seasonal': result.seasonal.tolist(),
                'residual': result.resid.tolist(),
                'trend_strength': float(trend_var / total_var) if total_var > 0 else 0.0,
                'seasonal_strength': float(seasonal_var / total_var) if total_var > 0 else 0.0,
                'residual_strength': float(residual_var / total_var) if total_var > 0 else 0.0
            }
        except Exception as e:
            logger.warning(f"Additive decomposition failed: {e}")
            return {'error': str(e), 'method': 'additive'}


class MultiplicativeDecomposer(TimeSeriesDecomposer):
    """Multiplicative decomposition: Y = Trend × Seasonal × Residual."""
    
    def decompose(self, series: pd.Series, period: int) -> Dict[str, Any]:
        """
        Perform multiplicative decomposition.
        
        Args:
            series: Time series (positive values only)
            period: Seasonal period
            
        Returns:
            Dictionary with decomposition components and metrics
        """
        try:
            if (series <= 0).any():
                logger.warning("Multiplicative decomposition requires positive values; skipping.")
                return {'error': 'Negative or zero values present', 'method': 'multiplicative'}
            
            result = seasonal_decompose(series, model='multiplicative', period=period, extrapolate='fill_mean')
            
            # Calculate variance explained
            trend_component = result.trend.dropna()
            seasonal_component = result.seasonal
            residual_component = result.resid
            
            trend_var = np.var(np.log(trend_component + 1e-8))
            seasonal_var = np.var(np.log(seasonal_component + 1e-8))
            residual_var = np.var(np.log(residual_component + 1e-8))
            total_var = trend_var + seasonal_var + residual_var
            
            return {
                'method': 'multiplicative',
                'trend': result.trend.tolist(),
                'seasonal': result.seasonal.tolist(),
                'residual': result.resid.tolist(),
                'trend_strength': float(trend_var / total_var) if total_var > 0 else 0.0,
                'seasonal_strength': float(seasonal_var / total_var) if total_var > 0 else 0.0,
                'residual_strength': float(residual_var / total_var) if total_var > 0 else 0.0
            }
        except Exception as e:
            logger.warning(f"Multiplicative decomposition failed: {e}")
            return {'error': str(e), 'method': 'multiplicative'}


class TemporalAnalyzer:
    """Comprehensive time series analysis without data modification."""
    
    def __init__(self, df: pd.DataFrame, time_col: str, value_cols: Optional[List[str]] = None):
        """
        Initialize temporal analyzer.
        
        Args:
            df: DataFrame with time series data
            time_col: Column name containing datetime/time index
            value_cols: Numeric columns to analyze (default: all numeric)
        """
        self.df = df.copy()
        self.time_col = time_col
        
        # Parse and set time index
        try:
            self.df[time_col] = pd.to_datetime(self.df[time_col])
            self.df.set_index(time_col, inplace=True)
            self.df.sort_index(inplace=True)
            logger.info(f"Time index set: {self.df.index.min()} to {self.df.index.max()}")
        except Exception as e:
            logger.error(f"Failed to parse time column {time_col}: {e}")
            raise
        
        # Select value columns
        if value_cols is None:
            self.value_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.value_cols = value_cols
        
        self.results = {}
        logger.info(f"Initialized TemporalAnalyzer with {len(self.value_cols)} time series")
    
    def detect_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Test stationarity using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with ADF test results
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            return {'method': 'ADF', 'status': 'insufficient_data', 'n_obs': len(series_clean)}
        
        try:
            adf_result = adfuller(series_clean, autolag='AIC')
            
            return {
                'method': 'ADF',
                'statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'is_stationary': adf_result[1] < 0.05,
                'n_lags': int(adf_result[2]),
                'n_obs': int(adf_result[3]),
                'critical_values': {k: float(v) for k, v in adf_result[4].items()}
            }
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            return {'method': 'ADF', 'error': str(e)}
    
    def detect_autocorrelation(self, series: pd.Series, nlags: int = 40) -> Dict[str, Any]:
        """
        Calculate ACF and PACF for seasonality detection.
        
        Args:
            series: Time series to analyze
            nlags: Number of lags
            
        Returns:
            Dictionary with ACF/PACF results
        """
        series_clean = series.dropna()
        
        if len(series_clean) < nlags + 5:
            return {'error': 'Insufficient data', 'n_obs': len(series_clean), 'required': nlags + 5}
        
        try:
            acf_vals = acf(series_clean, nlags=nlags, fft=True)
            pacf_vals = pacf(series_clean, nlags=nlags, method='ywm')
            
            # Significance threshold (95% CI)
            ci = 1.96 / np.sqrt(len(series_clean))
            significant_acf = [i for i, val in enumerate(acf_vals[1:], 1) if abs(val) > ci]
            significant_pacf = [i for i, val in enumerate(pacf_vals[1:], 1) if abs(val) > ci]
            
            return {
                'acf': acf_vals.tolist(),
                'pacf': pacf_vals.tolist(),
                'confidence_interval': float(ci),
                'significant_acf_lags': significant_acf[:10],  # Top 10 for brevity
                'significant_pacf_lags': significant_pacf[:10]
            }
        except Exception as e:
            logger.warning(f"Autocorrelation failed: {e}")
            return {'error': str(e)}
    
    def analyze_trend(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze trend direction and strength via linear regression.
        
        Args:
            series: Time series to analyze
            
        Returns:
            Dictionary with trend metrics
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 3:
            return {'error': 'Insufficient data', 'n_obs': len(series_clean)}
        
        x = np.arange(len(series_clean))
        y = series_clean.values
        
        # Linear regression
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        
        # Determine direction
        std_y = np.std(y)
        noise_level = std_y / len(series_clean)
        
        if abs(slope) < noise_level:
            direction = 'flat'
        elif slope > 0:
            direction = 'upward'
        else:
            direction = 'downward'
        
        # Calculate trend strength (R-squared)
        y_pred = np.polyval(coefficients, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'slope': float(slope),
            'direction': direction,
            'r_squared': float(r_squared),
            'mean': float(np.mean(y)),
            'std': float(std_y),
            'min': float(np.min(y)),
            'max': float(np.max(y))
        }
    
    def detect_changepoints(self, series: pd.Series, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect sudden changes in time series level.
        
        Args:
            series: Time series to analyze
            threshold: Number of standard deviations for changepoint threshold
            
        Returns:
            Dictionary with changepoint indices and magnitudes
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            return {'error': 'Insufficient data', 'changepoints': []}
        
        # Compute differences
        diff = np.diff(series_clean.values)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        # Identify large deviations
        threshold_val = mean_diff + threshold * std_diff
        changepoints = [i + 1 for i, d in enumerate(diff) if abs(d - mean_diff) > threshold_val]
        
        return {
            'changepoint_indices': changepoints[:20],  # Top 20 for brevity
            'n_changepoints': len(changepoints),
            'mean_change': float(mean_diff),
            'std_change': float(std_diff)
        }
    
    def decompose_series(self, series: pd.Series, period: int = 12) -> Dict[str, Any]:
        """
        Apply decomposition methods.
        
        Args:
            series: Time series to decompose
            period: Seasonal period
            
        Returns:
            Dictionary with decompositions
        """
        decompositions = {}
        
        # Additive
        additive = AdditiveDecomposer()
        result_add = additive.decompose(series, period)
        if 'error' not in result_add:
            decompositions['additive'] = result_add
        
        # Multiplicative (only if all positive)
        if (series > 0).all():
            multiplicative = MultiplicativeDecomposer()
            result_mult = multiplicative.decompose(series, period)
            if 'error' not in result_mult:
                decompositions['multiplicative'] = result_mult
        
        return decompositions
    
    def analyze(self, period: int = 12) -> Dict[str, Any]:
        """
        Run full temporal analysis on all value columns.
        
        Args:
            period: Seasonal period
            
        Returns:
            Complete analysis results
        """
        for col in self.value_cols:
            logger.info(f"Analyzing temporal patterns in {col}...")
            series = self.df[col]
            
            self.results[col] = {
                'total_observations': len(series),
                'missing_values': int(series.isnull().sum()),
                'date_range': {
                    'start': str(series.index.min()),
                    'end': str(series.index.max()),
                    'span_days': (series.index.max() - series.index.min()).days
                },
                'stationarity': self.detect_stationarity(series),
                'autocorrelation': self.detect_autocorrelation(series),
                'trend': self.analyze_trend(series),
                'changepoints': self.detect_changepoints(series),
                'decomposition': self.decompose_series(series, period=period)
            }
        
        return self.results
    
    def export(self, file_path: str) -> None:
        """Export temporal analysis to JSON."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Temporal analysis exported to {file_path}")


def analyze_temporal_patterns(input_file: str, time_col: str, output_file: str, period: int = 12):
    """
    Main entry point: load, analyze, and export temporal results.
    
    Args:
        input_file: Path to CSV with time series data
        time_col: Column name containing datetime
        output_file: Path for JSON output
        period: Seasonal period
    """
    try:
        df = pd.read_csv(input_file)
        analyzer = TemporalAnalyzer(df, time_col=time_col)
        results = analyzer.analyze(period=period)
        analyzer.export(output_file)
        
        print(f"✓ Temporal analysis complete on {len(results)} time series")
        print(f"✓ Exported to {output_file}")
        
        # Print summary
        for col, analysis in results.items():
            trend = analysis['trend'].get('direction', 'unknown')
            stationarity = "stationary" if analysis['stationarity'].get('is_stationary') else "non-stationary"
            print(f"  {col}: {trend} trend, {stationarity}")
        
    except Exception as e:
        logger.error(f"Temporal analysis failed: {e}")
        raise


if __name__ == "__main__":
    analyze_temporal_patterns(
        input_file="data/timeseries.csv",
        time_col="date",
        output_file="output/temporal_analysis.json",
        period=12
    )
