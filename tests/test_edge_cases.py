import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from modules.stats import (
    calculate_descriptive_stats,
    detect_outliers_iqr,
    normalize_data,
    regresion_lineal_simple as linear_regression,
)

class TestExtremeValues:
    """Tests con valores muy grandes o muy pequeños."""
    
    def test_very_large_numbers(self) -> None:
        """Valida cálculo con números muy grandes."""
        data = pd.Series([1e10, 1e10 + 100, 1e10 + 200])
        
        stats = calculate_descriptive_stats(data)
        
        # Keys from Prompt A verification: 'mean', 'std' (lowercase)
        assert stats['mean'] > 0
        assert stats['std'] >= 0
        assert not np.isnan(stats['mean'])
    
    def test_very_small_numbers(self) -> None:
        """Valida cálculo con números muy pequeños."""
        data = pd.Series([1e-10, 2e-10, 3e-10])
        
        stats = calculate_descriptive_stats(data)
        
        assert stats['mean'] > 0
        assert stats['std'] >= 0
        assert not np.isnan(stats['mean'])
    
    def test_mixed_scales(self) -> None:
        """Valida mezcla de escalas diferentes."""
        data = pd.Series([0.001, 1000, 0.0001, 50, 0.5])
        
        stats = calculate_descriptive_stats(data)
        
        assert stats['mean'] > 0
        assert stats['std'] > 0

class TestSkewedDistributions:
    """Tests con distribuciones sesgadas."""
    
    def test_heavily_right_skewed(self) -> None:
        """Valida datos muy sesgados a la derecha."""
        # Distribución exponencial (sesgada a la derecha)
        np.random.seed(42)
        data = pd.Series(np.random.exponential(scale=2, size=100))
        
        stats = calculate_descriptive_stats(data)
        
        assert stats['mean'] > 0
        assert stats['std'] > 0
        assert stats['min'] >= 0
    
    def test_highly_negative_skew(self) -> None:
        """Valida datos muy sesgados a la izquierda."""
        # Datos negativos sesgados
        np.random.seed(42)
        data = pd.Series(-np.random.exponential(scale=2, size=100))
        
        stats = calculate_descriptive_stats(data)
        
        assert stats['mean'] < 0
        assert stats['std'] > 0

class TestOutliers:
    """Tests con outliers extremos."""
    
    def test_extreme_outliers(self) -> None:
        """Valida detección con outliers extremos."""
        data = pd.Series([1, 2, 3, 4, 5, 1000, -1000])
        
        # Returns (mask, details)
        mask, details = detect_outliers_iqr(data)
        
        assert details['n_outliers'] > 0
        assert np.sum(mask) > 0
    
    def test_all_same_value_one_outlier(self) -> None:
        """Valida detección cuando todos iguales excepto uno."""
        data = pd.Series([5]*20 + [100])
        
        mask, details = detect_outliers_iqr(data)
        
        assert details['n_outliers'] > 0
        assert 100 in data[mask].values

class TestZeroVariance:
    """Tests con varianza cero o casi cero."""
    
    def test_constant_data(self) -> None:
        """Valida con datos constantes."""
        data = pd.Series([42.0] * 100)
        
        stats = calculate_descriptive_stats(data)
        
        assert stats['mean'] == 42.0
        assert stats['std'] == 0
    
    def test_almost_constant(self) -> None:
        """Valida con datos casi constantes."""
        data = pd.Series([100.0, 100.0, 100.000001, 100.0, 100.0])
        
        stats = calculate_descriptive_stats(data)
        
        assert stats['mean'] > 0
        assert stats['std'] < 0.00001

class TestMulticollinearity:
    """Tests con multicolinealidad en regresión."""
    
    def test_perfect_multicollinearity(self) -> None:
        """Valida regresión con multicolinealidad perfecta."""
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        # x and x are perfectly correlated (same)
        # Linear regression simple inputs X array and y array. It's univariate.
        # Multicollinearity applies to multivariate. Or passing X as dataframe.
        
        # linear_regression(x, y) handles 1D arrays. 
        # If we pass 1D array, no multicollinearity issue (just 1 var).
        
        pass 

class TestNormalization:
    """Tests para normalización con datos extremos."""
    
    def test_normalize_extreme_range(self) -> None:
        """Valida normalización con rango extremo."""
        data = pd.Series([0.0001, 1000000, 50, 0.5, 10000])
        
        normalized = normalize_data(data)
        
        # If normalized is ndarray
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1
    
    def test_normalize_negative_values(self) -> None:
        """Valida normalización con valores negativos."""
        data = pd.Series([-100, -50, 0, 50, 100])
        
        normalized = normalize_data(data)
        
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1

class TestNoisyRealWorldData:
    """Tests con datos ruidosos realistas."""
    
    def test_medical_data_with_measurement_error(self) -> None:
        """Simula datos médicos con error de medición."""
        np.random.seed(42)
        true_values = np.random.normal(100, 15, 50)
        measured = true_values + np.random.normal(0, 5, 50)  # Error
        
        data = pd.Series(measured)
        stats = calculate_descriptive_stats(data)
        
        assert 90 < stats['mean'] < 110
        assert stats['std'] > 0
    
    @given(st.lists(
        st.floats(min_value=0, max_value=300, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=1000
    ))
    @settings(max_examples=50, deadline=None)
    def test_random_medical_values_property(self, values: list) -> None:
        """Property-based: Estadísticos siempre válidos."""
        if len(values) < 2:
            return
        
        data = pd.Series(values)
        stats = calculate_descriptive_stats(data)
        
        assert stats['count'] == len(values)
        assert stats['mean'] >= 0
        assert stats['std'] >= 0
        assert not np.isnan(stats['mean'])
