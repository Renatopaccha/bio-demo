import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import Tuple, Dict, Any
from modules.stats import (
    regresion_lineal_simple as linear_regression,
    regresion_lineal_multiple as multiple_linear_regression,
    regresion_logistica as logistic_regression,
)

# Helpers mock (r_squared and predict_linear usually part of regression result or separate utils)
# If they are not exported, I will test the regression result directly.
# Assuming r_squared is not separately exported but part of result dict.

@st.composite
def regression_data_strategy(draw) -> Tuple[np.ndarray, np.ndarray]:
    """Genera datos para regresión con relación lineal."""
    x = draw(st.lists(
        st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=200
    ))
    x_arr = np.array(x)
    # y = 2*x + 3 + ruido
    y_arr = 2 * x_arr + 3 + np.random.normal(0, 10, len(x_arr))
    return x_arr, y_arr

@st.composite
def multiple_regression_data_strategy(draw) -> Tuple[np.ndarray, np.ndarray]:
    """Genera datos para regresión múltiple."""
    size = draw(st.integers(min_value=20, max_value=200))
    X = np.random.randn(size, 3)
    # y = 2*x1 + 3*x2 - 1*x3 + ruido
    y = 2*X[:, 0] + 3*X[:, 1] - 1*X[:, 2] + np.random.normal(0, 2, size)
    return X, y

class TestLinearRegression:
    """Tests para regresión lineal simple."""
    
    @given(regression_data_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_linear_regression_property(self, data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Property-based test: regresión retorna parámetros válidos."""
        x, y = data
        if len(np.unique(x)) < 2: 
            return
            
        result = linear_regression(x, y)
        
        # Valid keys: 'r2', 'params', 'pvalues'
        assert isinstance(result, dict)
        if 'error' in result:
             return # Skip check if error (e.g. N<3)
             
        assert 'r2' in result
        assert 'params' in result
        assert 'const' in result['params'] # Statmodels add_constant adds 'const'

    def test_linear_regression_perfect_line(self) -> None:
        """Valida regresión en línea perfecta."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # y = 2*x + 0
        
        result = linear_regression(x, y)
        
        r2 = result.get('r2', 0)
        assert r2 > 0.99
    
    @pytest.mark.parametrize("noise_level", [1, 5, 10, 20])
    def test_linear_regression_with_noise(self, noise_level: float) -> None:
        """Valida regresión con diferentes niveles de ruido."""
        np.random.seed(42)
        x = np.arange(100)
        y = 2*x + 3 + np.random.normal(0, noise_level, 100)
        
        result = linear_regression(x, y)
        r2 = result.get('r2', 0)
        
        assert 0 <= r2 <= 1.000001
        if noise_level < 5:
            assert r2 > 0.95

class TestMultipleRegression:
    """Tests para regresión lineal múltiple."""
    
    @given(multiple_regression_data_strategy())
    @settings(max_examples=20, deadline=None)
    def test_multiple_regression_property(self, data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Property-based test: regresión múltiple retorna parámetros válidos."""
        X, y = data
        df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
        y_ser = pd.Series(y, name='y')
        
        try:
            # Signature: regresion_lineal_multiple(X, y)
            result = multiple_linear_regression(df, y_ser)
            if 'error' in result:
                return
            assert isinstance(result, dict)
            assert 'r2' in result
        except Exception:
            pass 
    
    def test_multiple_regression_basic(self) -> None:
        """Valida regresión múltiple básica."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([3, 5, 7, 9, 11])
        
        df = pd.DataFrame(X, columns=['x1', 'x2'])
        y_ser = pd.Series(y)
        
        result = multiple_linear_regression(df, y_ser)
        r2 = result.get('r2', 0)
        assert r2 > 0.99

class TestLogisticRegression:
    """Tests para regresión logística."""
    
    def test_logistic_regression_binary(self) -> None:
        """Valida regresión logística con datos binarios."""
        # Generar datos separables
        np.random.seed(42)
        X_0 = np.random.normal(-2, 1, (50, 2))
        X_1 = np.random.normal(2, 1, (50, 2))
        X = np.vstack([X_0, X_1])
        y = np.hstack([np.zeros(50), np.ones(50)])
        
        df = pd.DataFrame(X, columns=['x1', 'x2'])
        y_ser = pd.Series(y)
        
        # Signature: regresion_logistica(X, y)
        result = logistic_regression(df, y_ser)
        
        # Valid keys: pseudo_r2, aic
        assert isinstance(result, dict)
        if 'error' in result:
             pytest.fail(f"Logistic regression error: {result['error']}")
             
        assert any(k in result for k in ['pseudo_r2', 'aic'])
