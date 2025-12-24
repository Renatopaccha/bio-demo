import pytest
import pandas as pd
import numpy as np
from modules import stats

class TestStatsImports:
    def test_stats_module_imports(self) -> None:
        """Verifica que el módulo stats importa sin errores."""
        assert hasattr(stats, '__doc__')
    
    def test_descriptive_functions_available(self) -> None:
        """Verifica que funciones descriptivas están disponibles (Prompt 2)."""
        assert hasattr(stats, 'validate_data_input')
        assert hasattr(stats, 'calculate_descriptive_stats')
    
    def test_hypothesis_test_functions_available(self) -> None:
        """Verifica que funciones de pruebas están disponibles (Prompt 3)."""
        assert hasattr(stats, 'ttest_independiente')
        assert hasattr(stats, 'mann_whitney_u')
    
    def test_regression_functions_available(self) -> None:
        """Verifica que funciones de regresión están disponibles (Prompt 4)."""
        assert hasattr(stats, 'regresion_lineal_simple')
        assert hasattr(stats, 'pca_analysis')
    
    def test_survival_functions_available(self) -> None:
        """Verifica que funciones de supervivencia están disponibles (Prompt 5)."""
        assert hasattr(stats, 'kaplan_meier_estimator')
        assert hasattr(stats, 'cronbach_alpha')

class TestStatsBasicCalls:
    def test_validate_data_input_with_valid_data(self) -> None:
        """Prueba validación con datos válidos."""
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        # validate_data_input returns (bool, str)
        is_valid, msg = stats.validate_data_input(data)
        assert is_valid is True
    
    def test_validate_data_input_with_invalid_data(self) -> None:
        """Prueba validación con datos inválidos."""
        # validate_data_input returns (False, msg) for invalid input, doesn't raise
        is_valid, msg = stats.validate_data_input(None)
        assert is_valid is False

    def test_calculate_descriptive_stats_smoke(self) -> None:
        """Prueba smoke de estadística descriptiva."""
        data = pd.Series([1, 2, 3, 4, 5])
        res = stats.calculate_descriptive_stats(data)
        # s.describe() returns 'count', not 'n'
        assert res['count'] == 5
        assert res['mean'] == 3.0
