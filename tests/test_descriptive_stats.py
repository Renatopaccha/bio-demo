import pytest
import pandas as pd
import numpy as np
from modules.stats.descriptive import (
    validate_data_input,
    calculate_descriptive_stats,
    analyze_missing_values,
    detect_outliers_iqr,
    normalize_data,
)

class TestValidateDataInput:
    """Tests para validación de entrada de datos."""
    
    def test_validate_valid_dataframe(self, simple_medical_data: pd.DataFrame) -> None:
        """Valida que DataFrame válido retorna True."""
        is_valid, msg = validate_data_input(simple_medical_data)
        assert is_valid is True
    
    def test_validate_valid_series(self, simple_medical_data: pd.DataFrame) -> None:
        """Valida que Series válida retorna True."""
        is_valid, msg = validate_data_input(simple_medical_data['age'])
        assert is_valid is True
    
    def test_validate_empty_dataframe_raises(self, empty_data: pd.DataFrame) -> None:
        """Valida que DataFrame vacío retorna False (según implementación actual)."""
        # validate_data_input returns (False, error_msg) for invalid
        is_valid, msg = validate_data_input(empty_data)
        assert is_valid is False
        assert "vacío" in msg.lower() or "empty" in msg.lower()
    
    def test_validate_none_returns_false(self) -> None:
        """Valida que None retorna False."""
        is_valid, msg = validate_data_input(None)
        assert is_valid is False

class TestCalculateDescriptiveStats:
    """Tests para cálculo de estadísticos descriptivos."""
    
    def test_descriptive_stats_basic(self, simple_medical_data: pd.DataFrame) -> None:
        """Valida cálculo básico de estadísticos."""
        stats = calculate_descriptive_stats(simple_medical_data['age'])
        
        # Validaciones básicas
        assert 'count' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert stats['count'] == 100
        assert 30 < stats['mean'] < 60  # Normal ~45 from fixture (mean 45, std 15)
        assert stats['std'] > 0
    
    def test_descriptive_stats_with_nan(self, medical_data_with_nan: pd.DataFrame) -> None:
        """Valida manejo de NaN en estadísticos."""
        stats = calculate_descriptive_stats(medical_data_with_nan['age'])
        
        # Con 10% NaN, count debe ser 90
        # s.describe() returns count of non-null values
        # Implementation in Prompt 2 handles dropna inside
        assert stats['count'] < 100
        assert stats['count'] == 90
    
    def test_descriptive_stats_single_value(self, single_value_data: pd.Series) -> None:
        """Valida estadísticos con un solo valor."""
        stats = calculate_descriptive_stats(single_value_data)
        
        assert stats['count'] == 1
        assert stats['mean'] == 42.0
        # std is NaN for single value in pandas default ddof=1
        assert np.isnan(stats['std']) 

class TestAnalyzeMissingValues:
    """Tests para análisis de valores faltantes."""
    
    def test_missing_values_no_nan(self, simple_medical_data: pd.DataFrame) -> None:
        """Valida que sin NaN, total_missing_count es 0."""
        result = analyze_missing_values(simple_medical_data)
        
        # Valid keys from implementation: total_missing_count, total_missing_pct
        assert result['total_missing_count'] == 0
        assert result['total_missing_pct'] == 0.0
    
    def test_missing_values_with_nan(self, medical_data_with_nan: pd.DataFrame) -> None:
        """Valida que con NaN, se cuenta correctamente."""
        result = analyze_missing_values(medical_data_with_nan)
        
        assert result['total_missing_count'] > 0
        assert 0 < result['total_missing_pct'] < 100
        assert 'by_column' in result

class TestDetectOutliersIQR:
    """Tests para detección de outliers con IQR."""
    
    def test_detect_outliers_with_outliers(self, medical_data_with_outliers: pd.DataFrame) -> None:
        """Valida detección de outliers cuando existen."""
        # Returns (boolean_mask, details_dict)
        mask, details = detect_outliers_iqr(medical_data_with_outliers['age'])
        
        assert details['n_outliers'] > 0
        assert np.sum(mask) > 0 # At least one true
        
        # Verify specific outlier value (200 is in the fixture)
        outlier_values = medical_data_with_outliers['age'][mask].values
        assert 200 in outlier_values
    
    def test_detect_outliers_no_outliers(self, simple_medical_data: pd.DataFrame) -> None:
        """Valida que sin outliers, retorna pocos o ninguno."""
        mask, details = detect_outliers_iqr(simple_medical_data['age'])
        
        # details['n_outliers'] should be small
        assert details['n_outliers'] <= 5

class TestNormalizeData:
    """Tests para normalización de datos."""
    
    def test_normalize_creates_unit_variance(self, simple_medical_data: pd.DataFrame) -> None:
        """Valida que normalización crea media ~0 y SD ~1."""
        # Pass Series, not DataFrame
        series = simple_medical_data['age']
        normalized = normalize_data(series)
        
        # Result is np.ndarray
        assert isinstance(normalized, np.ndarray)
        assert abs(normalized.mean()) < 0.01  # Media cercana a 0
        assert abs(normalized.std() - 1.0) < 0.01  # SD cercana a 1
    
    def test_normalize_preserves_length(self, simple_medical_data: pd.DataFrame) -> None:
        """Valida que normalización preserva longitud."""
        series = simple_medical_data['age']
        normalized = normalize_data(series)
        
        assert len(normalized) == len(series)
