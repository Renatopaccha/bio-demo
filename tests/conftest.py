import pytest
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Generator
from modules.types import SeriesLike, DataFrameLike

# FIXTURES: Datos de prueba médicos simples
@pytest.fixture
def simple_medical_data() -> pd.DataFrame:
    """Datos médicos simples: edad y glucosa sin valores faltantes."""
    np.random.seed(42)  # Ensure reproducibility locally
    return pd.DataFrame({
        'patient_id': range(1, 101),
        'age': np.random.normal(45, 15, 100),
        'glucose': np.random.normal(100, 20, 100),
        'systolic_bp': np.random.normal(120, 15, 100),
        'diastolic_bp': np.random.normal(80, 10, 100),
    })

@pytest.fixture
def medical_data_with_nan() -> pd.DataFrame:
    """Datos médicos con valores faltantes (simula datos reales)."""
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.normal(45, 15, 100),
        'glucose': np.random.normal(100, 20, 100),
    })
    # Agregar NaN en posiciones aleatorias
    data.loc[np.random.choice(100, 10, replace=False), 'age'] = np.nan
    data.loc[np.random.choice(100, 15, replace=False), 'glucose'] = np.nan
    return data

@pytest.fixture
def medical_data_with_outliers() -> pd.DataFrame:
    """Datos médicos con outliers (casos extremos)."""
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.concatenate([
            np.random.normal(45, 15, 95),
            np.array([120, 150, -10, 200, 180])  # Outliers
        ])
    })
    return data

@pytest.fixture
def two_groups_data() -> Tuple[pd.Series, pd.Series]:
    """Dos grupos para pruebas de comparación."""
    np.random.seed(42)
    grupo1 = pd.Series(np.random.normal(100, 15, 50), name='control')
    grupo2 = pd.Series(np.random.normal(110, 15, 50), name='treatment')
    return grupo1, grupo2

@pytest.fixture
def empty_data() -> pd.DataFrame:
    """DataFrame vacío para validación de error handling."""
    return pd.DataFrame()

@pytest.fixture
def single_value_data() -> pd.Series:
    """Serie con un solo valor."""
    return pd.Series([42.0])

# FIXTURES: Configuración
@pytest.fixture(scope="session")
def random_seed() -> int:
    """Semilla para reproducibilidad."""
    return 42
