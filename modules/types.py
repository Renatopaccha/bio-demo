"""Definiciones de tipos para aplicación BioStat Easy."""

# ─────────────────────────────────────────
# SECCIÓN 1: IMPORTS
# ─────────────────────────────────────────
from typing import (
    Union, 
    List, 
    Tuple, 
    Dict, 
    Any, 
    Optional, 
    Callable, 
    TypeVar
)
import pandas as pd
import numpy as np

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# ─────────────────────────────────────────
# SECCIÓN 2: TYPE VARIABLES
# ─────────────────────────────────────────
# Variables genéricas para parametrizar funciones
T = TypeVar('T')  # Tipo genérico básico
NumT = TypeVar('NumT', int, float, np.number)  # Limitado a tipos numéricos
DF_Type = TypeVar('DF_Type', pd.DataFrame, np.ndarray)  # Estructuras de datos tabulares
DataType = TypeVar('DataType', bound=Union[pd.DataFrame, pd.Series])  # Datos tipo pandas

# ─────────────────────────────────────────
# SECCIÓN 3: NUMERIC TYPES
# ─────────────────────────────────────────
# Alias para valores y arrays numéricos
NumericType = Union[int, float, np.integer, np.floating]
"""Valor numérico escalar (int, float o tipos numpy)."""

NumericValue = Union[int, float]
"""Valor numérico simple (nativo Python)."""

NumericArray = Union[np.ndarray, List[Union[int, float]]]
"""Colección de valores numéricos."""

# ─────────────────────────────────────────
# SECCIÓN 4: SERIES/DATAFRAME TYPES
# ─────────────────────────────────────────
# Alias para estructuras de datos flexibles
SeriesLike = Union[pd.Series, np.ndarray, List, Dict]
"""Objeto tipo Serie (1D array-like)."""

DataFrameLike = Union[pd.DataFrame, Dict[str, SeriesLike]]
"""Objeto tipo DataFrame (2D array-like o dict de series)."""

ArrayLike = Union[np.ndarray, List, pd.Series]
"""Objeto tipo Array (secuencial)."""

# ─────────────────────────────────────────
# SECCIÓN 5: CUSTOM ALIASES (ESPECÍFICO BIOESTADÍSTICA)
# ─────────────────────────────────────────
# Tipos semánticos para el dominio de la aplicación
StatResult = Dict[str, Any]
"""Resultado de prueba estadística (diccionario con estadísticos, p-values, etc)."""

GroupData = Dict[str, SeriesLike]
"""Grupos de datos organizados por nombre de grupo."""

ContinuousData = SeriesLike
"""Datos de variable continua (ej: edad, peso)."""

CategoricalData = Union[pd.Series, List[str], np.ndarray]
"""Datos de variable categórica (ej: género, tratamiento)."""

HypothesisResult = Tuple[float, float, str]
"""Resultado simple: (estadístico, p-value, conclusión)."""

"""
EJEMPLO DE USO:

from modules.types import SeriesLike, StatResult

def calculate_stats(data: SeriesLike) -> StatResult:
    return {'mean': 0.0, 'sd': 1.0}
"""
