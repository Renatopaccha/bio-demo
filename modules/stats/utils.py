"""
Módulo de Utilidades Estadísticas (BioStat Easy)
------------------------------------------------
Este módulo proporciona funciones auxiliares para limpieza de datos, 
formateo de resultados, interpretaciones textuales y cálculos estadísticos comunes.
Son funciones 'stateless' y puras diseñadas para ser reutilizadas.

Responsabilidades:
- Limpieza y tratamiento básico (Outliers, NaNs).
- Normalización y Estandarización manual.
- Formateo de p-valor, intervalos y estadísticos.
- Interpretación automática de tamaños de efecto.
- Cálculos manuables (Cohen's d, SEM, etc.).

Autor: BioStat Easy Team
Versión: 2.5
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Optional, Any

# ==============================================================================
# 1. LIMPIEZA Y TRATAMIENTO DE DATOS
# ==============================================================================

def handle_missing_values(data_series: pd.Series, method: str = 'drop') -> pd.Series:
    """Manejo básico de valores faltantes."""
    # Nota: Esta lógica existe en core.py más robusta, aquí versión ligera utilitaria
    if method == 'drop':
        return data_series.dropna()
    elif method == 'mean':
        return data_series.fillna(data_series.mean())
    elif method == 'median':
        return data_series.fillna(data_series.median())
    elif method == 'mode':
        return data_series.fillna(data_series.mode()[0])
    return data_series


def remove_outliers(data: pd.Series, method: str = 'iqr', multiplier: float = 1.5) -> pd.Series:
    """Elimina outliers de una serie (retorna serie filtrada)."""
    data_clean = pd.to_numeric(data, errors='coerce').dropna()
    
    if method == 'iqr':
        Q1 = data_clean.quantile(0.25)
        Q3 = data_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        return data_clean[(data_clean >= lower) & (data_clean <= upper)]
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data_clean))
        return data_clean[z_scores < 3] # Standard 3 sigma
        
    return data # Default return


def normalize_data(data: pd.Series) -> pd.Series:
    """Escalamiento Min-Max (0-1)."""
    d = pd.to_numeric(data, errors='coerce')
    return (d - d.min()) / (d.max() - d.min())


def standardize_data(data: pd.Series) -> pd.Series:
    """Estandarización Z (Media 0, SD 1)."""
    d = pd.to_numeric(data, errors='coerce')
    return (d - d.mean()) / d.std()


def robust_scale(data: pd.Series) -> pd.Series:
    """Escalamiento robusto usando Mediana e IQR."""
    d = pd.to_numeric(data, errors='coerce')
    q1 = d.quantile(0.25)
    q3 = d.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0: return d - d.median()
    return (d - d.median()) / iqr


def trim_extreme_values(data: pd.Series, percentile: float = 0.05) -> pd.Series:
    """Recorta valores extremos (Winsorizing o Truncating). Aquí truncating."""
    lower = data.quantile(percentile)
    upper = data.quantile(1 - percentile)
    return data[(data >= lower) & (data <= upper)]


# ==============================================================================
# 2. FORMATO DE SALIDA (PRESENTACIÓN)
# ==============================================================================

def format_p_value(p_value: float, decimals: int = 4) -> str:
    """Formatea p-valor (ej: <0.0001)."""
    if pd.isna(p_value): return "NaN"
    if p_value < 0.0001:
        return "<0.0001"
    return f"{p_value:.{decimals}f}"


def format_ci(lower: float, upper: float, decimals: int = 2) -> str:
    """Formatea intervalo de confianza."""
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def format_statistic(stat_value: float, decimals: int = 4) -> str:
    """Formatea estadístico genérico."""
    if pd.isna(stat_value): return "-"
    return f"{stat_value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Formatea porcentaje."""
    # Asume value 0-100 si es > 1, o 0-1 ?? Asumiremos input es 0-100 segun uso común stats
    # O input es ratio 0-1. Estandaricemos:
    # Si viene 0.85 -> 85.0%? O si viene 85 -> 85.0%?
    # Para seguridad, asumiremos que el input YA ES el número a mostrar (ej 85.4)
    return f"{value:.{decimals}f}%"


# ==============================================================================
# 3. INTERPRETACIONES CLÍNICAS/ESTADÍSTICAS
# ==============================================================================

def interpret_effect_size(cohens_d: float) -> str:
    """Interpreta d de Cohen."""
    d = abs(cohens_d)
    if d < 0.2: return "Despreciable"
    if d < 0.5: return "Pequeño"
    if d < 0.8: return "Mediano"
    if d < 1.2: return "Grande"
    return "Muy Grande"


def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """Interpreta significancia estadística."""
    if p_value < alpha:
        return "Estadísticamente Significativo"
    return "No Significativo"


def interpret_correlation(r_value: float) -> str:
    """Interpreta magnitud de correlación (Pearson/Spearman)."""
    r = abs(r_value)
    if r < 0.1: return "Nula"
    if r < 0.3: return "Débil"
    if r < 0.5: return "Moderada"
    if r < 0.7: return "Fuerte"
    return "Muy Fuerte"


# ==============================================================================
# 4. CÁLCULOS ESTADÍSTICOS COMUNES
# ==============================================================================

def calculate_cohens_d(group1: Any, group2: Any) -> float:
    """Calcula d de Cohen para 2 grupos independientes."""
    g1 = np.array(group1)
    g2 = np.array(group2)
    
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def calculate_hedges_g(group1: Any, group2: Any, n1: int, n2: int) -> float:
    """Calcula g de Hedges (Cohen's d corregida para N pequeño)."""
    d = calculate_cohens_d(group1, group2)
    # Corrección aproximada: 1 - (3 / (4(n1+n2)-9))
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction


def calculate_cramers_v(chi2_stat: float, n: int, min_dim: int) -> float:
    """Calcula V de Cramer."""
    if n == 0 or min_dim == 0: return 0.0
    return np.sqrt(chi2_stat / (n * min_dim))


def calculate_sem(data: Any) -> float:
    """Calcula Error Estándar de la Media."""
    return stats.sem(data)


# ==============================================================================
# 5. HELPERS GENERALES
# ==============================================================================

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Retorna lista de columnas numéricas."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Retorna lista de columnas categóricas (objeto/category)."""
    return df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """División segura evitando ZeroDivisionError."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default
        
