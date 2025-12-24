"""
Módulo de Psicometría y Análisis de Escalas (BioStat Easy)
----------------------------------------------------------
Este módulo proporciona herramientas para el análisis de propiedades psicométricas
de instrumentos de medición (encuestas, tests), incluyendo confiabilidad, validez
y análisis de ítems.

Responsabilidades:
- Confiabilidad: Alpha de Cronbach, Test-retest, Split-half.
- Validez: Convergente, Discriminante.
- Análisis de Ítems: Dificultad, Discriminación, Correlación Ítem-Total.

Autor: BioStat Easy Team
Versión: 2.5
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Union, Optional, List, Any, Tuple

# ==============================================================================
# 1. CONFIABILIDAD
# ==============================================================================

def cronbach_alpha(items_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula el coeficiente Alpha de Cronbach para confiabilidad interna.
    """
    items = items_df.select_dtypes(include=[np.number]).dropna()
    n_items = items.shape[1]
    
    if n_items < 2:
        return {"error": "Se requieren al menos 2 ítems."}
        
    # Varianza de cada ítem
    item_vars = items.var(ddof=1) # Usar varianza muestral
    total_score = items.sum(axis=1)
    # Varianza del puntaje total
    total_var = total_score.var(ddof=1)
    
    if total_var == 0:
        return {"error": "Varianza total es 0 (todas las respuestas idénticas)."}
        
    # Formula: (k / (k-1)) * (1 - (Sum(Var_i) / Var_t))
    alpha = (n_items / (n_items - 1)) * (1 - (item_vars.sum() / total_var))
    
    # Interpretación
    if alpha < 0.5: interp = "Inaceptable"
    elif alpha < 0.6: interp = "Pobre"
    elif alpha < 0.7: interp = "Cuestionable"
    elif alpha < 0.8: interp = "Aceptable"
    elif alpha < 0.9: interp = "Bueno"
    else: interp = "Excelente"
    
    return {
        "alpha": float(alpha),
        "n_items": int(n_items),
        "n_obs": len(items),
        "interpretation": interp,
        "is_reliable": alpha >= 0.7
    }


def test_retest_reliability(X1: Union[pd.Series, np.ndarray], 
                          X2: Union[pd.Series, np.ndarray], 
                          metodo: str = 'pearson') -> Dict[str, Any]:
    """
    Calcula confiabilidad test-retest (estabilidad temporal).
    Correlación entre puntajes en momento 1 (X1) y momento 2 (X2).
    """
    df = pd.DataFrame({'t1': X1, 't2': X2}).dropna()
    
    if len(df) < 2:
        return {"error": "Datos insuficientes."}
    
    if metodo == 'pearson':
        stat, p = stats.pearsonr(df['t1'], df['t2'])
    elif metodo == 'spearman':
        stat, p = stats.spearmanr(df['t1'], df['t2'])
    else:
        return {"error": "Método desconocido."}
        
    icc_value = stat # Simple approximation via correlation
    
    return {
        "reliability_coeff": float(stat),
        "p_value": float(p),
        "method": f"Test-Retest ({metodo})",
        "interpretation": "Estable" if stat > 0.7 else "Poca estabilidad"
    }


def split_half_reliability(X: pd.DataFrame, 
                         metodo: str = 'spearman') -> Dict[str, Any]:
    """
    Calcula confiabilidad por mitades (Split-Half) con corrección Spearman-Brown.
    """
    items = X.select_dtypes(include=[np.number]).dropna()
    n_items = items.shape[1]
    
    if n_items < 2: return {"error": "Mínimo 2 ítems."}
    
    # Dividir en pares e impares
    odd_items = items.iloc[:, ::2]
    even_items = items.iloc[:, 1::2]
    
    score_odd = odd_items.sum(axis=1)
    score_even = even_items.sum(axis=1)
    
    # Correlación entre mitades
    r_half, p = stats.pearsonr(score_odd, score_even)
    
    # Corrección Spearman-Brown: 2r / (1 + r)
    r_sb = (2 * r_half) / (1 + r_half)
    
    return {
        "split_half_coeff": float(r_sb),
        "correlation_halves": float(r_half),
        "interpretation": "Consistente" if r_sb > 0.7 else "Baja consistencia"
    }


# ==============================================================================
# 2. VALIDEZ
# ==============================================================================

def item_total_correlation(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validez interna: Correlación de cada ítem con el puntaje total (corregido).
    Corregido significa que se excluye el ítem evaluado de la suma total.
    """
    items = items_df.select_dtypes(include=[np.number]).dropna()
    
    results = []
    
    total_scores = items.sum(axis=1)
    
    for col in items.columns:
        # Puntuación corregida (Total - Este ítem)
        corrected_total = total_scores - items[col]
        
        # Correlación polyserial o simple pearson
        r, p = stats.pearsonr(items[col], corrected_total)
        
        results.append({
            "Item": col,
            "Item-Total Corr": r,
            "P-Value": p,
            "Keep": "Keep" if r > 0.3 else "Check/Drop"
        })
        
    return pd.DataFrame(results).set_index("Item")


def convergent_validity(X: Union[pd.Series, np.ndarray], 
                      Y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Validez convergente: Correlación con un "Gold Standard" o constructo similar (Y).
    """
    df = pd.DataFrame({'Scale': X, 'GoldStandard': Y}).dropna()
    
    r, p = stats.pearsonr(df['Scale'], df['GoldStandard'])
    
    return {
        "validity_type": "Convergent",
        "coefficient": float(r),
        "p_value": float(p),
        "interpretation": "Validez Convergente Fuerte" if r > 0.5 and p < 0.05 else "Baja convergencia"
    }


def discriminant_validity(X: Union[pd.Series, np.ndarray], 
                        Y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Validez discriminante: Correlación con un constructo teóricamente diferente (Y).
    Se espera correlación baja.
    """
    df = pd.DataFrame({'Scale': X, 'DifferentConstruct': Y}).dropna()
    
    r, p = stats.pearsonr(df['Scale'], df['DifferentConstruct'])
    
    return {
        "validity_type": "Discriminant",
        "coefficient": float(r),
        "p_value": float(p),
        "interpretation": "Validez Discriminante OK" if abs(r) < 0.3 else "Problema: Correlación alta con constructo diferente"
    }


# ==============================================================================
# 3. ANÁLISIS DE ÍTEMS
# ==============================================================================

def item_difficulty(respuestas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula índice de dificultad (p) para ítems binarios o politómicos.
    Para binarios: Proporción de aciertos (1s).
    Para politómicos: Promedio / Max_Score.
    
    Aquí asumiremos una implementación genérica basada en promedios.
    """
    items = respuestas_df.select_dtypes(include=[np.number]).dropna()
    
    results = []
    
    for col in items.columns:
        mean_val = items[col].mean()
        max_val = items[col].max()
        min_val = items[col].min()
        
        # Dificuldad normalizada 0-1 (si min es 0 y max conocido)
        # Si es binario (0,1), mean = p-value dificultad
        range_val = max_val - min_val if (max_val - min_val) > 0 else 1
        
        diff_index = (mean_val - min_val) / range_val
        
        # Interpretación clásica de "p" (proporción de aciertos)
        # p alto = muy fácil
        # p bajo = muy difícil
        if diff_index > 0.8: diff_str = "Muy Fácil"
        elif diff_index < 0.2: diff_str = "Muy Difícil"
        else: diff_str = "Moderada (Óptima)"
        
        results.append({
            "Item": col,
            "Mean": mean_val,
            "Difficulty Index (p)": diff_index,
            "Interpretation": diff_str
        })
        
    return pd.DataFrame(results).set_index("Item")


def item_discrimination(respuestas_df: pd.DataFrame, 
                      criterio: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Calcula discriminación usando el método de grupos extremos (Superior 27% vs Inferior 27%).
    D = p_superior - p_inferior
    """
    items = respuestas_df.select_dtypes(include=[np.number]).dropna()
    
    if criterio is None:
        criterio = items.sum(axis=1) # Puntaje total como criterio interno
        
    df = items.copy()
    df['Score'] = criterio
    
    # Cuartiles (o 27%)
    n = len(df)
    n_cutoff = int(n * 0.27)
    
    if n_cutoff < 1:
         return pd.DataFrame({"Error": "N insuficiente para grupos extremos."})
         
    df_sorted = df.sort_values('Score', ascending=False)
    
    upper_group = df_sorted.head(n_cutoff)
    lower_group = df_sorted.tail(n_cutoff)
    
    results = []
    
    for col in items.columns:
        # Asumiendo escalas 0-1 para simplificar explicación D, o normalizando
        # D = Mean_Upper_Norm - Mean_Lower_Norm
        
        max_v = items[col].max()
        min_v = items[col].min()
        rng = max_v - min_v if (max_v - min_v) > 0 else 1
        
        mean_upper = (upper_group[col].mean() - min_v) / rng
        mean_lower = (lower_group[col].mean() - min_v) / rng
        
        D = mean_upper - mean_lower
        
        if D > 0.4: eval_d = "Excelente"
        elif D > 0.3: eval_d = "Buena"
        elif D > 0.2: eval_d = "Regular"
        else: eval_d = "Pobre (Revisar)"
        
        results.append({
            "Item": col,
            "Discrimination (D)": D,
            "Evaluation": eval_d
        })
        
    return pd.DataFrame(results).set_index("Item")


def item_analysis(items_df: pd.DataFrame, 
                 criterion_var: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Reporte completo integrando Dificultad, Discriminación y Correlación It-Tot.
    """
    df_diff = item_difficulty(items_df)
    df_disc = item_discrimination(items_df, criterion_var)
    df_corr = item_total_correlation(items_df)
    
    # Merge all
    full_analysis = pd.concat([df_diff, df_disc, df_corr], axis=1)
    
    # Limpiar columnas duplicadas si las hay
    full_analysis = full_analysis.loc[:,~full_analysis.columns.duplicated()]
    
    return full_analysis
