"""
Módulo de Análisis de Supervivencia (BioStat Easy)
--------------------------------------------------
Este módulo implementa técnicas estadísticas para datos de tiempo hasta evento (censurados).
Incluye estimadores no paramétricos (Kaplan-Meier), pruebas de hipótesis (Log-rank),
y modelos de regresión semiparamétricos (Cox) y paramétricos (Weibull).

Responsabilidades:
- Estimación Kaplan-Meier
- Regresión de Cox (Proportional Hazards)
- Prueba de Log-rank
- Modelo Weibull
- Verificación de datos de supervivencia

Autor: BioStat Easy Team
Versión: 2.5
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, List, Any, Tuple
import matplotlib.pyplot as plt

# Importación robusta de lifelines (Librería estándar para supervivencia en Python)
LIFELINES_AVAILABLE = False
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullFitter
    from lifelines.statistics import logrank_test as lifelines_logrank
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    pass


# ==============================================================================
# 1. VERIFICACIÓN DE DATOS (UTILIDADES)
# ==============================================================================

def verify_survival_data(df: pd.DataFrame, 
                        col_tiempo: str, 
                        col_evento: str) -> Tuple[bool, str]:
    """
    Verifica si los datos son aptos para análisis de supervivencia.
    """
    if df is None or df.empty:
        return False, "DataFrame vacío."
        
    if col_tiempo not in df.columns or col_evento not in df.columns:
        return False, f"Faltan columnas: {col_tiempo}, {col_evento}."
        
    # Verificar Tiempo (Debe ser numérico y >= 0)
    try:
        times = pd.to_numeric(df[col_tiempo], errors='coerce')
        if times.isna().any():
            return False, "La columna de tiempo contiene valores no numéricos."
        if (times < 0).any():
            return False, "La columna de tiempo contiene valores negativos."
    except:
        return False, "Error validando columna de tiempo."
        
    # Verificar Evento (Debe ser binario 0/1 o True/False)
    try:
        events = pd.to_numeric(df[col_evento], errors='coerce').dropna()
        uniques = events.unique()
        if not set(uniques).issubset({0, 1}):
             return False, f"La columna de evento debe ser binaria (0/1). Valores encontrados: {uniques}"
    except:
         return False, "Error validando columna de evento."
         
    return True, "OK"


def check_event_coding(evento_series: pd.Series) -> Dict[str, Any]:
    """
    Analiza la codificación de la serie de eventos.
    Retorna conteos de censura vs eventos.
    """
    try:
        clean_events = pd.to_numeric(evento_series, errors='coerce').dropna()
        n_total = len(clean_events)
        n_events = (clean_events == 1).sum()
        n_censored = (clean_events == 0).sum()
        
        return {
            "total_samples": int(n_total),
            "observed_events": int(n_events),
            "censored_samples": int(n_censored),
            "censoring_rate": float(n_censored / n_total) if n_total > 0 else 0
        }
    except Exception as e:
        return {"error": str(e)}


def survival_risk_groups(tiempo: pd.Series, 
                       evento: pd.Series, 
                       cutpoint: float = None) -> Dict[str, Any]:
    """
    Calcula tasas de riesgo simples dividiendo por un punto de corte temporal opcional.
    """
    df = pd.DataFrame({'T': tiempo, 'E': evento}).dropna()
    
    if cutpoint is None:
        cutpoint = df['T'].median()
        
    group_early = df[df['T'] < cutpoint]
    group_late = df[df['T'] >= cutpoint]
    
    rate_early = group_early['E'].sum() / len(group_early) if len(group_early) > 0 else 0
    rate_late = group_late['E'].sum() / len(group_late) if len(group_late) > 0 else 0
    
    return {
        "cutpoint": cutpoint,
        "rate_below_cutpoint": rate_early,
        "rate_above_cutpoint": rate_late
    }


# ==============================================================================
# 2. KAPLAN-MEIER
# ==============================================================================

def kaplan_meier_estimator(tiempo: pd.Series, 
                         evento: pd.Series, 
                         grupos: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Calcula el estimador de Kaplan-Meier.
    Soporta desglose por grupos (estratificado).
    """
    if not LIFELINES_AVAILABLE:
        return {"error": "Librería 'lifelines' no instalada."}
        
    df = pd.DataFrame({'T': tiempo, 'E': evento})
    if grupos is not None:
        df['Group'] = grupos
        
    df = df.dropna()
    
    results = {}
    
    kmf = KaplanMeierFitter()
    
    if grupos is None:
        kmf.fit(df['T'], event_observed=df['E'], label='All')
        results['Global'] = {
            "median_survival_time": float(kmf.median_survival_time_),
            "survival_table": kmf.survival_function_.to_dict(),
            "confidence_interval": kmf.confidence_interval_.to_dict()
        }
    else:
        unique_groups = df['Group'].unique()
        group_results = {}
        for g in unique_groups:
            mask = df['Group'] == g
            kmf.fit(df.loc[mask, 'T'], event_observed=df.loc[mask, 'E'], label=str(g))
            group_results[str(g)] = {
                "median_survival_time": float(kmf.median_survival_time_)
            }
        results['ByGroup'] = group_results
        
    return results


def kaplan_meier_plot(tiempo: pd.Series, 
                     evento: pd.Series, 
                     grupos: Optional[pd.Series] = None) -> Any:
    """
    Genera un objeto Figura de Matplotlib con las curvas KM.
    """
    if not LIFELINES_AVAILABLE: return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    kmf = KaplanMeierFitter()
    
    df = pd.DataFrame({'T': tiempo, 'E': evento})
    
    if grupos is None:
        kmf.fit(df['T'], event_observed=df['E'], label='Global Estimator')
        kmf.plot_survival_function(ax=ax)
    else:
        df['G'] = grupos
        df = df.dropna()
        unique_groups = df['G'].unique()
        
        for g in unique_groups:
            mask = df['G'] == g
            kmf.fit(df.loc[mask, 'T'], event_observed=df.loc[mask, 'E'], label=str(g))
            kmf.plot_survival_function(ax=ax)
            
    ax.set_title("Curvas de Supervivencia Kaplan-Meier")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Probabilidad de Supervivencia")
    ax.grid(True, alpha=0.3)
    
    return fig


# ==============================================================================
# 3. MODELO COX (PROPORTIONAL HAZARDS)
# ==============================================================================

def cox_regression(tiempo: pd.Series, 
                 evento: pd.Series, 
                 covariates: pd.DataFrame) -> Dict[str, Any]:
    """
    Ajusta un modelo de Regresión de Cox.
    """
    if not LIFELINES_AVAILABLE: return {"error": "Lifelines requerido."}
    
    # Preparar dataset único
    # Asegurar numéricos para evitar ValueError de statsmodels/lifelines con objects
    t_clean = pd.to_numeric(tiempo, errors='coerce')
    e_clean = pd.to_numeric(evento, errors='coerce')
    
    # Covariables: intentar convertir a numérico. 
    # Lifelines CoxPHFitter maneja dummies internos si se usa formula API, pero fit(df) espera numéricos o lo hace auto?
    # Mejor asegurar numérico para evitar "object".
    c_clean = covariates.apply(pd.to_numeric, errors='coerce')
    
    # Unir
    data = c_clean.copy()
    data['T'] = t_clean
    data['E'] = e_clean
    
    data = data.dropna()
    
    # Casting explícito a float para seguridad numpy
    data = data.astype(float)
    
    if data.empty:
        return {"error": "Datos vacíos tras limpieza (NaNs)."}
    
    cph = CoxPHFitter()
    
    try:
        cph.fit(data, duration_col='T', event_col='E')
        
        summary = cph.summary
        # Extraer campos clave
        coefs = []
        for idx, row in summary.iterrows():
            coefs.append({
                "covariate": idx,
                "coef": row['coef'],
                "exp_coef (HR)": row['exp(coef)'], # Hazard Ratio
                "se(coef)": row['se(coef)'],
                "p_value": row['p'],
                "lower_0.95": row['lower 0.95'],
                "upper_0.95": row['upper 0.95']
            })
            
        return {
            "method": "Cox Proportional Hazards",
            "concordance_index": cph.concordance_index_, # C-index
            "log_likelihood": cph.log_likelihood_,
            "coefficients": coefs,
            "aic": cph.AIC_partial_
        }
    except Exception as e:
        return {"error": f"Fallo ajuste Cox: {str(e)}"}


def verify_proportional_hazards(model_fitted: Any, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Verifica el supuesto de riesgos proporcionales.
    model_fitted debe ser una instancia de CoxPHFitter ya ajustada.
    """
    if not LIFELINES_AVAILABLE: return {}
    
    try:
        # check_assumptions method in lifelines runs several tests
        # Retorna una lista de tuplas o imprime. 
        # proportional_hazard_test retorna estadisticos
        from lifelines.statistics import proportional_hazard_test
        
        results = proportional_hazard_test(model_fitted, data, time_transform='km')
        
        return {
            "test_name": "Proportional Hazard Test",
            "p_values": results.p_value.to_dict(), # p-value por variable
            "assumption_met": all(p > 0.05 for p in results.p_value)
        }
    except Exception as e:
         return {"error": str(e)}


# ==============================================================================
# 4. LOG-RANK TEST
# ==============================================================================

def logrank_test(tiempo1: pd.Series, evento1: pd.Series,
                tiempo2: pd.Series, evento2: pd.Series) -> Dict[str, Any]:
    """
    Prueba de Log-rank para comparar dos curvas de supervivencia.
    """
    if not LIFELINES_AVAILABLE: return {"error": "Lifelines requerido."}
    
    # Limpieza independiente
    t1 = pd.to_numeric(tiempo1, errors='coerce').dropna()
    e1 = pd.to_numeric(evento1, errors='coerce').dropna()
    t2 = pd.to_numeric(tiempo2, errors='coerce').dropna()
    e2 = pd.to_numeric(evento2, errors='coerce').dropna()
    
    # Alinear índices si es necesario (o asumir listas limpias)
    # Lifelines maneja arrays de diferente longitud
    
    try:
        results = lifelines_logrank(t1, t2, event_observed_A=e1, event_observed_B=e2)
        
        return {
            "test_name": "Log-rank Test",
            "statistic": results.test_statistic,
            "p_value": results.p_value,
            "interpretation": "Curvas Significativamente Diferentes" if results.p_value < 0.05 else "Sin Diferencia Significativa"
        }
    except Exception as e:
        return {"error": str(e)}


# ==============================================================================
# 5. WEIBULL MODEL
# ==============================================================================

def weibull_analysis(tiempo: pd.Series, evento: pd.Series) -> Dict[str, Any]:
    """
    Ajusta un modelo paramétrico Weibull.
    Útil para estimar vida media o tasas de fallo.
    """
    if not LIFELINES_AVAILABLE: return {"error": "Lifelines requerido."}
    
    df = pd.DataFrame({'T': tiempo, 'E': evento}).dropna()
    
    wf = WeibullFitter()
    
    try:
        wf.fit(df['T'], event_observed=df['E'])
        
        return {
            "method": "Weibull Parametric Model",
            "lambda_": wf.lambda_,
            "rho_": wf.rho_,
            "median_survival_time": wf.median_survival_time_,
            "aic": wf.AIC_
        }
    except Exception as e:
        return {"error": str(e)}
