"""
Módulo de Regresión y Modelado (BioStat Easy)
---------------------------------------------
Este módulo gestiona modelos estadísticos predictivos y de asociación multivariable.
Incluye regresión lineal (simple y múltiple), regresión logística y modelos lineales generalizados (GLM),
así como herramientas de diagnóstico (VIF, verificación de supuestos).

Responsabilidades:
- Regresión Lineal (Simple y Múltiple)
- Regresión Logística (Binaria)
- GLM (Poisson, Binomial)
- Diagnósticos de modelo (Residuales, Heterocedasticidad)
- Verificación de multicolinealidad (VIF)

Autor: BioStat Easy Team
Versión: 2.5
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Union, Optional, List, Any

# Importación robusta de statsmodels
STATSMODELS_AVAILABLE = False
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    STATSMODELS_AVAILABLE = True
except ImportError:
    pass

# Importación opcional de sklearn (solo si se necesita para algo específico, pero priorizamos statsmodels para inferencia)
SKLEARN_AVAILABLE = False
try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    pass


# ==============================================================================
# 1. REGRESIÓN LINEAL
# ==============================================================================

def linear_regression_simple(x: Union[pd.Series, np.ndarray], 
                           y: Union[pd.Series, np.ndarray], 
                           intervalo_confianza: float = 0.95) -> Dict[str, Any]:
    """
    Realiza una regresión lineal simple OLS.
    """
    try:
        # 1. Limpieza y Alineación
        # Convertir a Series si es necesario para usar métodos de pandas simplificados
        if not isinstance(x, pd.Series): x = pd.Series(x)
        if not isinstance(y, pd.Series): y = pd.Series(y)

        # PASO CRÍTICO: Convertir a numérico explicitamente
        x_numeric = pd.to_numeric(x, errors='coerce')
        y_numeric = pd.to_numeric(y, errors='coerce')
        
        # Eliminar NaN conjuntos
        df_temp = pd.DataFrame({'x': x_numeric, 'y': y_numeric}).dropna()
        
        if len(df_temp) < 2:
            return {"error": "Datos insuficientes (N < 2)"}
        
        # Convertir a numpy float64 explícito
        x_clean = np.asarray(df_temp['x'], dtype='float64')
        y_clean = np.asarray(df_temp['y'], dtype='float64')

        # 2. Preferir statsmodels para reporte estadístico completo
        if STATSMODELS_AVAILABLE:
            X_const = sm.add_constant(x_clean) # Agregar intercepto
            model = sm.OLS(y_clean, X_const).fit()
            
            # Extracción segura de parámetros
            params = model.params
            conf = model.conf_int(alpha=1-intervalo_confianza)
            
            # Param keys: 'const' and 'x1' usually for numpy array input
            # Cuando pasamos numpy array a OLS, la variable se llama 'x1' por defecto
            intercept_val = float(params[0]) # const is usually first if added by add_constant
            slope_val = float(params[1])
            
            return {
                "method": "OLS (statsmodels)",
                "n_obs": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_pvalue": float(model.f_pvalue),
                "intercept": intercept_val,
                "slope": slope_val,
                "p_value_slope": float(model.pvalues[1]) if len(model.pvalues)>1 else np.nan,
                "conf_int_slope": (float(conf[1][0]), float(conf[1][1])) if len(conf)>1 else (np.nan, np.nan),
                "residuals": model.resid.tolist(),
                "fitted": model.fittedvalues.tolist(),
                "summary_obj": model 
            }
            
        # Fallback con scipy.stats.linregress
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        return {
            "method": "Linear Regression (scipy)",
            "n_obs": len(x_clean),
            "r_squared": r_value**2,
            "adj_r_squared": None, 
            "f_pvalue": None,
            "intercept": intercept,
            "slope": slope,
            "p_value_slope": p_value,
            "conf_int_slope": None, 
            "residuals": (y_clean - (slope*x_clean + intercept)).tolist(),
            "fitted": (slope*x_clean + intercept).tolist()
        }
    except Exception as e:
        return {'error': f"Error en regresión lineal simple: {str(e)}"}


def linear_regression_multiple(X: pd.DataFrame, 
                             y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Realiza regresión lineal múltiple.
    Requiere statsmodels para resultados detallados.
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "Statsmodels no está instalado. Requerido para regresión múltiple."}

    try:
        # 1. Conversión y Limpieza Robusta
        # Aseguramos que X sea DataFrame numérico
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        y_numeric = pd.to_numeric(y, errors='coerce')
        
        # Eliminar filas con NaN en cualquiera
        mask = ~(X_numeric.isna().any(axis=1) | y_numeric.isna())
        
        X_clean_df = X_numeric[mask]
        y_clean_series = y_numeric[mask]
        
        if X_clean_df.empty:
             return {"error": "Datos vacíos tras limpieza (NaNs detectados)."}
             
        # Convertir a numpy float64 explícito para evitar problemas de object-dtype
        X_clean = np.asarray(X_clean_df.values, dtype='float64')
        y_clean = np.asarray(y_clean_series.values, dtype='float64')
        
        # Nombres de columnas para reporte
        col_names = X_clean_df.columns.tolist()

        # 2. Modelo
        X_const = sm.add_constant(X_clean)
        # add_constant en numpy agrega columna de 1s al inicio (idx 0)
        # Ajustamos col_names para que coincidan
        col_names_const = ['const'] + col_names
        
        model = sm.OLS(y_clean, X_const).fit()
        
        # 3. Formateo de Resultados
        coefs = []
        conf_int = model.conf_int()
        
        # model.params es numpy array si input fue numpy array
        for i, col_name in enumerate(col_names_const):
            coefs.append({
                "variable": col_name,
                "coef": float(model.params[i]),
                "std_err": float(model.bse[i]),
                "t_stat": float(model.tvalues[i]),
                "p_value": float(model.pvalues[i]),
                "lower_ci": float(conf_int[i][0]),
                "upper_ci": float(conf_int[i][1])
            })
            
        return {
            "method": "OLS Multiple",
            "n_obs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "coefficients": coefs,
            "aic": float(model.aic),
            "bic": float(model.bic),
            "residuals": model.resid.tolist(),
            "fitted": model.fittedvalues.tolist()
        }
    except Exception as e:
        return {'error': f"Error en regresión múltiple: {str(e)}"}


# ==============================================================================
# 2. REGRESIÓN LOGÍSTICA
# ==============================================================================

def logistic_regression(X: pd.DataFrame, 
                       y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """Wraper para regresión logística binaria general."""
    return logistic_regression_binary(X, y) # Por ahora solo implementamos binaria


def logistic_regression_binary(X: pd.DataFrame, 
                             y_binary: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Regresión Logística para variable respuesta binaria (0/1).
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "Se requiere statsmodels."}

    try:
        # Limpieza robusta
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        y_numeric = pd.to_numeric(y_binary, errors='coerce')
        
        mask = ~(X_numeric.isna().any(axis=1) | y_numeric.isna())
        
        X_clean_df = X_numeric[mask]
        y_clean_series = y_numeric[mask]
        
        if X_clean_df.empty:
             return {"error": "Datos vacíos tras limpieza."}
             
        # Validación binaria
        unique_vals = y_clean_series.dropna().unique()
        if len(unique_vals) != 2:
            # check if 0 or 1, maybe just 1 value present effectively
            if len(unique_vals) < 2: return {"error": "Variable respuesta constante."}
            return {"error": f"La variable respuesta debe ser binaria. Valores encontrados: {unique_vals}"}

        X_clean = np.asarray(X_clean_df.values, dtype='float64')
        y_clean = np.asarray(y_clean_series.values, dtype='float64')
        
        col_names = X_clean_df.columns.tolist()

        X_const = sm.add_constant(X_clean)
        col_names_const = ['const'] + col_names
        
        model = sm.Logit(y_clean, X_const).fit(disp=0)
        
        # Resultados
        params = model.params
        conf = model.conf_int()
        odds_ratios = np.exp(params)
        odds_ci_lower = np.exp(conf[:,0])
        odds_ci_upper = np.exp(conf[:,1])
        
        coefs = []
        for i, col in enumerate(col_names_const):
            coefs.append({
                "variable": col,
                "coef": float(params[i]),
                "p_value": float(model.pvalues[i]),
                "odds_ratio": float(odds_ratios[i]),
                "or_ci_lower": float(odds_ci_lower[i]),
                "or_ci_upper": float(odds_ci_upper[i])
            })
            
        return {
            "method": "Logistic Regression (Logit)",
            "n_obs": int(model.nobs),
            "pseudo_r2": float(model.prsquared),
            "llr_pvalue": float(model.llr_pvalue), 
            "coefficients": coefs,
            "aic": float(model.aic)
        }
    except Exception as e:
        return {"error": f"Error al ajustar modelo Logit: {str(e)}"}


# ==============================================================================
# 3. GLM (Modelos Lineales Generalizados)
# ==============================================================================

def glm_poisson(X: pd.DataFrame, 
                y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    GLM con familia Poisson (datos de conteo).
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "Requiere statsmodels."}
        
    try:
        # Limpieza robusta
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        y_numeric = pd.to_numeric(y, errors='coerce')
        
        mask = ~(X_numeric.isna().any(axis=1) | y_numeric.isna())
        
        X_clean_df = X_numeric[mask]
        y_clean_series = y_numeric[mask]
        
        if X_clean_df.empty: return {"error": "Datos vacíos."}
        
        X_clean = np.asarray(X_clean_df.values, dtype='float64')
        y_clean = np.asarray(y_clean_series.values, dtype='float64')
        col_names = X_clean_df.columns.tolist()

        X_const = sm.add_constant(X_clean)
        col_names_const = ['const'] + col_names

        model = sm.GLM(y_clean, X_const, family=sm.families.Poisson()).fit()
        
        coefs = []
        for i, col in enumerate(col_names_const):
            coefs.append({
                "variable": col,
                "coef": float(model.params[i]),
                "p_value": float(model.pvalues[i]),
                "exp_coef": float(np.exp(model.params[i])) # IRR
            })
            
        return {
            "method": "GLM Poisson",
            "n_obs": int(model.nobs),
            "aic": float(model.aic),
            "coefficients": coefs,
            "deviance": float(model.deviance)
        }
    except Exception as e:
        return {"error": f"Error ajustando GLM Poisson: {str(e)}"}


def glm_binomial(X: pd.DataFrame, 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """GLM con familia Binomial (equivalente a Logit pero vía GLM framework)."""
    if not STATSMODELS_AVAILABLE:
        return {"error": "Requiere statsmodels."}
        
    try:
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        y_numeric = pd.to_numeric(y, errors='coerce')
        mask = ~(X_numeric.isna().any(axis=1) | y_numeric.isna())
        
        X_clean_df = X_numeric[mask]
        y_clean_series = y_numeric[mask]
        
        if X_clean_df.empty: return {"error": "Datos vacíos."}
        
        X_clean = np.asarray(X_clean_df.values, dtype='float64')
        y_clean = np.asarray(y_clean_series.values, dtype='float64')
        col_names = X_clean_df.columns.tolist()

        X_const = sm.add_constant(X_clean)
        col_names_const = ['const'] + col_names

        model = sm.GLM(y_clean, X_const, family=sm.families.Binomial()).fit()
        return {
            "method": "GLM Binomial",
            "aic": float(model.aic),
            "coefficients": [{ 
                "variable": col_names_const[i], 
                "coef": float(model.params[i]), 
                "p_value": float(model.pvalues[i]) 
            } for i in range(len(col_names_const))]
        }
    except Exception as e:
        return {"error": f"Error GLM Binomial: {str(e)}"}


# ==============================================================================
# 4. DIAGNÓSTICOS
# ==============================================================================

def verify_regression_assumptions(residuals: List[float], 
                                fitted: List[float]) -> Dict[str, Any]:
    """
    Verifica supuestos básicos: Normalidad de residuos y Homocedasticidad.
    """
    if not residuals or len(residuals) < 3:
        return {"error": "Insuficientes residuales."}
        
    res_array = np.array(residuals)
    
    # 1. Normalidad (Shapiro)
    try:
        stat_norm, p_norm = stats.shapiro(res_array)
        is_normal = p_norm > 0.05
    except:
        stat_norm, p_norm, is_normal = np.nan, np.nan, False
        
    # 2. Homocedasticidad (Breusch-Pagan) - Requiere X, pero aquí estimamos visualmente o simple
    # Si tenemos statsmodels, podemos intentar un test simple usando fitted como proxy simple o retornar solo datos
    # Como esta función solo recibe residuales/fitted, haremos un proxy simple o indicaremos revisión visual.
    
    # Nota: Breusch-Pagan requiere las variables exógenas (X). Como no las pasamos a esta función
    # específica por firma, nos limitamos aquí a Normalidad.
    # Para Breusch-Pagan completo, úsese regression_diagnostics pasando el objeto del modelo.
    
    return {
        "normality_test": "Shapiro-Wilk",
        "normality_p": float(p_norm),
        "residuals_normally_distributed": is_normal
    }


def regression_diagnostics(model_result: Any) -> Dict[str, Any]:
    """
    Ejecuta diagnósticos completos dado un objeto Result de statsmodels.
    
    Args:
        model_result: Objeto retornado por .fit() de statsmodels (OLSResults, etc.)
    """
    if not STATSMODELS_AVAILABLE:
        return {"warning": "Statsmodels no disponible."}
        
    if not hasattr(model_result, 'resid'):
        return {"error": "Objeto modelo inválido."}
        
    # 1. Normalidad Residuos (Jarque-Bera dentro de statsmodels summary, o Shapiro externo)
    resid = model_result.resid
    shapiro_curr, shapiro_p = stats.shapiro(resid)
    
    # 2. Homocedasticidad (Breusch-Pagan)
    # Requiere model.model.exog
    try:
        bp_test = het_breuschpagan(resid, model_result.model.exog)
        bp_p_value = bp_test[1] # p-value del test LM
        homoscedastic = bp_p_value > 0.05
    except:
        bp_p_value = np.nan
        homoscedastic = None
        
    # 3. Autocorrelación (Durbin-Watson)
    # statsmodels suele tenerlo en summary, o durbin_watson(resid)
    # Valor cerca de 2 es bueno (1.5 - 2.5).
    from statsmodels.stats.stattools import durbin_watson
    dw_stat = durbin_watson(resid)
    
    return {
        "shapiro_p_value": float(shapiro_p),
        "normality_assumption_met": shapiro_p > 0.05,
        "breusch_pagan_p_value": float(bp_p_value),
        "homoscedasticity_assumption_met": homoscedastic,
        "durbin_watson": float(dw_stat),
        "no_autocorrelation": 1.5 < dw_stat < 2.5
    }


def multicollinearity_check(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula VIF (Variance Inflation Factor) para detectar multicolinealidad.
    
    Returns:
        Dict con DataFrame de VIF y lista de variables problemáticas (VIF > 10).
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "Statsmodels requerido para VIF."}
        
    # Limpieza y conversión
    X_num = X.select_dtypes(include=[np.number]).dropna()
    
    # Debe haber constante para cálculo correcto de VIF en statsmodels
    X_const = sm.add_constant(X_num)
    
    vif_data = []
    high_vif_vars = []
    
    for i in range(X_const.shape[1]):
        col_name = X_const.columns[i]
        if col_name == 'const':
            continue
            
        try:
            val = variance_inflation_factor(X_const.values, i)
        except:
             val = np.inf
             
        vif_data.append({"variable": col_name, "VIF": val})
        
        if val > 10: # Umbral común
            high_vif_vars.append(col_name)
            
    return {
        "vif_table": pd.DataFrame(vif_data),
        "high_multicollinearity_vars": high_vif_vars,
        "has_issues": len(high_vif_vars) > 0
    }
