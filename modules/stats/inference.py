"""
Módulo de Inferencia Estadística (BioStat Easy)
-----------------------------------------------
Este módulo contiene todas las funciones relacionadas con pruebas de hipótesis,
tanto paramétricas como no paramétricas, análisis de asociación y correlación,
así como pruebas post-hoc y verificación de supuestos.

Responsabilidades:
- Pruebas t (Student y Welch)
- ANOVA y pruebas no paramétricas equivalentes (Kruskal-Wallis)
- Pruebas de asociación (Chi2, Fisher)
- Correlaciones (Pearson, Spearman)
- Análisis Post-hoc (Tukey, Dunn/Bonferroni)
- Verificación de supuestos (Normalidad, Homocedasticidad)

Autor: BioStat Easy Team
Versión: 2.5
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Union, Optional, List, Tuple, Any

# Importación segura de statsmodels
STATSMODELS_AVAILABLE = False
try:
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_AVAILABLE = True
except ImportError:
    pass

# ==============================================================================
# 1. PRUEBAS PARAMÉTRICAS
# ==============================================================================

def ttest_independiente(group1: Union[pd.Series, np.ndarray], 
                       group2: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Realiza una prueba t de Student para muestras independientes.
    Calcula automáticamente la d de Cohen para el tamaño del efecto.
    Determina si usar corrección de Welch (varianzas desiguales) basado en Levene interno.

    Args:
        group1: Datos del grupo 1.
        group2: Datos del grupo 2.

    Returns:
        Diccionario con estadístico t, p-value, d de Cohen, y conclusión.
    """
    g1 = pd.to_numeric(group1, errors='coerce').dropna()
    g2 = pd.to_numeric(group2, errors='coerce').dropna()
    
    if len(g1) < 2 or len(g2) < 2:
        return {"error": "Tamaño de muestra insuficiente (n < 2)"}

    # Verificar homogeneidad de varianzas (Levene)
    stat_levene, p_levene = stats.levene(g1, g2)
    equal_var = p_levene > 0.05
    
    # Prueba t
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
    
    # Tamaño del efecto (Cohen's d)
    # d = (mean1 - mean2) / pooled_std
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    
    cohen_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std != 0 else 0
    
    # Interpretación Cohen
    if abs(cohen_d) < 0.2: eff_interp = "Despreciable"
    elif abs(cohen_d) < 0.5: eff_interp = "Pequeño"
    elif abs(cohen_d) < 0.8: eff_interp = "Mediano"
    else: eff_interp = "Grande"

    return {
        "test_name": "T-Student (Welch)" if not equal_var else "T-Student (Standard)",
        "statistic": float(t_stat),
        "p_value": float(p_val),
        "effect_size": float(cohen_d),
        "effect_size_name": "Cohen's d",
        "interpretation": "Diferencia Significativa" if p_val < 0.05 else "No Significativo",
        "effect_interpretation": eff_interp,
        "assumptions": {"homoscedasticity": equal_var, "levene_p": p_levene}
    }


def anova_oneway(groups_data: List[Union[pd.Series, np.ndarray]]) -> Dict[str, Any]:
    """
    Realiza un ANOVA de una vía para comparar medias de 3+ grupos.
    
    Args:
        groups_data: Lista de arrays/series, uno por cada grupo.

    Returns:
        Diccionario con estadístico F, p-value, Eta cuadrado.
    """
    clean_groups = [pd.to_numeric(g, errors='coerce').dropna() for g in groups_data]
    clean_groups = [g for g in clean_groups if len(g) > 1]
    
    if len(clean_groups) < 2:
        return {"error": "Se requieren al menos 2 grupos con datos válidos."}
        
    f_stat, p_val = stats.f_oneway(*clean_groups)
    
    # Tamaño del efecto: Eta Cuadrado (η²) = SS_between / SS_total
    # Método simplificado aproximado
    k = len(clean_groups)
    N = sum(len(g) for g in clean_groups)
    grand_mean = np.mean(np.concatenate(clean_groups))
    
    ss_benzween = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in clean_groups)
    ss_total = sum(sum((x - grand_mean)**2 for x in g) for g in clean_groups)
    
    eta_sq = ss_benzween / ss_total if ss_total != 0 else 0
    
    # Interpretación Eta Cuadrado
    if eta_sq < 0.01: eta_interp = "Despreciable"
    elif eta_sq < 0.06: eta_interp = "Pequeño"
    elif eta_sq < 0.14: eta_interp = "Mediano"
    else: eta_interp = "Grande"
    
    return {
        "test_name": "ANOVA One-Way",
        "statistic": float(f_stat),
        "p_value": float(p_val),
        "effect_size": float(eta_sq),
        "effect_size_name": "Eta-Squared (η²)",
        "interpretation": "Diferencia Significativa" if p_val < 0.05 else "No Significativo",
        "effect_interpretation": eta_interp
    }


def friedman_test(*args) -> Dict[str, Any]:
    """
    Prueba de Friedman para medidas repetidas (no paramétrica).
    
    Args:
        *args: Series o arrays de datos (uno por condición). Debe tener mismo largo.
        
    Returns:
        Diccionario con resultados.
    """
    # Limpieza
    clean_args = []
    for arg in args:
        clean_args.append(pd.to_numeric(arg, errors='coerce').fillna(0)) # Friedman no tolera NaNs, cuidadoso aquí
        
    # Verificar longitudes iguales (requerido para medidas repetidas)
    lengths = [len(x) for x in clean_args]
    if len(set(lengths)) > 1:
        # Intentar alinear índices si son pandas Series
        try:
            df_temp = pd.DataFrame({f"g{i}": arg for i, arg in enumerate(args)})
            df_temp = df_temp.dropna()
            if df_temp.empty:
                 return {"error": "Datos vacíos tras alinear muestras."}
            clean_args = [df_temp[col] for col in df_temp.columns]
        except:
             return {"error": "Muestras de diferente tamaño para Friedman (requiere pareadas)."}

    if len(clean_args) < 3:
        return {"error": "Friedman requiere al menos 3 mediciones/grupos."}

    stat, p_val = stats.friedmanchisquare(*clean_args)
    
    # Kendall's W como tamaño del efecto
    # W = X2 / (N * (k - 1))
    n = len(clean_args[0])
    k = len(clean_args)
    kendall_w = stat / (n * (k - 1)) if (n * (k - 1)) > 0 else 0
    
    return {
        "test_name": "Friedman Test",
        "statistic": float(stat),
        "p_value": float(p_val),
        "effect_size": float(kendall_w),
        "effect_size_name": "Kendall's W",
        "interpretation": "Diferencia Significativa" if p_val < 0.05 else "No Significativo"
    }


# ==============================================================================
# 2. PRUEBAS NO PARAMÉTRICAS
# ==============================================================================

def mann_whitney_u(group1: Union[pd.Series, np.ndarray], 
                   group2: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Prueba U de Mann-Whitney para dos muestras independientes.
    """
    g1 = pd.to_numeric(group1, errors='coerce').dropna()
    g2 = pd.to_numeric(group2, errors='coerce').dropna()
    
    if len(g1) < 2 or len(g2) < 2:
        return {"error": "Tamaño de muestra insuficiente."}
        
    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    
    # Tamaño del efecto: Rank Biserial Correlation
    # r = 1 - (2U / (n1 * n2))
    n1, n2 = len(g1), len(g2)
    r_biserial = 1 - (2 * u_stat) / (n1 * n2)
    
    return {
        "test_name": "Mann-Whitney U",
        "statistic": float(u_stat),
        "p_value": float(p_val),
        "effect_size": float(r_biserial),
        "effect_size_name": "Rank Biserial r",
        "interpretation": "Diferencia Significativa" if p_val < 0.05 else "No Significativo"
    }


def kruskal_wallis(*groups) -> Dict[str, Any]:
    """
    Prueba de Kruskal-Wallis para k muestras independientes.
    Equivalent ANOVA no paramétrico.
    """
    clean_groups = [pd.to_numeric(g, errors='coerce').dropna() for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 1]
    
    if len(clean_groups) < 2:
        return {"error": "Se requieren al menos 2 grupos."}
        
    h_stat, p_val = stats.kruskal(*clean_groups)
    
    # Epsilon squared (ε²)
    # ε² = H / ((n^2 - 1) / (n + 1))
    n = sum(len(g) for g in clean_groups)
    k = len(clean_groups)
    epsilon_sq = h_stat / ((n**2 - 1) / (n + 1)) if n > 1 else 0
    
    return {
        "test_name": "Kruskal-Wallis",
        "statistic": float(h_stat),
        "p_value": float(p_val),
        "effect_size": float(epsilon_sq),
        "effect_size_name": "Epsilon-squared (ε²)",
        "interpretation": "Diferencia Significativa" if p_val < 0.05 else "No Significativo"
    }


def wilcoxon_paired(x: Union[pd.Series, np.ndarray], 
                    y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Prueba de los rangos con signo de Wilcoxon (muestras pareadas).
    """
    # Alinear y limpiar pareados
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        df_temp = pd.DataFrame({'x': x, 'y': y}).dropna()
        x_clean, y_clean = df_temp['x'], df_temp['y']
    else:
        # Asumir arrays alineados
        x = np.array(x)
        y = np.array(y)
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean, y_clean = x[mask], y[mask]
        
    if len(x_clean) < 2:
        return {"error": "Datos insuficientes (requiere pares completos)."}

    w_stat, p_val = stats.wilcoxon(x_clean, y_clean)
    
    # Rank Biserial Correlation para Wilcoxon
    # r = W / Total Ranks Sum. Approx simple con z-score puede usarse r = z / sqrt(N)
    # Aquí reportaremos simplemente el stat y p como principal.
    
    return {
        "test_name": "Wilcoxon Signed-Rank",
        "statistic": float(w_stat),
        "p_value": float(p_val),
        "effect_size": None, # Complex to calc accurately standardized
        "effect_size_name": "-",
        "interpretation": "Diferencia Significativa" if p_val < 0.05 else "No Significativo"
    }


# ==============================================================================
# 3. ASOCIACIÓN (CHI2 & FISHER)
# ==============================================================================

def chi_cuadrado(var1: Union[pd.Series, np.ndarray], 
                 var2: Union[pd.Series, np.ndarray], 
                 alpha: float = 0.05) -> Dict[str, Any]:
    """
    Prueba de Chi-cuadrado de independencia.
    Incluye V de Cramer.
    """
    # Crear tabla de contingencia
    crosstab = pd.crosstab(var1, var2)
    
    if crosstab.empty or crosstab.size < 4:
        return {"error": "Datos insuficientes para tabla 2x2 mínima."}
        
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    
    # Cramer's V
    n = crosstab.sum().sum()
    min_dim = min(crosstab.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0
    
    return {
        "test_name": "Chi-Cuadrado Pearson",
        "statistic": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "effect_size": float(cramer_v),
        "effect_size_name": "Cramer's V",
        "interpretation": "Existe Asociación" if p < alpha else "Independientes"
    }


def fisher_exact(tabla_2x2: Union[pd.DataFrame, np.ndarray], 
                 alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Prueba exacta de Fisher para tablas 2x2.
    """
    tabla = np.array(tabla_2x2)
    if tabla.shape != (2, 2):
        return {"error": "Fisher requiere tabla estricta 2x2."}
        
    odds_ratio, p_value = stats.fisher_exact(tabla, alternative=alternative)
    
    return {
        "test_name": "Fisher Exact Test",
        "statistic": float(odds_ratio), # Statistic is Odds Ratio
        "p_value": float(p_value),
        "effect_size": float(odds_ratio),
        "effect_size_name": "Odds Ratio",
        "interpretation": "Asociación Significativa" if p_value < 0.05 else "No Significativo"
    }


# ==============================================================================
# 4. CORRELACIÓN
# ==============================================================================

def pearson_correlation(x: Any, y: Any, alpha: float = 0.05) -> Dict[str, Any]:
    """Coeficiente de correlación de Pearson."""
    
    # Limpieza conjunta
    df_temp = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df_temp) < 2: return {"error": "Datos insuficientes"}
    
    stat, p = stats.pearsonr(df_temp['x'], df_temp['y'])
    
    # Interpretación magnitud
    r = abs(stat)
    if r < 0.3: mag = "Débil"
    elif r < 0.7: mag = "Moderada"
    else: mag = "Fuerte"
    
    return {
        "test_name": "Correlación Pearson",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_interpretation": mag,
        "interpretation": "Correlación Significativa" if p < alpha else "No Correlacionados"
    }


def spearman_correlation(x: Any, y: Any, alpha: float = 0.05) -> Dict[str, Any]:
    """Coeficiente de correlación de Spearman (rangos)."""
    
    df_temp = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df_temp) < 2: return {"error": "Datos insuficientes"}
    
    stat, p = stats.spearmanr(df_temp['x'], df_temp['y'])
    
    r = abs(stat)
    if r < 0.3: mag = "Débil"
    elif r < 0.7: mag = "Moderada"
    else: mag = "Fuerte"
    
    return {
        "test_name": "Correlación Spearman",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_interpretation": mag,
        "interpretation": "Correlación Significativa" if p < alpha else "No Correlacionados"
    }


# ==============================================================================
# 5. POST-HOC TESTS
# ==============================================================================

def tukey_hsd(df: pd.DataFrame, var_num: str, var_grp: str) -> Optional[pd.DataFrame]:
    """
    Calcula Tukey HSD Post-hoc.
    Requiere statsmodels.
    
    Returns: DataFrame con tabla de resultados detallada.
    """
    if not STATSMODELS_AVAILABLE:
        return None
        
    data_clean = df[[var_num, var_grp]].copy()
    data_clean[var_num] = pd.to_numeric(data_clean[var_num], errors='coerce')
    data_clean = data_clean.dropna()
    
    try:
        tukey = pairwise_tukeyhsd(endog=data_clean[var_num], 
                                  groups=data_clean[var_grp], 
                                  alpha=0.05)
        
        # Convertir a DataFrame amigable
        summary_data = tukey.summary().data
        header = summary_data[0]
        rows = summary_data[1:]
        result_df = pd.DataFrame(rows, columns=header)
        return result_df
        
    except Exception:
        return None


def dunn_posthoc_mannwhitney(df: pd.DataFrame, 
                           var_num: str, 
                           var_grp: str, 
                           group_names: List[str]) -> pd.DataFrame:
    """
    Implementación manual de comparaciones por pares (estilo Dunn/Bonferroni)
    para post-hoc no paramétrico. Usa U de Mann-Whitney con corrección de Bonferroni.
    """
    import itertools
    
    data_clean = df[[var_num, var_grp]].dropna()
    
    results = []
    # Generar todas las combinaciones de pares
    pairs = list(itertools.combinations(group_names, 2))
    num_tests = len(pairs)
    
    # Bonferroni correction: alpha_new = 0.05 / num_tests
    corrected_alpha = 0.05 / num_tests if num_tests > 0 else 0.05
    
    for g1_name, g2_name in pairs:
        g1_data = data_clean[data_clean[var_grp] == g1_name][var_num]
        g2_data = data_clean[data_clean[var_grp] == g2_name][var_num]
        
        try:
            u, p = stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
            
            sig = p < corrected_alpha
            
            results.append({
                "Grupo 1": g1_name,
                "Grupo 2": g2_name,
                "P-Valor": p,
                "P-Corregido (Bonf)": p * num_tests if (p*num_tests) <= 1 else 1.0,
                "Significativo": "Sí" if sig else "No"
            })
        except:
             results.append({
                "Grupo 1": g1_name,
                "Grupo 2": g2_name,
                "P-Valor": np.nan,
                "Significativo": "Error"
            })

    return pd.DataFrame(results)


# ==============================================================================
# 6. VERIFICACIÓN DE SUPUESTOS
# ==============================================================================

def shapiro_wilk_test(data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Prueba de normalidad Shapiro-Wilk.
    Recomendada para N < 5000.
    """
    clean_data = pd.to_numeric(data, errors='coerce').dropna()
    
    if len(clean_data) < 3:
        return {"error": "N < 3 insuficiente para Shapiro."}
        
    stat, p = stats.shapiro(clean_data)
    
    return {
        "test_name": "Shapiro-Wilk",
        "statistic": float(stat),
        "p_value": float(p),
        "is_normal": p > 0.05,
        "interpretation": "Distribución Normal" if p > 0.05 else "No Normal"
    }


def levene_test(groups_data: List[Union[pd.Series, np.ndarray]], 
                alpha: float = 0.05) -> Dict[str, Any]:
    """
    Prueba de Levene para homogeneidad de varianzas.
    """
    clean_groups = [pd.to_numeric(g, errors='coerce').dropna() for g in groups_data]
    clean_groups = [g for g in clean_groups if len(g) > 1]
    
    if len(clean_groups) < 2:
        return {"error": "Se requieren al menos 2 grupos."}
        
    stat, p = stats.levene(*clean_groups)
    
    return {
        "test_name": "Levene",
        "statistic": float(stat),
        "p_value": float(p),
        "homoscedasticity": p > alpha,
        "interpretation": "Varianzas Homogéneas" if p > alpha else "Varianzas Diferentes"
    }


# ==============================================================================
# 7. PRUEBAS PAREADAS Y OTRAS (NUEVO STEP 1)
# ==============================================================================

def ttest_paired(group1: Union[pd.Series, np.ndarray],
                 group2: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Prueba t de Student para muestras pareadas (relacionadas).
    """
    try:
        # Alinear estrictamente por índice usando concat (inner join por defecto si series)
        # Esto maneja Series de distinta longitud resultantes de dropna independientes
        s1 = group1 if isinstance(group1, pd.Series) else pd.Series(group1)
        s2 = group2 if isinstance(group2, pd.Series) else pd.Series(group2)
        
        df_temp = pd.concat([s1, s2], axis=1, join='inner').dropna()
        g1 = df_temp.iloc[:, 0]
        g2 = df_temp.iloc[:, 1]

        if len(g1) < 2: 
            return {"error": "Tamaño de muestra insuficiente tras alinear pares (requiere pares completos)."}
            
        t_stat, p_val = stats.ttest_rel(g1, g2)
        
        # Cohen's d paied
        # d = mean(diff) / std(diff)
        diff = np.array(g1) - np.array(g2)
        d_mean = np.mean(diff)
        d_std = np.std(diff, ddof=1)
        
        cohen_d = d_mean / d_std if d_std != 0 else 0
        
        if abs(cohen_d) < 0.2: eff_interp = "Despreciable"
        elif abs(cohen_d) < 0.5: eff_interp = "Pequeño"
        elif abs(cohen_d) < 0.8: eff_interp = "Mediano"
        else: eff_interp = "Grande"

        return {
            "test_name": "Paired T-Test",
            "statistic": float(t_stat),
            "p_value": float(p_val),
            "effect_size": float(cohen_d),
            "effect_size_name": "Cohen's d (paired)",
            "interpretation": "Diferencia Significativa" if p_val < 0.05 else "No Significativo",
            "effect_interpretation": eff_interp
        }
    except Exception as e:
        return {"error": str(e)}


def wilcoxon_paired(group1: Union[pd.Series, np.ndarray],
                    group2: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Prueba de Wilcoxon Signed-Rank para muestras pareadas.
    """
    try:
        # Alinear estrictamente por índice
        s1 = group1 if isinstance(group1, pd.Series) else pd.Series(group1)
        s2 = group2 if isinstance(group2, pd.Series) else pd.Series(group2)
        
        df_temp = pd.concat([s1, s2], axis=1, join='inner').dropna()
        g1c = df_temp.iloc[:, 0]
        g2c = df_temp.iloc[:, 1]
        
        if len(g1c) < 2:
            return {"error": "Tamaño de muestra insuficiente tras alinear pares."}
            
        w_stat, p_val = stats.wilcoxon(g1c, g2c)
        
        return {
            "test_name": "Wilcoxon Signed-Rank (Paired)",
            "statistic": float(w_stat),
            "p_value": float(p_val),
            "interpretation": "Diferencia Significativa" if p_val < 0.05 else "No Significativo"
        }
    except Exception as e:
        return {"error": f"Error Wilcoxon: {str(e)}"}


def mcnemar_test(group1: Union[pd.Series, np.ndarray],
                 group2: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Prueba de McNemar para datos nominales pareados (tablas 2x2).
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "Statsmodels requerido para McNemar."}

    try:
        # Crear tabla contingencia pareada
        # McNemar requiere tabla de contingencia cuadrada
        crosstab = pd.crosstab(group1, group2)
        
        if crosstab.shape != (2, 2):
            return {"error": f"McNemar requiere tabla 2x2 exacta. Se obtuvo {crosstab.shape}."}
            
        from statsmodels.stats.contingency_tables import mcnemar
        
        # exact=True para muestras pequeñas (binomial), False para chi2 (grandes)
        # Usamos exact=True por seguridad
        res = mcnemar(crosstab, exact=True)
        
        return {
            "test_name": "McNemar Test",
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "interpretation": "Cambio Significativo" if res.pvalue < 0.05 else "No Significativo"
        }
    except Exception as e:
         return {"error": str(e)}


def cochran_q_test(*groups) -> Dict[str, Any]:
    """
    Prueba Q de Cochran para k muestras relacionadas (binarias).
    Extensión de McNemar.
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "Statsmodels requerido."}
        
    try:
        # Preparar DataFrame
        data_list = []
        for g in groups:
            data_list.append(pd.to_numeric(g, errors='coerce').fillna(0))
            
        df_temp = pd.DataFrame(data_list).T
        df_temp = df_temp.dropna() # Filas completas
        
        # Validar binario
        if not ((df_temp.isin([0, 1])).all().all()):
             return {"error": "Cochran Q requiere datos binarios (0/1)."}

        from statsmodels.stats.contingency_tables import cochrans_q
        
        res = cochrans_q(df_temp)
        
        return {
            "test_name": "Cochran's Q Test",
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "dof": float(res.df),
            "interpretation": "Diferencia Significativa" if res.pvalue < 0.05 else "No Significativo"
        }
    except Exception as e:
         return {"error": str(e)}


def kendall_tau(x, y) -> Dict[str, Any]:
    """
    Correlación tau de Kendall. Robusta para muestras pequeñas y rangos.
    """
    try:
        df_temp = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(df_temp) < 2: return {"error": "Datos insuficientes"}
        
        stat, p = stats.kendalltau(df_temp['x'], df_temp['y'])
        
        return {
            "test_name": "Kendall's Tau",
            "statistic": float(stat),
            "p_value": float(p),
            "interpretation": "Correlación Significativa" if p < 0.05 else "No Significativo"
        }
    except Exception as e:
        return {"error": str(e)}


def odds_ratio_risk(case_control_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula Odds Ratio para tabla 2x2.
    """
    try:
        df = case_control_df.copy()
        
        # Si ya es contingencia 2x2
        if df.shape == (2,2) and df.applymap(np.isreal).all().all():
             table = df.values
        else:
             # Intentar crear crosstab de las primeras 2 columnas
             if df.shape[1] < 2: return {"error": "Se requieren 2 columnas para tabla 2x2."}
             c1 = df.iloc[:,0]
             c2 = df.iloc[:,1]
             table = pd.crosstab(c1, c2).values
             
        if table.shape != (2,2):
            return {"error": "No se pudo formar una tabla 2x2 válida."}
            
        # Tabla: [[a, b], [c, d]]
        # OR = (a*d) / (b*c)
        # Ajuste de Haldane-Anscombe si hay ceros (sumar 0.5)
        if (table == 0).any():
             table = table + 0.5
             
        a, b = table[0,0], table[0,1]
        c, d = table[1,0], table[1,1]
        
        or_val = (a * d) / (b * c)
        
        # IC 95% ln(OR) +/- 1.96 * SE
        # SE = sqrt(1/a + 1/b + 1/c + 1/d)
        import math
        se = math.sqrt(1/a + 1/b + 1/c + 1/d)
        log_or = math.log(or_val)
        
        lower_log = log_or - 1.96 * se
        upper_log = log_or + 1.96 * se
        
        lower_ci = math.exp(lower_log)
        upper_ci = math.exp(upper_log)
        
        return {
            "test_name": "Odds Ratio (2x2)",
            "statistic": float(or_val),
            "ci_lower": float(lower_ci),
            "ci_upper": float(upper_ci),
            "p_value": None, 
            "interpretation":f"Riesgo aumentado ({or_val:.2f}x)" if or_val > 1 else f"Factor protector ({or_val:.2f}x)"
        }
        
    except Exception as e:
        return {"error": f"Error Odds Ratio: {str(e)}"}

# ==============================================================================
# 5. NUEVAS UTILIDADES AVANZADAS (TIER 1 PRO)
# ==============================================================================

def calculate_mean_ci(data: Union[pd.Series, np.ndarray], alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Calcula el Intervalo de Confianza para la media de una muestra.
    Retorna: (lower_bound, upper_bound, mean)
    """
    try:
        a = np.array(data)
        a = a[~np.isnan(a)]
        n = len(a)
        if n < 2: return (np.nan, np.nan, np.nan)
        
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + (1 - alpha)) / 2., n-1)
        return (m - h, m + h, m)
    except:
        return (np.nan, np.nan, np.nan)

def calculate_power_analysis(effect_size: float, nobs: int, alpha: float = 0.05, ratio: float = 1.0) -> float:
    """
    Calcula la potencia estadística (Power) post-hoc test t.
    """
    try:
        from statsmodels.stats.power import TTestIndPower
        analysis = TTestIndPower()
        # power = func(effect_size, nobs, alpha, ratio)
        # solve_power arguments: effect_size, nobs1, alpha, power, ratio
        # Se pasa power=None para que lo calcule
        power = analysis.solve_power(effect_size=effect_size, nobs1=nobs, alpha=alpha, ratio=ratio)
        return float(power)
    except:
        return 0.0

def pairwise_welch_games_howell(df: pd.DataFrame, val_col: str, group_col: str) -> pd.DataFrame:
    """
    Implementación robusta manual tipo 'Games-Howell' usando pairwise T-Welch con corrección de Bonferroni/Holm.
    Ideal para ANOVA heterocedástico significativo.
    """
    try:
        from statsmodels.stats.multitest import multipletests
        import itertools
        
        groups = df[group_col].dropna().unique()
        if len(groups) < 2: return pd.DataFrame()
        
        combs = list(itertools.combinations(groups, 2))
        results = []
        pvals = []
        
        for g1, g2 in combs:
            d1 = df[df[group_col] == g1][val_col].dropna()
            d2 = df[df[group_col] == g2][val_col].dropna()
            
            # Welch T-test (equal_var=False)
            t_stat, p_val = stats.ttest_ind(d1, d2, equal_var=False)
            results.append([str(g1), str(g2), p_val])
            pvals.append(p_val)
            
        # Corrección Multiplicidad (Holm es mejor que Bonferroni, pero Bonferroni es standard Games-Howell approx)
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='holm')
        
        final_table = []
        for i, (g1, g2, p_raw) in enumerate(results):
            final_table.append({
                "Grupo A": g1,
                "Grupo B": g2,
                "P-Value Raw": p_raw,
                "P-Value Corrected": pvals_corrected[i],
                "Significativo": "✅ Sí" if reject[i] else "❌ No"
            })
            
        return pd.DataFrame(final_table)
    except Exception as e:
        return pd.DataFrame()

def get_hypothesis_text(test_name: str, v1: str, v2: str = "") -> Dict[str, str]:
    """Genera texto educativo sobre las hipótesis del test seleccionado."""
    
    texts = {"H0": "", "H1": ""}
    
    if "Student" in test_name or "Welch" in test_name:
        texts["H0"] = f"La media de **{v1}** es igual a la media de **{v2}**."
        texts["H1"] = f"Existe una diferencia significativa entre las medias de **{v1}** y **{v2}**."
    elif "Mann" in test_name or "Wilcoxon" in test_name:
        texts["H0"] = f"La distribución (mediana) de **{v1}** es igual a la de **{v2}**."
        texts["H1"] = f"Existe una diferencia en la distribución entre **{v1}** y **{v2}**."
    elif "ANOVA" in test_name:
         texts["H0"] = f"Las medias de **{v1}** son iguales en todos los grupos de **{v2}**."
         texts["H1"] = f"Al menos un grupo de **{v2}** tiene una media diferente en **{v1}**."
    elif "Kruskal" in test_name:
         texts["H0"] = f"Las medianas de **{v1}** son iguales en todos los grupos."
         texts["H1"] = f"Al menos un grupo presenta una mediana diferente."
    elif "Pearson" in test_name:
        texts["H0"] = f"No existe correlación lineal entre **{v1}** y **{v2}** (r=0)."
        texts["H1"] = f"Existe una correlación lineal significativa entre **{v1}** y **{v2}**."
    elif "Spearman" in test_name:
        texts["H0"] = f"No existe correlación entre los rangos de **{v1}** y **{v2}**."
        texts["H1"] = f"Existe correlación significativa entre las variables."
    elif "Chi" in test_name or "Fisher" in test_name:
        texts["H0"] = f"Las variables **{v1}** y **{v2}** son independientes (no asociadas)."
        texts["H1"] = f"Existe una asociación dependiente entre **{v1}** y **{v2}**."
        
    return texts
