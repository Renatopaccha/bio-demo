"""
Módulo de Análisis Multivariado (BioStat Easy)
----------------------------------------------
Este módulo agrupa técnicas para el análisis simultáneo de múltiples variables.
Incluye reducción de dimensiones (PCA, Factor Analysis), clasificación (LDA),
análisis de varianza multivariado (MANOVA) y agrupamiento (Clustering).

Responsabilidades:
- MANOVA (Multivariate ANOVA)
- Reducción de dimensiones (PCA, Factor Analysis)
- Discriminante Lineal (LDA)
- Matrices de correlación y covarianza
- Clustering (K-Means, Jerárquico)
- Validaciones (Bartlett Sphericity, KMO)

Autor: BioStat Easy Team
Versión: 2.5
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Dict, Union, Optional, List, Any, Tuple

# Imports de librerías externas opcionales pero recomendadas para este módulo
SKLEARN_AVAILABLE = False
STATSMODELS_AVAILABLE = False

try:
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import statsmodels.api as sm
    from statsmodels.multivariate.manova import MANOVA
    STATSMODELS_AVAILABLE = True
except ImportError:
    pass


# ==============================================================================
# 1. MANOVA
# ==============================================================================

def manova_analysis(df: pd.DataFrame, 
                   dependent_vars: List[str], 
                   independent_var: str) -> Dict[str, Any]:
    """
    Realiza un análisis multivariado de varianza (MANOVA).
    Requiere statsmodels.
    
    Args:
        df: DataFrame con los datos.
        dependent_vars: Lista de variables dependientes (continuas).
        independent_var: Variable independiente (categórica/grupos).
        
    Returns:
        Diccionario con resultados del test (Wilks' lambda, Pillai's trace, etc).
    """
    if not STATSMODELS_AVAILABLE:
        return {"error": "Statsmodels no instalado. Requerido para MANOVA."}
        
    # Limpieza
    cols_needed = dependent_vars + [independent_var]
    
    # 1. Asegurar tipos numéricos en dependientes
    # El independent_var se deja tal cual (puede ser string/categórico)
    temp_df = df[cols_needed].copy()
    
    for col in dependent_vars:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
        
    data_clean = temp_df.dropna()
    
    if data_clean.empty:
        return {"error": "Datos vacíos tras limpieza (posibles valores no numéricos)."}
        
    # Statsmodels MANOVA syntax: "dep1 + dep2 ~ independ"
    # Asegurar que column names no tengan espacios o caracteres raros para la fórmula
    # (Esto podría requerir renombrado si los nombres son complejos, pero asumimos nombres limpios o quoteados)
    # Una forma segura es renombrar cols temporalmente si fallara, pero por ahora confiamos en nombres standard.
    
    formula = ' + '.join(dependent_vars) + f' ~ {independent_var}'
    
    try:
        manova = MANOVA.from_formula(formula, data=data_clean)
        res = manova.mv_test()
        
        # Extraer resultados clave (Wilks' Lambda suele ser el standard)
        # El objeto mv_test retorna un objeto complejo, parseamos lo básico
        summary_frame = res.results[independent_var]['stat']
        
        return {
            "test_name": "MANOVA",
            "n_obs": len(data_clean),
            "results_table": summary_frame, # DataFrame con Wilks, Pillai, Hotelling, Roy
            "wilks_lambda_p": summary_frame.loc["Wilks' lambda", 'Pr > F'],
            "interpretation": "Diferencias Multivariadas Significativas" 
                              if summary_frame.loc["Wilks' lambda", 'Pr > F'] < 0.05 
                              else "No Significativo"
        }
    except Exception as e:
        return {"error": f"Error en MANOVA: {str(e)}"}


# ==============================================================================
# 2. ANÁLISIS FACTORIAL Y PCA
# ==============================================================================

def pca_analysis(df: pd.DataFrame, 
                n_components: Optional[int] = None) -> Dict[str, Any]:
    """
    Análisis de Componentes Principales (PCA).
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "Scikit-learn requerido para PCA."}
        
    # Solo numéricas
    data_num = df.select_dtypes(include=[np.number]).dropna()
    if data_num.shape[0] < 2 or data_num.shape[1] < 2:
        return {"error": "Dimensiones insuficientes para PCA."}
        
    # Estandarización (Critical for PCA)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_num)
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data_scaled)
    
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    
    # Loadings (Correlación variable-componente)
    # Loadings = Eigenvectors * sqrt(Eigenvalues)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, 
                             index=data_num.columns, 
                             columns=[f"PC{i+1}" for i in range(loadings.shape[1])])
    
    return {
        "method": "PCA",
        "n_components": pca.n_components_,
        "explained_variance_ratio": explained_var.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "loadings": loadings_df,
        "transformed_data": pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(components.shape[1])]),
        "eigenvalues": pca.explained_variance_.tolist()
    }


def factor_analysis(df: pd.DataFrame, 
                   n_factors: Optional[int] = None) -> Dict[str, Any]:
    """
    Análisis Factorial Exploratorio (EFA).
    """
    if not SKLEARN_AVAILABLE: return {"error": "Scikit-learn requerido."}
    
    data_num = df.select_dtypes(include=[np.number]).dropna()
    
    # Estandarización
    data_scaled = StandardScaler().fit_transform(data_num)
    
    fa = FactorAnalysis(n_components=n_factors, rotation=None) # Sklearn basic rotation support is limited out of box
    factors = fa.fit_transform(data_scaled)
    
    return {
        "method": "Factor Analysis",
        "loadings": pd.DataFrame(fa.components_.T, index=data_num.columns),
        "variances": fa.noise_variance_.tolist(), # Uniqueness specifics
        "transformed_data": pd.DataFrame(factors)
    }


# ==============================================================================
# 3. LDA (Discrimante Lineal)
# ==============================================================================

def lda_analysis(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Linear Discriminant Analysis.
    """
    if not SKLEARN_AVAILABLE: return {"error": "Scikit-learn requerido."}
    
    # Limpieza
    data = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_clean = data.drop(columns='target')
    y_clean = data['target']
    
    lda = LinearDiscriminantAnalysis()
    X_r = lda.fit_transform(X_clean, y_clean)
    
    return {
        "method": "LDA",
        "classes": lda.classes_.tolist(),
        "explained_variance_ratio": lda.explained_variance_ratio_.tolist(),
        "means": lda.means_.tolist(),
        "scalings": pd.DataFrame(lda.scalings_, index=X_clean.columns), # Coeficientes discriminantes
        "transformed_data": X_r
    }


# ==============================================================================
# 4. CORRELACIÓN Y COVARIANZA
# ==============================================================================

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna matriz de correlación de Pearson."""
    return df.corr(method='pearson')

def covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna matriz de covarianza."""
    # Estandarizar no es obligatorio para covarianza pura, pero covarianza depende de escala.
    return df.cov()

def partial_correlation(df: pd.DataFrame, control_vars: List[str]) -> pd.DataFrame:
    """
    Calcula matriz de correlación parcial controlando por 'control_vars'.
    Implementación básica usando la inversa de la matriz de covarianza (precision matrix).
    """
    # Nota: Pingouin es mejor para esto, pero usaremos manipulación de covarianza con sklearn/numpy
    # Partial Corr(i, j | k) ~ - P_ij / sqrt(P_ii * P_jj) donde P es matriz de precisión
    
    data_num = df.select_dtypes(include=[np.number]).dropna()
    
    if data_num.shape[1] < 2: return pd.DataFrame()
    
    try:
        cov = data_num.cov()
        inv_cov = np.linalg.inv(cov) # Precision matrix
        
        # Convertir a correlación parcial
        d = np.diag(inv_cov)
        part_corr = -inv_cov / np.sqrt(np.outer(d, d))
        
        # La diagonal debe ser 1 (técnicamente el negativo de la inversa normalizada da 1 en diag?)
        # Corr parcial de x consigo misma es 1.
        np.fill_diagonal(part_corr, 1.0)
        
        return pd.DataFrame(part_corr, index=data_num.columns, columns=data_num.columns)
        
    except np.linalg.LinAlgError:
        return pd.DataFrame() # Singular matrix


# ==============================================================================
# 5. CLUSTERING
# ==============================================================================

def kmeans_clustering(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
    """Clustering K-Means."""
    if not SKLEARN_AVAILABLE: return {"error": "Scikit-learn requerido."}
    
    data_num = df.select_dtypes(include=[np.number]).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_num)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Métricas de calidad
    silhouette = metrics.silhouette_score(X_scaled, labels)
    
    return {
        "method": "K-Means",
        "n_clusters": n_clusters,
        "labels": labels.tolist(),
        "centroids": kmeans.cluster_centers_.tolist(), # Centros en espacio escalado
        "inertia": kmeans.inertia_,
        "silhouette_score": silhouette
    }


def hierarchical_clustering(df: pd.DataFrame, method: str = 'ward') -> Dict[str, Any]:
    """Clustering Jerárquico Aglomerativo."""
    if not SKLEARN_AVAILABLE: return {"error": "Scikit-learn requerido."}
    
    data_num = df.select_dtypes(include=[np.number]).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_num)
    
    # Usando AgglomerativeClustering de sklearn para consistencia de API
    # O linkage de scipy para dendrogramas. Devolveremos matriz linkage Z para plotting.
    
    Z = linkage(X_scaled, method=method)
    
    # Calcular Coplhenetic Correlation Coeff
    c, coph_dists = stats.cophenet(Z, pdist(X_scaled))
    
    return {
        "method": "Hierarchical",
        "linkage_matrix": Z,
        "cophenetic_correlation": c,
        "n_obs": len(data_num)
    }


# ==============================================================================
# 6. VALIDACIÓN MULTIVARIADA
# ==============================================================================

def bartlett_test_sphericity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prueba de Esfericidad de Bartlett.
    Comprueba si la matriz de correlación es significativamente diferente de la identidad.
    H0: Variables no correlacionadas (Matriz Identidad).
    """
    data = df.select_dtypes(include=[np.number]).dropna()
    n, p = data.shape
    
    if p < 2: return {"error": "Mínimo 2 variables."}
    
    corr_matrix = data.corr()
    det = np.linalg.det(corr_matrix)
    
    if det <= 0:
        return {"error": "Determinante no positivo (matriz singular)."}
        
    statistic = - (n - 1 - (2*p + 5)/6) * np.log(det)
    df_chi = (p * (p - 1)) / 2
    
    p_value = 1 - stats.chi2.cdf(statistic, df_chi)
    
    return {
        "test_name": "Bartlett Sphericity",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "is_suitable_for_reduction": p_value < 0.05
    }


def kaiser_meyer_olkin(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Índice KMO (Kaiser-Meyer-Olkin) de adecuación muestral.
    KMO Global.
    Formula: Sum(r_ij^2) / (Sum(r_ij^2) + Sum(p_ij^2))
    Donde r es obs correlation y p es partial correlation.
    """
    data = df.select_dtypes(include=[np.number]).dropna()
    if data.shape[1] < 2: return {"error": "Mínimo 2 variables."}
    
    # 1. Matriz Correlación (R)
    corr = data.corr().values
    
    # 2. Matriz Correlación Parcial (U) aprox via Inversa (P)
    try:
        inv_corr = np.linalg.inv(corr)
        d = np.sqrt(np.diag(inv_corr))
        part_corr = -inv_corr / np.outer(d, d)
    except np.linalg.LinAlgError:
         return {"error": "Matriz singular, no se puede calcular KMO."}
         
    # KMO Calculation
    # Excluir diagonal
    np.fill_diagonal(corr, 0)
    np.fill_diagonal(part_corr, 0)
    
    sum_r2 = np.sum(corr**2)
    sum_p2 = np.sum(part_corr**2)
    
    kmo_score = sum_r2 / (sum_r2 + sum_p2)
    
    # Interpretación
    if kmo_score < 0.5: interp = "Inaceptable"
    elif kmo_score < 0.6: interp = "Miserable"
    elif kmo_score < 0.7: interp = "Mediocre"
    elif kmo_score < 0.8: interp = "Regular"
    elif kmo_score < 0.9: interp = "Meritorio"
    else: interp = "Sobresaliente"
    
    return {
        "test_name": "KMO Index",
        "kmo_score": float(kmo_score),
        "interpretation": interp,
        "is_suitable": kmo_score > 0.6
    }
    
# ==============================================================================
# 7. OTROS / UTILIDADES
# ==============================================================================

def friedman_test(*args) -> Dict[str, Any]:
    """
    Wrapper (alias) para pruebas de Friedman (si no se importó desde inference).
    Implementación directa para independencia del módulo.
    """
    clean_args = [pd.to_numeric(a, errors='coerce').dropna() for a in args]
    # Check lengths... (Simplificado aqui, ver inference para full robustez)
    try:
        stat, p = stats.friedmanchisquare(*clean_args)
        return {"statistic": stat, "p_value": p}
    except Exception as e:
        return {"error": str(e)}

def correlation_circle(pca_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genera coordenadas para el Círculo de Correlaciones dado un resultado PCA.
    Usa los 'loadings' (correlación entre variable original y Componente).
    """
    if 'loadings' not in pca_results:
        return {"error": "Resultado PCA sin loadings."}
    
    loadings = pca_results['loadings']
    # Tomar PC1 y PC2 por defecto
    if loadings.shape[1] < 2:
        return {"error": "Se requieren al menos 2 componentes."}
        
    coords = loadings.iloc[:, 0:2].copy()
    coords.columns = ['x', 'y']
    
    return {
        "coordinates": coords,
        "variables": coords.index.tolist()
    }
