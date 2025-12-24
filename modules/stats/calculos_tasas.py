import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go


# ==============================================================================
# 1. UTILIDADES DE LIMPIEZA Y NORMALIZACIÓN
# ==============================================================================

def _normalizar_clave(serie):
    """
    Normaliza agresivamente los textos para cruce automático.
    Hace que '10-14', '10 a 14 años' y '10 - 14' sean idénticos.
    """
    return (serie.astype(str)
            .str.lower()
            .str.strip()
            .str.replace(" ", "")
            .str.replace("años", "")
            .str.replace("years", "")
            .str.replace("a", "-")
            .str.replace("to", "-"))


# ==============================================================================
# 2. INTERVALOS DE CONFIANZA - MÉTODOS ROBUSTOS EPIDEMIOLÓGICOS
# ==============================================================================

def _ic_byar_poisson(obs, conf=0.95):
    """
    Aproximación de Byar para IC de Poisson (Breslow & Day, 1987).
    Estándar de Oro en Epidemiología para observados.
    
    Args:
        obs: Número de eventos observados
        conf: Nivel de confianza (default 0.95)
    
    Returns:
        tuple: (lower, upper) límites del IC
    """
    if obs == 0:
        return 0.0, 3.69
    
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    lower = obs * (1 - 1/(9*obs) - z/3 * np.sqrt(1/obs))**3
    upper = (obs + 1) * (1 - 1/(9*(obs + 1)) + z/3 * np.sqrt(1/(obs + 1)))**3
    
    return max(0.0, lower), upper


def _ic_gamma_ajustada(tasa_ajustada, varianza_ajustada, conf=0.95):
    """
    IC Gamma (Fay & Feuer, 1997) para tasa ajustada.
    MÁS ROBUSTO que IC normal, especialmente para tasas bajas.
    
    VENTAJAS:
    - No genera límites negativos (problema con IC normal)
    - Más preciso para casos pequeños (n < 100)
    - Recomendado por CDC y OMS
    
    Args:
        tasa_ajustada: Tasa ajustada calculada
        varianza_ajustada: Varianza de la tasa
        conf: Nivel de confianza (default 0.95)
    
    Returns:
        tuple: (lower, upper) límites del IC
    """
    if tasa_ajustada <= 0 or varianza_ajustada <= 0:
        return 0.0, 0.0
    
    alpha = 1 - conf
    scale = varianza_ajustada / tasa_ajustada
    shape = (tasa_ajustada ** 2) / varianza_ajustada
    
    lower = stats.gamma.ppf(alpha / 2, a=shape, scale=scale)
    upper = stats.gamma.ppf(1 - alpha / 2, a=shape, scale=scale)
    
    return max(0.0, lower), upper


def _ic_chi2_exacto_smr(observados, esperados, conf=0.95):
    """
    IC 95% EXACTO para SMR usando distribución Chi-cuadrado.
    MÁS PRECISO que la aproximación de Byar (Breslow & Day, 1987).
    
    REFERENCIA: Breslow, N.E. & Day, N.E. (1987). Statistical Methods in 
    Cancer Research, Vol 2. International Agency for Research on Cancer.
    
    Args:
        observados: Casos observados
        esperados: Casos esperados
        conf: Nivel de confianza (default 0.95)
    
    Returns:
        tuple: (lower, upper) límites del SMR
    """
    alpha = 1 - conf
    
    if observados == 0:
        chi2_lower = 0
    else:
        chi2_lower = stats.chi2.ppf(alpha / 2, 2 * observados) / 2
    
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, 2 * (observados + 1)) / 2
    
    ic_lower = chi2_lower / esperados if esperados > 0 else 0
    ic_upper = chi2_upper / esperados if esperados > 0 else 0
    
    return ic_lower, ic_upper


# ==============================================================================
# 3. VALIDACIÓN DE DATOS
# ==============================================================================

def validar_datos_tasas(df, col_casos, col_pob):
    """
    Valida datos antes de calcular tasas.
    Retorna lista de advertencias.
    
    Args:
        df: DataFrame
        col_casos: Columna de casos
        col_pob: Columna de población
    
    Returns:
        list: Lista de advertencias (vacía si está OK)
    """
    advertencias = []
    
    # Casos no pueden ser > población
    if (df[col_casos] > df[col_pob]).any():
        advertencias.append("⚠️ ADVERTENCIA: Hay casos > población en algunos grupos")
    
    # Valores negativos
    if (df[col_casos] < 0).any() or (df[col_pob] < 0).any():
        advertencias.append("⚠️ ADVERTENCIA: Hay valores negativos (imposible en conteos)")
    
    # Población cero
    if (df[col_pob] == 0).any():
        advertencias.append("⚠️ ADVERTENCIA: Hay grupos con población=0")
    
    # NaN
    if df[[col_casos, col_pob]].isna().any().any():
        advertencias.append("⚠️ ADVERTENCIA: Hay valores faltantes (NaN)")
    
    return advertencias


def recomendar_metodo(df, col_casos):
    """
    Recomienda usar método Directo o Indirecto según los datos.
    
    RECOMENDACIÓN EPIDEMIOLÓGICA:
    - Si casos totales < 50: Usar INDIRECTO (SMR)
    - Si algún grupo tiene < 5 casos: Usar INDIRECTO (SMR)
    - Si casos > 50 y todos los grupos > 5: Usar DIRECTO
    
    Args:
        df: DataFrame
        col_casos: Columna de casos
    
    Returns:
        str: Recomendación formateada
    """
    total_casos = df[col_casos].sum()
    casos_min = df[col_casos].min()
    
    if total_casos < 50:
        return ("🔵 INDIRECTO (SMR) RECOMENDADO\n"
                f"Tu n es pequeño ({int(total_casos)} casos < 50)")
    elif casos_min < 5:
        return ("🔵 INDIRECTO (SMR) RECOMENDADO\n"
                f"Algunos grupos tienen < 5 casos (inestables)")
    else:
        return ("🟢 DIRECTO RECOMENDADO\n"
                f"Tu n es suficiente ({int(total_casos)} casos) y estable")


# ==============================================================================
# 4. MÉTODO DIRECTO - AJUSTE DE TASAS (MEJORADO)
# ==============================================================================

def calcular_ajuste_directo(df_local, df_std, col_grupo_local, col_casos, 
                           col_pob, col_pob_std, multiplicador=100000):
    """
    MÉTODO DIRECTO: Calcula tasa ajustada estandarizando por población.
    
    FÓRMULA:
    Tasa Ajustada = Σ(Tasa Específica_i × Población Estándar_i) / Σ(Población Estándar_i)
    
    VENTAJAS:
    - Directamente interpretable
    - Útil cuando n es grande
    
    DESVENTAJAS:
    - Inestable si n < 50 o hay grupos con < 5 casos
    
    Args:
        df_local: DataFrame con datos locales
        df_std: DataFrame con población estándar
        col_grupo_local: Columna de grupos en datos locales
        col_casos: Columna de casos
        col_pob: Columna de población local
        col_pob_std: Columna de población estándar en df_std
        multiplicador: Expresar tasa por (1000, 10000, 100000)
    
    Returns:
        dict: Diccionario con resultados completos
    """
    df = df_local.copy()
    std = df_std.copy()
    
    # Normalizar datos
    df[col_casos] = pd.to_numeric(df[col_casos], errors='coerce').fillna(0)
    df[col_pob] = pd.to_numeric(df[col_pob], errors='coerce').fillna(0)
    std[col_pob_std] = pd.to_numeric(std[col_pob_std], errors='coerce').fillna(0)
    
    # Validación
    advertencias = validar_datos_tasas(df, col_casos, col_pob)
    
    # Cruce Inteligente de Grupos
    
    # CORRECCIÓN: Identificar la columna de grupo en el DF estándar.
    # Es aquella columna que NO es la columna de población (col_pob_std).
    cols_posibles = [c for c in std.columns if c != col_pob_std]
    
    if not cols_posibles:
        return {"error": "No se encontró columna de grupo en los datos estándar."}
    
    col_grupo_std = cols_posibles[0] # Tomamos la primera columna que no es la de población
    
    df['_key'] = _normalizar_clave(df[col_grupo_local])
    std['_key'] = _normalizar_clave(std[col_grupo_std])
    
    merged = pd.merge(df, std, on='_key', how='inner')
    
    if merged.empty:
        return {
            "error": "No coinciden los grupos. Verifique que sus categorías (ej: rangos de edad) sean compatibles.",
            "advertencias": advertencias
        }
    
    # CÁLCULOS EPIDEMIOLÓGICOS
    
    # 1. Tasa Específica = Casos / Población Local
    merged['Tasa_Esp'] = np.where(merged[col_pob] > 0, 
                                   merged[col_casos] / merged[col_pob], 
                                   0.0)
    
    # 2. Casos Esperados en Estándar = Tasa Específica × Población Estándar
    merged['Casos_Esp_Std'] = merged['Tasa_Esp'] * merged[col_pob_std]
    
    # 3. Tasa Ajustada
    total_casos_esp = merged['Casos_Esp_Std'].sum()
    total_pob_std = merged[col_pob_std].sum()
    
    if total_pob_std == 0:
        return {
            "error": "La población estándar suma 0. Revise los datos ingresados.",
            "advertencias": advertencias
        }
    
    tasa_ajustada_raw = total_casos_esp / total_pob_std
    tasa_ajustada = tasa_ajustada_raw * multiplicador
    
    # 4. Varianza para IC (Método Keyfitz / Fay & Feuer 1997)
    # Var(tasa_adj) = Σ[(w_i)² × (casos_i / pob_i²)]
    # donde w_i = pob_std_i / total_pob_std
    
    merged['w_i'] = merged[col_pob_std] / total_pob_std
    merged['Var_Term'] = (merged['w_i']**2 * np.where(merged[col_pob] > 0,
                                   merged[col_casos] / (merged[col_pob]**2),
                                   0.0))
    
    var_ajustada_raw = merged['Var_Term'].sum()
    var_ajustada = var_ajustada_raw * (multiplicador**2)
    
    # 5. IC usando Gamma (Fay & Feuer 1997) - MÁS ROBUSTO
    ic_inf_raw, ic_sup_raw = _ic_gamma_ajustada(tasa_ajustada_raw, var_ajustada_raw)
    ic_inf = ic_inf_raw * multiplicador
    ic_sup = ic_sup_raw * multiplicador
    
    # 6. Tasa Bruta (sin ajuste)
    tasa_bruta = (merged[col_casos].sum() / merged[col_pob].sum()) * multiplicador if merged[col_pob].sum() > 0 else 0
    
    return {
        "tasa_bruta": tasa_bruta,
        "tasa_ajustada": tasa_ajustada,
        "ic_lower": ic_inf,
        "ic_upper": ic_sup,
        "casos_totales": int(merged[col_casos].sum()),
        "poblacion_total": int(merged[col_pob].sum()),
        "varianza": var_ajustada,
        "tabla_resumen": merged[[col_grupo_local, col_casos, col_pob, 'Tasa_Esp', 
                                 col_pob_std, 'Casos_Esp_Std']].copy(),
        "advertencias": advertencias,
        "multiplicador": multiplicador
    }


# ==============================================================================
# 5. MÉTODO INDIRECTO - SMR (MEJORADO)
# ==============================================================================

def calcular_ajuste_indirecto(df_local, df_ref, col_grupo_local, col_obs, 
                              col_pob, col_tasa_ref):
    """
    MÉTODO INDIRECTO: Calcula SMR (Standard Mortality Ratio).
    
    FÓRMULA:
    SMR = Observados / Esperados
    Esperados = Σ(Población Local_i × Tasa Referencia_i)
    
    VENTAJAS:
    - Estable incluso con n pequeño
    - Mejor para datos raros o pequeños
    
    DESVENTAJAS:
    - Menos directamente interpretable
    - Requiere tasa de referencia externa
    
    IC: Chi-cuadrado exacto (Breslow & Day, 1987)
    
    Args:
        df_local: DataFrame con datos locales
        df_ref: DataFrame con tasas de referencia
        col_grupo_local: Columna de grupos en datos locales
        col_obs: Columna de casos observados
        col_pob: Columna de población local
        col_tasa_ref: Columna de tasas de referencia
    
    Returns:
        dict: Diccionario con resultados completos
    """
    df = df_local.copy()
    ref = df_ref.copy()
    
    # Normalizar datos
    df[col_obs] = pd.to_numeric(df[col_obs], errors='coerce').fillna(0)
    df[col_pob] = pd.to_numeric(df[col_pob], errors='coerce').fillna(0)
    ref[col_tasa_ref] = pd.to_numeric(ref[col_tasa_ref], errors='coerce').fillna(0)
    
    # Cruce Inteligente
    
    # CORRECCIÓN: Identificar la columna de grupo en el DF referencia.
    # Es aquella columna que NO es la columna de tasa (col_tasa_ref).
    cols_posibles = [c for c in ref.columns if c != col_tasa_ref]
    
    if not cols_posibles:
        return {"error": "No se encontró columna de grupo en los datos de referencia."}
    
    col_grupo_ref = cols_posibles[0]
    
    df['_key'] = _normalizar_clave(df[col_grupo_local])
    ref['_key'] = _normalizar_clave(ref[col_grupo_ref])
    
    merged = pd.merge(df, ref, on='_key', how='inner')
    
    if merged.empty:
        return {"error": "No coinciden los grupos entre tus datos y la referencia."}
    
    # CÁLCULOS EPIDEMIOLÓGICOS
    
    # 1. Esperados = Población Local × Tasa Referencia
    merged['Esperados'] = merged[col_pob] * merged[col_tasa_ref]
    
    # 2. Totales
    observados = merged[col_obs].sum()
    esperados = merged['Esperados'].sum()
    
    if esperados == 0:
        return {"error": "El número de casos esperados es 0. Revise las tasas de referencia."}
    
    # 3. SMR
    smr = observados / esperados
    
    # 4. IC Chi-cuadrado EXACTO (Breslow & Day 1987) - MÁS PRECISO
    ic_lower_smr, ic_upper_smr = _ic_chi2_exacto_smr(observados, esperados)
    
    # 5. Significancia estadística
    es_significativo = not (ic_lower_smr <= 1.0 <= ic_upper_smr)
    
    # 6. P-value (test bilateral Poisson)
    if smr > 1:
        p_value = 1 - stats.poisson.cdf(observados - 1, esperados)
    else:
        p_value = stats.poisson.cdf(observados, esperados)
    
    # Asegurar que p_value está entre 0 y 1
    p_value = np.clip(p_value, 0, 1)
    
    # 7. Interpretación
    if es_significativo:
        if smr > 1:
            interpretacion = f"⚠️ EXCESO DE RIESGO SIGNIFICATIVO: {((smr-1)*100):.1f}% más casos de lo esperado"
        else:
            interpretacion = f"✅ PROTECCIÓN SIGNIFICATIVA: {((1-smr)*100):.1f}% menos casos de lo esperado"
    else:
        interpretacion = "ℹ️ SIN DIFERENCIA SIGNIFICATIVA"
    
    return {
        "observados": int(observados),
        "esperados": esperados,
        "smr": smr,
        "ic_lower": ic_lower_smr,
        "ic_upper": ic_upper_smr,
        "es_significativo": es_significativo,
        "p_value": p_value,
        "interpretacion": interpretacion,
        "tabla_resumen": merged[[col_grupo_local, col_obs, col_pob, 
                                col_tasa_ref, 'Esperados']].copy()
    }


# ==============================================================================
# 6. VISUALIZACIONES ESTADÍSTICAS
# ==============================================================================

def grafico_forestplot_smr(resultado_indirecto):
    """
    Forest plot para SMR (estándar en epidemiología).
    Visualiza si el IC cruza 1.0 (sin efecto).
    
    Args:
        resultado_indirecto: Dict retornado por calcular_ajuste_indirecto()
    
    Returns:
        plotly.graph_objects.Figure
    """
    smr = resultado_indirecto['smr']
    ic_lower = resultado_indirecto['ic_lower']
    ic_upper = resultado_indirecto['ic_upper']
    es_sig = resultado_indirecto['es_significativo']
    
    # Color según significancia
    color_ic = '#e74c3c' if es_sig else '#3498db'
    
    fig = go.Figure()
    
    # Línea de referencia en SMR=1.0 (sin efecto)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Sin efecto (SMR=1.0)",
                  annotation_position="right")
    
    # Intervalo de confianza (línea horizontal)
    fig.add_trace(go.Scatter(
        x=[ic_lower, ic_upper],
        y=[1, 1],
        mode='lines',
        line=dict(color=color_ic, width=4),
        name='IC 95%',
        hovertemplate=f'IC: [{ic_lower:.3f} - {ic_upper:.3f}]'
    ))
    
    # Punto SMR (diamante)
    fig.add_trace(go.Scatter(
        x=[smr],
        y=[1],  # <--- CORRECCIÓN AQUÍ: Agregado el valor [1]
        mode='markers',
        marker=dict(size=14, color=color_ic, symbol='diamond'),
        name=f'SMR = {smr:.3f}',
        hovertemplate=f'SMR: {smr:.3f}'
    ))
    
    fig.update_layout(
        title=f"Forest Plot - SMR = {smr:.3f} (IC 95%: {ic_lower:.3f} - {ic_upper:.3f})",
        xaxis_title="SMR (Razón Observado/Esperado)",
        yaxis=dict(showticklabels=False),
        height=350,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig


def grafico_comparacion_tasas(resultado_directo, multiplicador_label):
    """
    Gráfico de barras comparando Tasa Bruta vs Ajustada.
    
    Args:
        resultado_directo: Dict retornado por calcular_ajuste_directo()
        multiplicador_label: Label para el eje (ej: "por 100,000")
    
    Returns:
        plotly.graph_objects.Figure
    """
    datos = pd.DataFrame({
        'Tipo': ['Bruta', 'Ajustada'],
        'Tasa': [resultado_directo['tasa_bruta'], 
                resultado_directo['tasa_ajustada']]
    })
    
    fig = go.Figure(data=[
        go.Bar(
            name='Tasa',
            x=datos['Tipo'],
            y=datos['Tasa'],
            marker_color=['#95a5a6', '#2ecc71'],
            text=[f"{x:.2f}" for x in datos['Tasa']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Comparación de Tasas ({multiplicador_label})",
        yaxis_title="Tasa",
        height=400,
        showlegend=False
    )
    
    return fig
