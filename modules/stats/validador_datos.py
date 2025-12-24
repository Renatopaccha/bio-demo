
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional

def validar_datos_prevalentes(df: pd.DataFrame, col_grupo: str, col_casos: str, col_poblacion: str) -> Dict[str, Any]:
    """
    Valida la estructura básica de datos para ajuste directo de tasas.

    Args:
        df: DataFrame con datos de tasas por grupo de edad
        col_grupo: Nombre de la columna con grupos
        col_casos: Nombre de la columna con conteos
        col_poblacion: Nombre de la columna con denominador

    Returns:
        Dict con resultados de la validación
    """
    
    resultados = {
        "es_valido": True,
        "errores": [],
        "advertencias": [],
        "sugerencias": [],
        "estadisticas": {
            "n_filas": 0,
            "n_grupos": 0,
            "total_casos": 0,
            "total_poblacion": 0,
            "n_nans": 0,
            "n_infinitos": 0,
            "n_valores_negativos": 0,
            "n_inconsistencias": 0,
            "n_poblacion_cero": 0
        },
        "datos_limpios": pd.DataFrame(),
        "reporte_detallado": ""
    }

    # 1. Validación de Existencia
    columnas_faltantes = []
    if col_grupo not in df.columns:
        columnas_faltantes.append(col_grupo)
    if col_casos not in df.columns:
        columnas_faltantes.append(col_casos)
    if col_poblacion not in df.columns:
        columnas_faltantes.append(col_poblacion)
    
    if columnas_faltantes:
        resultados["es_valido"] = False
        resultados["errores"].append(f"Faltan columnas requeridas: {', '.join(columnas_faltantes)}")
        resultados["reporte_detallado"] = "Error Crítico: No se encontraron todas las columnas necesarias."
        return resultados

    # Trabajar con una copia para no modificar el original
    df_temp = df.copy()
    
    # 2. Validación de Tipos (Intento de conversión)
    try:
        df_temp[col_casos] = pd.to_numeric(df_temp[col_casos], errors='coerce')
        df_temp[col_poblacion] = pd.to_numeric(df_temp[col_poblacion], errors='coerce')
    except Exception as e:
        resultados["es_valido"] = False
        resultados["errores"].append(f"Error al convertir columnas numéricas: {str(e)}")
        return resultados

    # Estadísticas básicas preliminares
    resultados["estadisticas"]["n_filas"] = len(df_temp)
    resultados["estadisticas"]["n_grupos"] = df_temp[col_grupo].nunique()
    
    # 3. Validación de Valores Faltantes (NaNs)
    # Contamos NaNs que resultan de datos faltantes o de la conversión fallida
    n_nans_casos = df_temp[col_casos].isna().sum()
    n_nans_poblacion = df_temp[col_poblacion].isna().sum()
    n_nans_total = n_nans_casos + n_nans_poblacion
    resultados["estadisticas"]["n_nans"] = int(n_nans_total)

    if n_nans_total > 0:
        pct_nans_casos = (n_nans_casos / len(df_temp)) * 100 if len(df_temp) > 0 else 0
        pct_nans_poblacion = (n_nans_poblacion / len(df_temp)) * 100 if len(df_temp) > 0 else 0
        
        if pct_nans_casos == 100 or pct_nans_poblacion == 100:
            resultados["es_valido"] = False
            resultados["errores"].append("Una o más columnas numéricas contienen 100% de valores vacíos (NaN).")
        elif pct_nans_casos > 30 or pct_nans_poblacion > 30:
            resultados["advertencias"].append("Más del 30% de los datos en casos o población son vacíos.")
        else:
            resultados["advertencias"].append(f"{int(n_nans_total)} valores faltantes (NaN) detectados.")
            resultados["sugerencias"].append("Se encontraron valores vacíos. Se recomienda eliminar esas filas o rellenar con 0.")

    # 4. Validación de Valores Especiales (Infinitos y Negativos)
    # Infinitos
    n_inf_casos = np.isinf(df_temp[col_casos]).sum()
    n_inf_poblacion = np.isinf(df_temp[col_poblacion]).sum()
    n_inf_total = n_inf_casos + n_inf_poblacion
    resultados["estadisticas"]["n_infinitos"] = int(n_inf_total)

    if n_inf_total > 0:
        resultados["es_valido"] = False
        resultados["errores"].append(f"Se encontraron {int(n_inf_total)} valores infinitos. Revise sus datos.")
        resultados["sugerencias"].append("Eliminar filas con valores infinitos.")

    # Negativos (ignorar NaNs en esta comprobación)
    n_neg_casos = (df_temp[col_casos] < 0).sum()
    n_neg_poblacion = (df_temp[col_poblacion] < 0).sum()
    n_neg_total = n_neg_casos + n_neg_poblacion
    resultados["estadisticas"]["n_valores_negativos"] = int(n_neg_total)

    if n_neg_total > 0:
        resultados["es_valido"] = False
        resultados["errores"].append(f"Se encontraron {int(n_neg_total)} valores negativos. Las tasas no pueden ser negativas.")
        resultados["sugerencias"].append("Revisar filas con valores negativos.")

    # Si hay errores críticos de estructura, detener aquí
    if not resultados["es_valido"] and (n_inf_total > 0 or n_neg_total > 0 or pct_nans_casos == 100):
        # Se genera reporte preliminar
        resultados["reporte_detallado"] = generar_reporte_validacion(resultados, formato='texto')
        return resultados

    # Limpieza preliminar para validaciones lógicas (eliminar NaNs para el resto de checks)
    df_clean = df_temp.dropna(subset=[col_casos, col_poblacion]).copy()
    
    # Calcular totales si es posible
    if not df_clean.empty:
        resultados["estadisticas"]["total_casos"] = int(df_clean[col_casos].sum())
        resultados["estadisticas"]["total_poblacion"] = int(df_clean[col_poblacion].sum())

    # 5. Validación de Divisiones por Cero
    grupos_poblacion_cero = df_clean[df_clean[col_poblacion] == 0]
    n_pob_cero = len(grupos_poblacion_cero)
    resultados["estadisticas"]["n_poblacion_cero"] = n_pob_cero
    
    if not grupos_poblacion_cero.empty:
        grupos_afectados = grupos_poblacion_cero[col_grupo].tolist()
        msg_grupos = f"({', '.join(map(str, grupos_afectados[:3]))}{'...' if len(grupos_afectados)>3 else ''})"
        resultados["advertencias"].append(f"La población es 0 en k={len(grupos_afectados)} grupos {msg_grupos}. Causará división por cero.")
        resultados["sugerencias"].append("Eliminar grupos con población 0 o asignar un valor mínimo (ej: 1).")

    # 6. Validación de Consistencia Lógica (Casos > Población)
    inconsistencias = df_clean[df_clean[col_casos] > df_clean[col_poblacion]]
    resultados["estadisticas"]["n_inconsistencias"] = len(inconsistencias)
    
    if not inconsistencias.empty:
        grupos_inconsistentes = inconsistencias[col_grupo].tolist()
        msg_grupos = f"({', '.join(map(str, grupos_inconsistentes[:3]))}{'...' if len(grupos_inconsistentes)>3 else ''})"
        resultados["advertencias"].append(f"Hay {len(grupos_inconsistentes)} grupos donde los Casos superan a la Población {msg_grupos}.")
        resultados["sugerencias"].append("Verifique que las columnas de Casos y Población no estén invertidas o sean erróneas.")

    # 7. Detección de Outliers (Tasa > 0.5)
    mask_pop_ok = df_clean[col_poblacion] > 0
    if mask_pop_ok.any():
        # Usar .loc para evitar SettingWithCopyWarning implícito
        tasas = df_clean.loc[mask_pop_ok, col_casos] / df_clean.loc[mask_pop_ok, col_poblacion]
        outliers = df_clean.loc[mask_pop_ok][tasas > 0.5]
        
        if not outliers.empty:
            grupos_outliers = outliers[col_grupo].tolist()
            resultados["advertencias"].append(f"Se detectaron tasas inusualmente altas (>50%) en {len(grupos_outliers)} grupos.")

    # 8. Validación de Formato de Grupos (Fuzzy)
    if not _detectar_formato_grupos(df_clean[col_grupo]):
        resultados["advertencias"].append("La columna de grupos no parece contener patrones numéricos de edad habituales (ej: '0-4', '5-9').")
    
    # Generar Datos Limpios ("Mejor esfuerzo" retornados como base para steps siguientes)
    resultados["datos_limpios"] = df_clean
    
    # Construir reporte final
    resultados["reporte_detallado"] = generar_reporte_validacion(resultados, formato='texto')

    return resultados

def limpiar_datos_automatico(df: pd.DataFrame, col_grupo: str, col_casos: str, col_poblacion: str, estrategia: str = 'conservadora') -> Tuple[pd.DataFrame, List[str]]:
    """
    Aplica correcciones automáticas a datos con problemas leves.

    Args:
        df: DataFrame original
        col_grupo: Columna de grupos
        col_casos: Columna de casos
        col_poblacion: Columna de población
        estrategia: 'conservadora', 'moderada', 'agresiva'

    Returns:
        (DataFrame corregido, Lista de cambios realizados)
    """
    df_out = df.copy()
    cambios = []

    # Conversión de tipos inicial (segura)
    try:
        df_out[col_casos] = pd.to_numeric(df_out[col_casos], errors='coerce')
        df_out[col_poblacion] = pd.to_numeric(df_out[col_poblacion], errors='coerce')
    except:
        pass # Ya validado antes, aquí intentamos limpiar

    # 1. Estrategia Conservadora: Eliminar Inf y 100% NaN
    # Infinitos
    mask_inf = np.isinf(df_out[col_casos]) | np.isinf(df_out[col_poblacion])
    n_inf = mask_inf.sum()
    if n_inf > 0:
        df_out = df_out[~mask_inf]
        cambios.append(f"Eliminadas {n_inf} filas con valores infinitos.")

    # Filas con Todo NaN (o NaN en columnas clave)
    mask_nan_all = df_out[col_casos].isna() & df_out[col_poblacion].isna()
    n_nan_all = mask_nan_all.sum()
    if n_nan_all > 0:
        df_out = df_out[~mask_nan_all]
        cambios.append(f"Eliminadas {n_nan_all} filas con NaNs en ambas columnas numéricas.")
    
    # Filas con NaN en alguna (estrategia conservadora elimina si no se puede imputar fácil, 
    # pero aquí la conservadora dice 'Solo elimina filas con Inf o 100% NaN', 
    # asumimos que si falta UN valor se mantiene como NaN o se elimina?
    # El prompt dice: "'conservadora': Solo elimina filas con Inf o 100% NaN".
    # Así que mantenemos filas con 1 NaN por ahora.

    if estrategia == 'conservadora':
        return df_out.reset_index(drop=True), cambios

    # 2. Estrategia Moderada: Eliminar negativos, Poblacion 0 -> NaN
    if estrategia in ['moderada', 'agresiva']:
        # Eliminar negativos
        mask_neg = (df_out[col_casos] < 0) | (df_out[col_poblacion] < 0)
        n_neg = mask_neg.sum()
        if n_neg > 0:
            df_out = df_out[~mask_neg]
            cambios.append(f"Eliminadas {n_neg} filas con valores negativos.")
        
        # Poblacion 0 -> NaN (para evitar errores, o se podría eliminar)
        # Prompt: "convierte 0 poblaciones a NaN"
        mask_pob_0 = df_out[col_poblacion] == 0
        n_pob_0 = mask_pob_0.sum()
        if n_pob_0 > 0:
            df_out.loc[mask_pob_0, col_poblacion] = np.nan
            cambios.append(f"Convertida población 0 a NaN en {n_pob_0} filas.")
            
        if estrategia == 'moderada':
             # Limpiar NaNs resultantes si quedan
             # Ojo: si convertimos a NaN, luego qué?
             # Normalmente 'moderada' no rellena. Se devuelve así.
             return df_out.reset_index(drop=True), cambios

    # 3. Estrategia Agresiva: Rellenar NaN con 0, Eliminar Outliers extremos
    if estrategia == 'agresiva':
        # Rellenar NaN con 0
        n_nans = df_out[col_casos].isna().sum() + df_out[col_poblacion].isna().sum()
        if n_nans > 0:
            df_out[col_casos] = df_out[col_casos].fillna(0)
            df_out[col_poblacion] = df_out[col_poblacion].fillna(0)
            cambios.append(f"Rellenados {int(n_nans)} valores NaN con 0.")
            
        # Eliminar outliers extremos (Tasa > 1.0 (100%) por ejemplo, o > 0.5)
        # Prompt: "elimina outliers extremos"
        # Asumiremos Tasa > 1.0 es extremo (mas casos que poblacion)
        mask_valid_pop = df_out[col_poblacion] > 0
        if mask_valid_pop.any():
            tasa = df_out.loc[mask_valid_pop, col_casos] / df_out.loc[mask_valid_pop, col_poblacion]
            mask_outlier = (tasa > 1.0)
            # También debemos alinear el indice porque 'tasa' es un subset
             # Mejor hacerlo sobre todo el df
            tasa_full =  df_out[col_casos] / df_out[col_poblacion]
            # Ignorar donde pob es 0 o nan
            tasa_full = tasa_full.fillna(0) 
            # Si pob es 0, tasa es inf o nan. Ya manejado antes o nan.
            
            mask_extreme = tasa_full > 1.0
            n_extreme = mask_extreme.sum()
            if n_extreme > 0:
                df_out = df_out[~mask_extreme]
                cambios.append(f"Eliminadas {n_extreme} filas con tasa > 100% (Casos > Población).")

    return df_out.reset_index(drop=True), cambios

def generar_reporte_validacion(resultado: Dict, formato: str = 'texto') -> str:
    """
    Genera un reporte legible para mostrar al usuario.

    Args:
        resultado: Diccionario retornado por validar_datos_prevalentes
        formato: 'texto', 'markdown', 'html'

    Returns:
        String con el reporte formateado
    """
    stats = resultado["estadisticas"]
    errores = resultado["errores"]
    advertencias = resultado["advertencias"]
    sugerencias = resultado["sugerencias"]
    es_valido = resultado["es_valido"]
    
    if formato == 'texto':
        lines = []
        lines.append("REPORTE DE VALIDACIÓN DE DATOS")
        lines.append("="*30)
        lines.append(f"Estado: {'VÁLIDO' if es_valido else 'INVÁLIDO (Errores Críticos)'}")
        lines.append(f"Filas: {stats['n_filas']}, Grupos: {stats['n_grupos']}")
        lines.append(f"Casos Totales: {stats['total_casos']:,}, Pob. Total: {stats['total_poblacion']:,}")
        
        if errores:
            lines.append("\nERRORES:")
            for e in errores: lines.append(f" - {e}")
            
        if advertencias:
            lines.append("\nADVERTENCIAS:")
            for a in advertencias: lines.append(f" - {a}")
            
        if sugerencias:
            lines.append("\nSUGERENCIAS:")
            for s in sugerencias: lines.append(f" - {s}")
            
        return "\n".join(lines)

    elif formato == 'markdown':
        norm_case = "SÍ" if es_valido else "NO"
        adv_suffix = " (con advertencias)" if es_valido and advertencias else ""
        
        md = []
        md.append("## 📊 Reporte de Validación de Datos")
        md.append("### ✅ Estado General")
        md.append(f"- **Es válido para cálculo**: {norm_case}{adv_suffix}")
        md.append(f"- **Total de filas**: {stats['n_filas']}")
        md.append(f"- **Total de grupos**: {stats['n_grupos']}")
        md.append(f"- **Total de casos**: {stats['total_casos']:,}")
        md.append(f"- **Total de población**: {stats['total_poblacion']:,}")
        
        md.append(f"### ❌ Errores ({len(errores)})")
        if not errores:
            md.append("*(Sin errores críticos)*")
        else:
            for e in errores:
                md.append(f"- {e}")
        
        md.append(f"### ⚠️ Advertencias ({len(advertencias)})")
        if not advertencias:
            md.append("*(Sin advertencias)*")
        else:
            for i, a in enumerate(advertencias, 1):
                md.append(f"{i}. {a}")
        
        md.append(f"### 💡 Sugerencias Automáticas")
        if not sugerencias:
            md.append("*(Sin sugerencias)*")
        else:
            for i, s in enumerate(sugerencias, 1):
                md.append(f"{i}. {s}")
                
        return "\n".join(md)

    elif formato == 'html':
        # Simple HTML wrapper, similar to markdown but with tags
        html = []
        html.append("<div style='font-family: sans-serif;'>")
        html.append("<h2>📊 Reporte de Validación de Datos</h2>")
        
        color_status = "green" if es_valido else "red"
        status_text = "SÍ" if es_valido else "NO"
        html.append(f"<p><strong>Es válido para cálculo:</strong> <span style='color:{color_status}; font-weight:bold'>{status_text}</span></p>")
        
        html.append("<ul>")
        html.append(f"<li>Total de filas: {stats['n_filas']}</li>")
        html.append(f"<li>Total de grupos: {stats['n_grupos']}</li>")
        html.append(f"<li>Total de casos: {stats['total_casos']:,}</li>")
        html.append(f"<li>Total de población: {stats['total_poblacion']:,}</li>")
        html.append("</ul>")
        
        html.append(f"<h3>❌ Errores ({len(errores)})</h3>")
        if errores:
            html.append("<ul>")
            for e in errores: html.append(f"<li style='color:red'>{e}</li>")
            html.append("</ul>")
        else:
            html.append("<p><i>(Sin errores críticos)</i></p>")
            
        html.append(f"<h3>⚠️ Advertencias ({len(advertencias)})</h3>")
        if advertencias:
            html.append("<ul>")
            for a in advertencias: html.append(f"<li style='color:orange'>{a}</li>")
            html.append("</ul>")
        else:
            html.append("<p><i>(Sin advertencias)</i></p>")

        html.append("</div>")
        return "\n".join(html)
    
    return "Formato no soportado."

# --- Funciones Helper Privadas ---

def _detectar_formato_grupos(series_grupo: pd.Series) -> bool:
    """
    Detecta si los grupos siguen un patrón numérico o de edad esperado.
    """
    if series_grupo.empty:
        return True
        
    sample = series_grupo.astype(str).head(10).tolist()
    # Patrones comunes: "10-14", "80+", "5 a 9", "<1"
    # Buscaremos al menos un dígito
    patron_digito = r'\d+'
    
    coincidencias = [re.search(patron_digito, s) for s in sample]
    # Si al menos la mitad tiene números, asumimos que es edad o similar
    # Si ninguno tiene números (ej: "Grupo A", "Grupo B"), retorna False
    return any(coincidencias)

def _normalizar_nombres_grupos(series_grupo: pd.Series) -> pd.Series:
    """
    Normaliza los nombres de grupos (trim, lower) para comparaciones.
    """
    return series_grupo.astype(str).str.strip().str.lower()

def _calcular_estadisticas_grupos(df: pd.DataFrame, col_grupo: str, col_casos: str, col_poblacion: str) -> pd.DataFrame:
    """
    Calcula tasas por grupo. Helper para uso interno o futuro.
    """
    # Se asume df limpio de tipos
    res = df.copy()
    mask_ok = res[col_poblacion] > 0
    res['tasa'] = 0.0
    res.loc[mask_ok, 'tasa'] = res.loc[mask_ok, col_casos] / res.loc[mask_ok, col_poblacion]
    return res
