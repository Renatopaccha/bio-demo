# modules/data_normalization.py

"""
╔════════════════════════════════════════════════════════════════╗
║  Módulo de Normalización de Datos para Análisis Estadístico    ║
║  ============================================================   ║
║  VERSIÓN: 1.0.0                                                ║
║  ESTATUS: INDEPENDIENTE (Sin dependencias críticas)            ║
║  SEGURIDAD: Máxima                                             ║
╚════════════════════════════════════════════════════════════════╝

Proporciona funciones reutilizables para:
1. Limpiar y validar tipos de datos
2. Normalizar strings (espacios, minúsculas)
3. Convertir valores con validación robusta
4. Manejar valores faltantes consistentemente

IMPORTANCIA: Este módulo es COMPLETAMENTE INDEPENDIENTE.
No modifica ningún archivo existente. Se usa SOLO si se importa explícitamente.

Uso:
    from modules.data_normalization import (
        normalizar_grupo, 
        normalizar_valor_numerico,
        validar_estructura_tukey
    )

CAMBIO DE LOG:
- v1.0.0: Creación inicial con 8 funciones core + tests
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Dict, Any, Optional
import logging
import sys
from pathlib import Path

# ========================================
# CONFIGURACIÓN SEGURA DE LOGGING
# ========================================

# Crear logger con nombre único para evitar conflictos
logger = logging.getLogger(__name__)

# Solo agregar handler si no existe ya
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

# Información de seguridad al importar
logger.info("✅ Módulo data_normalization cargado de forma segura")


# ========================================
# VERIFICACIÓN DE DEPENDENCIAS
# ========================================

def verificar_dependencias() -> bool:
    """
    Verifica que las dependencias necesarias estén disponibles.
    
    Returns:
        bool: True si todas las dependencias están OK
    """
    dependencias_requeridas = {
        'pandas': pd,
        'numpy': np,
    }
    
    todas_ok = True
    for nombre, modulo in dependencias_requeridas.items():
        try:
            _ = modulo.__version__
            logger.debug(f"✅ {nombre} {modulo.__version__} disponible")
        except Exception as e:
            logger.error(f"❌ Error con {nombre}: {str(e)}")
            todas_ok = False
    
    return todas_ok


# Verificar dependencias al cargar el módulo
if not verificar_dependencias():
    logger.warning("⚠️ Algunas dependencias pueden no estar disponibles")


# ========================================
# 1. NORMALIZACIÓN DE GRUPOS (STRINGS)
# ========================================

def normalizar_grupo(valor: Any, 
                     convertir_a_minuscula: bool = False,
                     eliminar_espacios: bool = True) -> str:
    """
    Normaliza un valor de grupo (nombre de categoría).
    
    ⚠️ FUNCIÓN SEGURA:
    - No modifica valor original
    - Retorna copia normalizada
    - Maneja excepciones sin propagar errores
    
    Proceso:
    1. Convierte a string de forma segura
    2. Elimina espacios al inicio/final
    3. Opcionalmente: convierte a minúsculas
    
    Args:
        valor: Valor a normalizar (puede ser str, int, float, None)
        convertir_a_minuscula: Si True, convierte a minúsculas (default: False)
        eliminar_espacios: Si True, limpia espacios (default: True)
    
    Returns:
        str: Valor normalizado (NUEVA COPIA, no modifica original)
    
    Raises:
        Ninguna: Retorna "NA" si hay error
    
    Examples:
        >>> valor_original = "  Grupo A  "
        >>> resultado = normalizar_grupo(valor_original)
        >>> print(valor_original)  # Sin cambios
        '  Grupo A  '
        >>> print(resultado)
        'Grupo A'
    """
    try:
        # Manejar valores nulos
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            return "NA"
        
        # Convertir a string (crea nueva copia)
        grupo_str = str(valor)
        
        # Eliminar espacios al inicio/final (crea nueva copia)
        if eliminar_espacios:
            grupo_str = grupo_str.strip()
        
        # Convertir a minúsculas (crea nueva copia)
        if convertir_a_minuscula:
            grupo_str = grupo_str.lower()
        
        # Validar que no esté vacío
        if not grupo_str:
            logger.warning(f"Valor tras normalización está vacío. Retornando 'NA'")
            return "NA"
        
        return grupo_str
    
    except Exception as e:
        logger.error(f"Error normalizando grupo '{valor}': {str(e)}. Retornando 'NA'")
        return "NA"


# ========================================
# 2. NORMALIZACIÓN DE VALORES NUMÉRICOS
# ========================================

def normalizar_valor_numerico(valor: Any, 
                               decimales: int = 4,
                               valor_defecto: float = 0.0) -> float:
    """
    Convierte valor a numérico de forma segura.
    
    ⚠️ FUNCIÓN SEGURA:
    - No modifica valor original
    - Retorna nueva copia numérica
    - Maneja NaN y valores inválidos
    - Nunca lanza excepciones (retorna fallback)
    
    Proceso:
    1. Intenta conversión directa con float()
    2. Si falla, trata de extraer números de strings
    3. Usa valor por defecto si falla
    
    Args:
        valor: Valor a convertir (str, int, float, None)
        decimales: Número de decimales para redondeo (default: 4)
        valor_defecto: Valor a retornar si falla (default: 0.0)
    
    Returns:
        float: Valor numérico normalizado (NUEVA COPIA)
    
    Raises:
        Ninguna: Siempre retorna un número
    
    Examples:
        >>> valor_original = "3.14159"
        >>> resultado = normalizar_valor_numerico(valor_original, decimales=2)
        >>> print(valor_original)  # Sin cambios
        '3.14159'
        >>> print(resultado)
        3.14
    """
    try:
        # Manejar nulos explícitamente
        if valor is None:
            return valor_defecto
        
        # Si ya es numérico
        if isinstance(valor, (int, float, np.integer, np.floating)):
            # Validar NaN
            if isinstance(valor, float) and np.isnan(valor):
                return valor_defecto
            return round(float(valor), decimales)
        
        # Si es string, limpiar y convertir
        if isinstance(valor, str):
            valor_limpio = valor.strip()
            
            # Intentar conversión directa
            try:
                return round(float(valor_limpio), decimales)
            except ValueError:
                # Intentar extraer números (ej: "p < 0.05" → 0.05)
                import re
                numeros = re.findall(r'-?\d+\.?\d*', valor_limpio)
                if numeros:
                    return round(float(numeros), decimales)
                else:
                    logger.warning(f"No se pudo extraer número de '{valor}'. Usando defecto: {valor_defecto}")
                    return valor_defecto
        
        # Fallback
        logger.warning(f"Tipo inesperado para conversión numérica: {type(valor)}. Usando defecto: {valor_defecto}")
        return valor_defecto
    
    except Exception as e:
        logger.error(f"Error normalizando número '{valor}': {str(e)}. Retornando {valor_defecto}")
        return valor_defecto


# ========================================
# 3. NORMALIZACIÓN DE PARES (TUKEY)
# ========================================

def normalizar_par_tukey(group1_raw: Any, 
                        group2_raw: Any) -> Tuple[str, str]:
    """
    Normaliza un par de grupos para comparación en Tukey HSD.
    
    ⚠️ FUNCIÓN SEGURA:
    - No modifica valores originales
    - Retorna tupla nueva con valores normalizados
    - Valida que no sean iguales
    - Ordena alfabéticamente para consistencia
    
    Valida que:
    1. Ambos grupos se normalicen correctamente
    2. No sean iguales después de normalización
    3. Retorna tupla ordenada (grupo1, grupo2) alfabéticamente
    
    Args:
        group1_raw: Primer grupo (sin normalizar)
        group2_raw: Segundo grupo (sin normalizar)
    
    Returns:
        Tuple[str, str]: (grupo_normalizado_1, grupo_normalizado_2) ordenados
    
    Raises:
        ValueError: Si los grupos son idénticos tras normalización
    
    Examples:
        >>> g1, g2 = "  Grupo A  ", "Grupo B"
        >>> resultado = normalizar_par_tukey(g1, g2)
        >>> print(g1, g2)  # Sin cambios
        ('  Grupo A  ', 'Grupo B')
        >>> print(resultado)
        ('Grupo A', 'Grupo B')
    """
    try:
        g1 = normalizar_grupo(group1_raw)
        g2 = normalizar_grupo(group2_raw)
        
        # Validar que no sean iguales
        if g1 == g2:
            raise ValueError(f"Los grupos normalizados son idénticos: '{g1}' == '{g2}'")
        
        # Retornar ordenados alfabéticamente para consistencia
        return tuple(sorted([g1, g2]))
    
    except Exception as e:
        logger.error(f"Error normalizando par Tukey: {str(e)}")
        raise


# ========================================
# 4. VALIDACIÓN DE ESTRUCTURA TUKEY
# ========================================

def validar_estructura_tukey(row_data: List[Any]) -> bool:
    """
    Valida que una fila de datos de Tukey tenga estructura correcta.
    
    ⚠️ FUNCIÓN SEGURA:
    - Solo VERIFICA, no modifica
    - Retorna bool sin efectos secundarios
    
    Estructura esperada:
    [group1, group2, meandiff, p-adj, lower, upper, reject]
    (7 elementos)
    
    Args:
        row_data: Lista con datos de una fila de Tukey
    
    Returns:
        bool: True si estructura es válida, False si no
    
    Raises:
        Ninguna
    """
    if not isinstance(row_data, (list, tuple)):
        logger.warning(f"row_data no es lista/tupla: {type(row_data)}")
        return False
    
    if len(row_data) < 7:
        logger.warning(f"row_data tiene {len(row_data)} elementos, se esperan ≥7")
        return False
    
    return True


# ========================================
# 5. EXTRACCIÓN ROBUSTA DE FILA TUKEY
# ========================================

def extraer_fila_tukey(row_data: List[Any], 
                       indice_fila: int = 0) -> Optional[Dict[str, Any]]:
    """
    Extrae y normaliza datos de una fila de Tukey.
    
    ⚠️ FUNCIÓN SEGURA:
    - No modifica row_data original
    - Retorna diccionario nuevo con datos extraídos y normalizados
    - Maneja todos los errores sin propagar excepciones
    - Genera warnings para inconsistencias
    
    Convierte:
        [group1, group2, meandiff, p-adj, lower, upper, reject]
    
    A diccionario validado:
        {
            'grupo1': str,
            'grupo2': str,
            'diferencia': float,
            'pvalue_ajustado': float,
            'ic_inferior': float,
            'ic_superior': float,
            'es_significativo': bool,
            'validacion': str  # "OK" o descripción de advertencia
        }
    
    Args:
        row_data: Lista con 7+ elementos (NO se modifica)
        indice_fila: Índice de fila (para logging)
    
    Returns:
        Dict o None si hay errores críticos
    
    Raises:
        Ninguna
    """
    try:
        # Validar estructura (sin modificar)
        if not validar_estructura_tukey(row_data):
            logger.error(f"Fila {indice_fila}: Estructura inválida")
            return None
        
        # Extraer campos con manejo robusto (crea copias)
        try:
            grupo1, grupo2 = normalizar_par_tukey(row_data[0], row_data[1])
        except Exception as e:
            logger.error(f"Fila {indice_fila}: Error normalizando grupos: {str(e)}")
            return None
        
        # Valores numéricos (crea copias normalizadas)
        meandiff = normalizar_valor_numerico(row_data[2], valor_defecto=0.0)
        pvalue_adj = normalizar_valor_numerico(row_data[3], valor_defecto=1.0)
        lower_ci = normalizar_valor_numerico(row_data[4], valor_defecto=np.nan)
        upper_ci = normalizar_valor_numerico(row_data[5], valor_defecto=np.nan)
        
        # Normalizar 'reject' a bool (crea copia)
        is_significant = normalizar_bool(row_data[6], fallback_pval=pvalue_adj)
        
        # Validación y warnings (sin modificar datos)
        advertencias = []
        
        # Advertencia: IC muy amplio
        if not np.isnan(lower_ci) and not np.isnan(upper_ci):
            amplitud_ic = upper_ci - lower_ci
            if amplitud_ic > 100:
                advertencias.append(f"IC muy amplio ({amplitud_ic:.1f})")
        
        # Advertencia: Inconsistencia reject vs pvalue
        reject_esperado = pvalue_adj < 0.05
        if is_significant != reject_esperado:
            advertencias.append(f"Inconsistencia: reject={is_significant}, p={pvalue_adj:.3f}")
        
        validacion_status = "OK" if not advertencias else f"WARN: {'; '.join(advertencias)}"
        
        # Retornar diccionario nuevo (copia de datos)
        return {
            'grupo1': grupo1,
            'grupo2': grupo2,
            'diferencia': meandiff,
            'pvalue_ajustado': pvalue_adj,
            'ic_inferior': lower_ci,
            'ic_superior': upper_ci,
            'es_significativo': is_significant,
            'validacion': validacion_status
        }
    
    except Exception as e:
        logger.error(f"Fila {indice_fila}: Error general: {str(e)}")
        return None


# ========================================
# 6. NORMALIZACIÓN DE BOOLEANOS
# ========================================

def normalizar_bool(valor: Any, 
                   fallback_pval: Optional[float] = None) -> bool:
    """
    Convierte valor a booleano de forma segura.
    
    ⚠️ FUNCIÓN SEGURA:
    - No modifica valor original
    - Retorna bool nuevo
    - Estrategia de fallback conservadora
    
    Estrategia (en orden):
    1. Si es bool, devolver como está
    2. Si es string: "true"/"yes"/"1"/"sí" → True, resto → False
    3. Si es numérico: 0 → False, resto → True
    4. Si es None: usar fallback_pval (p < 0.05 → True)
    5. Else: False (conservador)
    
    Args:
        valor: Valor a convertir (NO se modifica)
        fallback_pval: P-value para usar como fallback
    
    Returns:
        bool: Valor booleano (NUEVA COPIA)
    
    Raises:
        Ninguna
    """
    try:
        # Ya es bool
        if isinstance(valor, (bool, np.bool_)):
            return bool(valor)
        
        # Es string
        if isinstance(valor, str):
            return valor.strip().lower() in ['true', 'yes', '1', 'sí', 'si']
        
        # Es numérico
        if isinstance(valor, (int, float, np.integer, np.floating)):
            # Validar NaN
            if isinstance(valor, float) and np.isnan(valor):
                return False
            return valor != 0
        
        # Es None - usar fallback
        if valor is None:
            if fallback_pval is not None:
                return fallback_pval < 0.05
            return False
        
        # Fallback: False (conservador)
        logger.warning(f"No se pudo convertir a bool: {valor} (tipo: {type(valor)}). Retornando False")
        return False
    
    except Exception as e:
        logger.error(f"Error normalizando bool '{valor}': {str(e)}. Retornando False")
        return False


# ========================================
# 7. VALIDACIÓN DE DATASET COMPLETO
# ========================================

def validar_dataset_comparacion(df: pd.DataFrame,
                                var_num: str,
                                var_grp: str) -> Tuple[bool, List[str]]:
    """
    Valida que un dataset sea adecuado para comparaciones.
    
    ⚠️ FUNCIÓN SEGURA:
    - Solo VERIFICA, no modifica el dataframe
    - Retorna lista de advertencias sin efectos secundarios
    - No toca datos originales
    
    Verifica:
    - Dataset no nulo ni vacío
    - var_num es numérica
    - var_grp existe y tiene ≥2 grupos
    - Hay datos suficientes (N ≥ 3)
    
    Args:
        df: DataFrame (NO se modifica)
        var_num: Nombre columna numérica
        var_grp: Nombre columna de grupos
    
    Returns:
        Tuple[bool, List[str]]: (es_válido, lista_de_advertencias)
    
    Raises:
        Ninguna
    """
    advertencias = []
    
    # Validaciones básicas (solo lectura)
    if df is None or df.empty:
        return False, ["DataFrame nulo o vacío"]
    
    if var_num not in df.columns:
        return False, [f"Columna '{var_num}' no existe"]
    
    if var_grp not in df.columns:
        return False, [f"Columna '{var_grp}' no existe"]
    
    # Validar tipo numérico
    if not pd.api.types.is_numeric_dtype(df[var_num]):
        advertencias.append(f"'{var_num}' no es numérica")
    
    # Datos suficientes
    n_total = len(df)
    if n_total < 3:
        advertencias.append(f"N total ({n_total}) < 3")
    
    # Grupos suficientes
    n_grupos = df[var_grp].nunique()
    if n_grupos < 2:
        advertencias.append(f"Menos de 2 grupos ({n_grupos})")
    
    # Nulos excesivos
    missing_pct = df[[var_num, var_grp]].isna().any(axis=1).mean()
    if missing_pct > 0.5:
        advertencias.append(f"Más del 50% nulos ({missing_pct:.1%})")
    
    es_valido = len(advertencias) == 0
    
    return es_valido, advertencias


# ========================================
# 8. FUNCIONES DE UTILIDAD EXPORTABLES
# ========================================

def limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas.
    
    ⚠️ FUNCIÓN SEGURA:
    - Retorna NUEVO dataframe con columnas normalizadas
    - No modifica el dataframe original
    
    Transformaciones:
    - Elimina espacios al inicio/final
    - Reemplaza espacios internos por guiones bajos
    - Convierte a minúsculas
    
    Args:
        df: DataFrame original (NO se modifica)
    
    Returns:
        pd.DataFrame: NUEVO dataframe con columnas limpias
    
    Examples:
        >>> df_original = pd.DataFrame({'  Edad ': [1, 2], 'Peso KG': [60, 70]})
        >>> df_limpio = limpiar_nombres_columnas(df_original)
        >>> print(df_original.columns)  # Sin cambios
        Index(['  Edad ', 'Peso KG'], dtype='object')
        >>> print(df_limpio.columns)
        Index(['edad', 'peso_kg'], dtype='object')
    """
    try:
        # Crear COPIA del dataframe
        df_copia = df.copy()
        
        # Normalizar nombres (sobre la copia)
        df_copia.columns = [col.strip().replace(' ', '_').lower() for col in df_copia.columns]
        
        logger.info(f"✅ Columnas normalizadas: {list(df_copia.columns)}")
        return df_copia
    
    except Exception as e:
        logger.error(f"Error limpiando nombres: {str(e)}")
        return df.copy()  # Retornar copia sin cambios


# ========================================
# METADATOS Y VERSION
# ========================================

__version__ = "1.0.0"
__author__ = "Sistema de Análisis Estadístico"
__created__ = "2025-12-04"
__status__ = "INDEPENDIENTE - Sin dependencias críticas"
__safety_level__ = "MÁXIMA - Nunca modifica datos originales"

# ========================================
# EXPORTAR FUNCIONES PÚBLICAS
# ========================================

__all__ = [
    'normalizar_grupo',
    'normalizar_valor_numerico',
    'normalizar_par_tukey',
    'validar_estructura_tukey',
    'extraer_fila_tukey',
    'normalizar_bool',
    'validar_dataset_comparacion',
    'limpiar_nombres_columnas',
    'verificar_dependencias',
    '__version__',
    '__status__',
    '__safety_level__',
]
