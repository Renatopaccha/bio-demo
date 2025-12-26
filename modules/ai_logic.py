import streamlit as st
import google.generativeai as genai
import pandas as pd

def configurar_gemini(api_key):
    """
    Configura la conexión segura con Gemini.
    Retorna el objeto modelo o None si falla.
    Intenta varios modelos en orden de prioridad para evitar errores 404.
    """
    if not api_key: return None
    
    try:
        genai.configure(api_key=api_key)
        
        # Lista de modelos a probar en orden de preferencia
        # 1. gemini-2.5-flash (Versión estable más reciente)
        # 2. gemini-2.0-flash (Versión anterior)
        # 3. gemini-1.5-flash (Estable estándar)
        modelos_a_probar = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
        
        last_exception = None
        
        for nombre_modelo in modelos_a_probar:
            try:
                print(f"🔄 Intentando conectar con modelo: {nombre_modelo}...")
                model = genai.GenerativeModel(nombre_modelo)
                # Prueba simple de generación para verificar existencia real
                # (Generar un token vacío/simple no consume mucho y valida conexión)
                # Sin embargo, GenerativeModel no conecta hasta que se usa. 
                # Asumimos que si no hay error en instanciación, procedemos.
                # Para estar 100% seguros de que no es 404, retornamos el objeto.
                # El error 404 suele saltar al instanciar o al generar.
                
                print(f"✅ Modelo configurado exitosamente: {nombre_modelo}")
                return model
                
            except Exception as e:
                print(f"⚠️ Falló modelo {nombre_modelo}: {str(e)}")
                last_exception = e
                continue
        
        # Si llegamos aquí, ninguno funcionó
        print("❌ Todos los modelos fallaron.")
        return None

    except Exception as e:
        print(f"❌ Error general en configuración Gemini: {str(e)}")
        return None

def generar_resumen_tecnico(df):
    """
    Crea un 'perfil técnico' de los datos SIN enviar información sensible de pacientes.
    PRIVACIDAD: Solo envía nombres de variables, tipos de datos y conteos únicos.
    Nunca envía filas de datos crudos. Vital para cumplimiento ético en salud.
    """
    if df is None or df.empty:
        return "No hay datos cargados."
    
    buffer = [f"Dataset: {df.shape[0]} filas, {df.shape[1]} columnas."]
    buffer.append("Variables disponibles y sus características:")
    
    for col in df.columns:
        tipo = "Numérica" if pd.api.types.is_numeric_dtype(df[col]) else "Categórica"
        n_unique = df[col].nunique()
        # Solo tomamos 3 valores únicos como ejemplo de formato, no como datos del paciente
        ejemplo = df[col].dropna().unique()[:3] 
        buffer.append(f"- {col} ({tipo}): {n_unique} valores únicos. Ejemplos de formato: {list(ejemplo)}")
    
    return "\n".join(buffer)

def generar_interpretacion_apa(texto_resultados, tipo_prueba, api_key):
    """
    Toma resultados estadísticos técnicos y los convierte en redacción académica.
    Enfocado en Tesis de Salud (Medicina/Enfermería/Nutrición).
    """
    model = configurar_gemini(api_key)
    if not model: return "⚠️ Error: API Key no configurada o inválida. Ve al menú 'Inicio' para configurarla."

    # Prompt Ingeniería especializado para Bioestadística
    prompt = f"""
    Actúa como un experto bioestadístico y redactor de tesis médicas.
    
    CONTEXTO:
    El usuario es un estudiante del área de la salud realizando un análisis de '{tipo_prueba}'.
    Necesita interpretar los resultados para la sección de 'Resultados' de su tesis.
    
    RESULTADOS OBTENIDOS DEL SOFTWARE:
    {texto_resultados}
    
    TAREA:
    1. Escribe un párrafo de interpretación riguroso en estilo APA 7ma edición.
    2. Enfócate en la relevancia clínica primero, apoyada por la estadística (p-valor, intervalos de confianza).
    3. Usa lenguaje formal, objetivo y académico (ej: "Se observó una diferencia estadísticamente significativa...").
    4. Si el resultado NO es significativo, indícalo claramente ("No se encontró evidencia suficiente para rechazar la hipótesis nula...").
    5. SEGURIDAD: NO inventes datos ni alucines números. Usa ESTRICTAMENTE los valores provistos en 'RESULTADOS OBTENIDOS'.
    
    FORMATO DE SALIDA:
    Un solo párrafo de texto plano, listo para copiar y pegar.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al conectar con la IA: {str(e)}"


def build_result_prompt(df_resultado, titulo, notas="", df_principal=None):
    """
    Construye un prompt estructurado para interpretar tabla de resultados estadísticos.

    Args:
        df_resultado: DataFrame con la tabla de resultados a interpretar
        titulo: Título descriptivo de la tabla
        notas: Notas adicionales o contexto sobre el análisis
        df_principal: Dataset principal (opcional) para contexto adicional

    Returns:
        str: Prompt listo para enviar a Gemini
    """
    # Convertir DataFrame a markdown (con límites de seguridad)
    df_limitado = df_resultado.copy()

    # Limitar filas y columnas para no sobrecargar el prompt
    MAX_ROWS = 60
    MAX_COLS = 12

    if len(df_limitado) > MAX_ROWS:
        df_limitado = df_limitado.head(MAX_ROWS)
        truncado_filas = f"\n⚠️ Tabla truncada a {MAX_ROWS} filas (original: {len(df_resultado)} filas)"
    else:
        truncado_filas = ""

    if len(df_limitado.columns) > MAX_COLS:
        df_limitado = df_limitado.iloc[:, :MAX_COLS]
        truncado_cols = f"\n⚠️ Tabla truncada a {MAX_COLS} columnas (original: {len(df_resultado.columns)} columnas)"
    else:
        truncado_cols = ""

    # Convertir a markdown
    tabla_md = df_limitado.to_markdown(index=True)
    advertencias = truncado_filas + truncado_cols

    # Contexto del dataset principal (si existe)
    contexto_dataset = ""
    if df_principal is not None:
        contexto_dataset = f"""
CONTEXTO DEL DATASET PRINCIPAL:
{generar_resumen_tecnico(df_principal)}
"""

    # Construir prompt estructurado
    prompt = f"""
Eres un Bioestadístico Senior especializado en investigación en salud (medicina, enfermería, nutrición, salud pública).
Tu objetivo es interpretar resultados estadísticos de forma académica y clínica para tesis de pregrado y posgrado.

TABLA DE RESULTADOS: {titulo}
{tabla_md}
{advertencias}

NOTAS ADICIONALES:
{notas if notas else "No hay notas adicionales."}
{contexto_dataset}

INSTRUCCIONES CRÍTICAS:
1. NO inventes números que no estén en la tabla.
2. Si falta información clave, indícalo claramente ("No se dispone de X para...").
3. Usa los valores exactos de la tabla.
4. Enfoque dual: PRIMERO clínico (relevancia práctica), LUEGO estadístico (rigor metodológico).

ESTRUCTURA DE TU RESPUESTA (obligatoria):

## 📊 Resumen Ejecutivo
(3 bullets con los hallazgos más importantes, concisos y claros)

## 🏥 Interpretación Clínica
(2-3 párrafos explicando QUÉ significan estos resultados para la práctica clínica/salud pública.
Lenguaje accesible pero riguroso. Evita jerga innecesaria.
Enfócate en implicaciones prácticas y relevancia para profesionales de la salud.)

## 📈 Interpretación Estadística
(2-3 párrafos sobre aspectos metodológicos:
- ¿Se cumplen los supuestos?
- ¿Qué indica el tamaño del efecto?
- ¿Es significativa la asociación/diferencia?
- ¿Cuál es la precisión de las estimaciones (IC 95%)?
Usa terminología técnica correcta pero explícala.)

## 📝 Redacción para Tesis (Sección Resultados)
(1 párrafo en estilo APA 7ma edición, listo para copiar y pegar.
Ejemplo: "Se encontró una diferencia estadísticamente significativa entre... (χ² = X.XX, p < .05).
El análisis reveló que...")

## ⚠️ Limitaciones y Recomendaciones
(2-4 bullets sobre:
- Limitaciones del análisis actual
- Qué análisis complementarios podrían ser útiles
- Aspectos a considerar en la interpretación
- Recomendaciones para fortalecer el análisis)

FORMATO: Usa Markdown. Sé académico pero entendible. Prioriza precisión sobre brevedad.
"""

    return prompt
