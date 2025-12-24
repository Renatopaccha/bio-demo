import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from .calculos_tasas import (
    calcular_ajuste_directo,
    calcular_ajuste_indirecto,
    grafico_forestplot_smr,
    grafico_comparacion_tasas,
    validar_datos_tasas,
    recomendar_metodo,
    _normalizar_clave
)


# ==============================================================================
# UI: MÉTODO DIRECTO (AJUSTE DE TASAS)
# ==============================================================================

def render_metodo_directo_mejorado(df_original):
    """
    Render del Método Directo con validación y visualización mejorada.
    """
    st.markdown("### 🟢 Ajuste Directo de Tasas")
    st.caption("Estandarización usando una Población de Referencia (ej: OMS, Nacional).")
    
    if df_original is None or df_original.empty:
        st.warning("⚠️ No hay datos para procesar.")
        return
    
    # ====== PASO 1: SELECCIÓN DE DATOS ======
    st.markdown("#### 1️⃣ Selecciona tus datos")
    
    cols = df_original.columns.tolist()
    c1, c2, c3 = st.columns(3)
    
    col_grupo = c1.selectbox("Columna de Grupos (Edad):", cols, key="d_grupo")
    col_casos = c2.selectbox("Columna de Casos:", 
                             df_original.select_dtypes('number').columns, 
                             key="d_casos")
    col_pob = c3.selectbox("Columna de Población Local:", 
                          df_original.select_dtypes('number').columns, 
                          key="d_pob")
    
    # ====== VALIDACIÓN PREVIA ======
    advertencias = validar_datos_tasas(df_original, col_casos, col_pob)
    if advertencias:
        for adv in advertencias:
            st.warning(adv)
    
    # ====== RECOMENDACIÓN DE MÉTODO ======
    st.markdown("---")
    recomendacion = recomendar_metodo(df_original, col_casos)
    st.info(recomendacion)
    
    # ====== PASO 2: POBLACIÓN ESTÁNDAR ======
    st.markdown("#### 2️⃣ Define la Población Estándar de Referencia")
    
    grupos_unicos = sorted(df_original[col_grupo].astype(str).unique())
    
    plantilla = pd.DataFrame({
        'Grupo': grupos_unicos,
        'Poblacion_Std': 100000  # Valor por defecto
    })
    
    df_std_input = st.data_editor(
        plantilla,
        hide_index=True,
        column_config={
            "Grupo": st.column_config.Column(disabled=True),
            "Poblacion_Std": st.column_config.NumberColumn(
                "Población Estándar", 
                required=True,
                format="%d"
            )
        },
        key="editor_std_directo",
        use_container_width=True
    )
    
    # ====== PASO 3: CONFIGURACIÓN ======
    st.markdown("#### 3️⃣ Configuración del cálculo")
    
    col_mult, col_button = st.columns([1, 2])
    
    multiplicador = col_mult.selectbox(
        "Expresar tasa por:",
        [1000, 10000, 100000],
        index=2,
        key="d_mult"
    )
    
    calcular = col_button.button("🚀 Calcular Ajuste Directo", 
                                 type="primary", 
                                 key="btn_directo")
    
    # ====== CÁLCULO Y RESULTADOS ======
    if calcular:
        resultado = calcular_ajuste_directo(
            df_original, df_std_input,
            col_grupo, col_casos, col_pob,
            "Poblacion_Std", multiplicador
        )
        
        # Guardar en session state
        st.session_state.ultimo_resultado_directo = resultado
    
    # Mostrar resultados si existen (CORREGIDO: Verificar que no sea None)
    if 'ultimo_resultado_directo' in st.session_state and st.session_state.ultimo_resultado_directo is not None:
        resultado = st.session_state.ultimo_resultado_directo
        
        if "error" in resultado:
            st.error(f"❌ {resultado['error']}")
        else:
            # ====== RESULTADOS - KPIs ======
            st.markdown("#### 📊 Resultados del Ajuste Directo")
            
            k1, k2, k3 = st.columns(3)
            
            k1.metric(
                "Tasa Bruta",
                f"{resultado['tasa_bruta']:.2f}",
                help="Tasa sin ajustar por estructura de edad"
            )
            k2.metric(
                "Tasa Ajustada",
                f"{resultado['tasa_ajustada']:.2f}",
                help="Tasa estandarizada por población de referencia"
            )
            k3.metric(
                "IC 95%",
                f"[{resultado['ic_lower']:.2f} - {resultado['ic_upper']:.2f}]",
                help="Intervalo de Confianza (Gamma - Fay & Feuer 1997)"
            )
            
            # ====== VISUALIZACIONES ======
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                st.markdown("**Comparación de Tasas:**")
                fig_comp = grafico_comparacion_tasas(
                    resultado,
                    f"por {multiplicador:,d}"
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            
            with col_graf2:
                st.markdown("**Diferencia Absoluta:**")
                diferencia = resultado['tasa_ajustada'] - resultado['tasa_bruta']
                st.metric(
                    "Ajuste por estandarización",
                    f"{diferencia:+.2f}",
                    help="Cambio debido al ajuste por edad"
                )
            
            # ====== DETALLES TÉCNICOS ======
            with st.expander("📋 Ver tabla de cálculos"):
                st.dataframe(
                    resultado['tabla_resumen'].rename(columns={
                        'Tasa_Esp': 'Tasa Específica',
                        'Poblacion_Std': 'Pob. Estándar',
                        'Casos_Esp_Std': 'Casos Esperados'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            # ====== METODOLOGÍA ======
            with st.expander("📚 Metodología"):
                st.markdown("""
                **Método Directo:**
                
                1. **Tasa Específica:** Casos / Población Local (por grupo)
                2. **Casos Esperados:** Tasa Específica × Población Estándar
                3. **Tasa Ajustada:** Σ(Casos Esperados) / Σ(Población Estándar)
                4. **IC (Gamma):** Utiliza distribución Gamma (Fay & Feuer, 1997)
                
                **Ventajas:**
                - Directamente interpretable
                - Estable para n > 50 y grupos > 5 casos
                
                **Limitaciones:**
                - Puede ser inestable con n < 50
                - Sensible a estructura de edad
                """)


# ==============================================================================
# UI: MÉTODO INDIRECTO (SMR)
# ==============================================================================

def render_metodo_indirecto_mejorado(df_original):
    """
    Render del Método Indirecto (SMR) con validación y tests estadísticos.
    """
    st.markdown("### 🔵 Ajuste Indirecto - SMR (Standard Mortality Ratio)")
    st.caption("Compara casos observados vs esperados usando tasa de referencia externa.")
    
    if df_original is None or df_original.empty:
        st.warning("⚠️ No hay datos para procesar.")
        return
    
    # ====== PASO 1: SELECCIÓN DE DATOS ======
    st.markdown("#### 1️⃣ Selecciona tus datos")
    
    cols = df_original.columns.tolist()
    c1, c2, c3 = st.columns(3)
    
    col_grupo = c1.selectbox("Columna de Grupos:", cols, key="i_grupo")
    col_obs = c2.selectbox("Columna Casos Observados:", 
                          df_original.select_dtypes('number').columns, 
                          key="i_obs")
    col_pob = c3.selectbox("Columna Población Local:", 
                          df_original.select_dtypes('number').columns, 
                          key="i_pob")
    
    # ====== RECOMENDACIÓN ======
    st.markdown("---")
    recomendacion = recomendar_metodo(df_original, col_obs)
    st.info(recomendacion)
    
    # ====== PASO 2: TASAS DE REFERENCIA ======
    st.markdown("#### 2️⃣ Define las tasas de referencia externa")
    
    grupos_unicos_ind = sorted(df_original[col_grupo].astype(str).unique())
    
    plantilla_ref = pd.DataFrame({
        'Grupo': grupos_unicos_ind,
        'Tasa_Ref': 0.005  # Valor por defecto
    })
    
    df_ref_input = st.data_editor(
        plantilla_ref,
        hide_index=True,
        column_config={
            "Grupo": st.column_config.Column(disabled=True),
            "Tasa_Ref": st.column_config.NumberColumn(
                "Tasa de Referencia",
                format="%.6f",
                required=True
            )
        },
        key="editor_ref_indirecto",
        use_container_width=True
    )
    
    st.caption("💡 Las tasas de referencia pueden ser tasas nacionales, mundiales, etc.")
    
    # ====== CÁLCULO ======
    st.markdown("---")
    
    if st.button("🚀 Calcular SMR", key="btn_indirecto", type="primary"):
        resultado = calcular_ajuste_indirecto(
            df_original, df_ref_input,
            col_grupo, col_obs, col_pob,
            "Tasa_Ref"
        )
        
        # Guardar en session state
        st.session_state.ultimo_resultado_indirecto = resultado
    
    # ====== RESULTADOS ======
    # CORREGIDO: Verificar que no sea None
    if 'ultimo_resultado_indirecto' in st.session_state and st.session_state.ultimo_resultado_indirecto is not None:
        resultado = st.session_state.ultimo_resultado_indirecto
        
        if "error" in resultado:
            st.error(f"❌ {resultado['error']}")
        else:
            st.markdown("#### 📊 Resultados del SMR")
            
            # KPIs principales
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Observados",
                int(resultado['observados']),
                help="Casos realmente observados"
            )
            col2.metric(
                "Esperados",
                f"{resultado['esperados']:.1f}",
                help="Casos esperados según tasas de referencia"
            )
            
            # SMR con color según significancia
            smr_color = "inverse" if resultado['es_significativo'] else "off"
            col3.metric(
                "SMR",
                f"{resultado['smr']:.3f}",
                delta=f"p={resultado['p_value']:.4f}",
                delta_color=smr_color,
                help="Razón Observado/Esperado"
            )
            
            sig_text = "✅ SIGNIFICATIVO" if resultado['es_significativo'] else "⚪ NO significativo"
            col4.metric(
                "Significancia",
                sig_text,
                help="¿El IC cruza 1.0? (α=0.05)"
            )
            
            # ====== INTERPRETACIÓN ======
            st.markdown("---")
            
            if resultado['es_significativo']:
                if resultado['smr'] > 1:
                    st.warning(resultado['interpretacion'])
                else:
                    st.success(resultado['interpretacion'])
            else:
                st.info(resultado['interpretacion'])
            
            st.caption(f"**IC 95% del SMR:** [{resultado['ic_lower']:.3f} - {resultado['ic_upper']:.3f}]")
            
            # ====== TABS DE VISUALIZACIÓN ======
            tab1, tab2, tab3 = st.tabs(["📈 Forest Plot", "📋 Tabla Detallada", "📚 Metodología"])
            
            with tab1:
                st.markdown("**Gráfico de Intervalo de Confianza:**")
                fig_forest = grafico_forestplot_smr(resultado)
                st.plotly_chart(fig_forest, use_container_width=True)
            
            with tab2:
                st.dataframe(
                    resultado['tabla_resumen'].rename(columns={
                        'Tasa_Ref': 'Tasa Referencia',
                        'Esperados': 'Casos Esperados'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            with tab3:
                st.markdown("""
                **Método Indirecto (SMR):**
                
                **Fórmula:**
                - SMR = Observados / Esperados
                - Esperados = Σ(Población Local × Tasa Referencia)
                
                **Interpretación:**
                - SMR = 1.0 → Igual al esperado
                - SMR > 1.0 → Más casos que esperado (exceso de riesgo)
                - SMR < 1.0 → Menos casos que esperado (protección)
                
                **IC (Chi-cuadrado exacto - Breslow & Day 1987):**
                - Si IC NO cruza 1.0 → Significativo (p < 0.05)
                - Si IC SÍ cruza 1.0 → NO significativo (p ≥ 0.05)
                
                **Ventajas:**
                - Estable para n pequeño
                - Robusto para grupos raros
                
                **Limitaciones:**
                - Requiere tasa de referencia externa
                - Menos directamente interpretable
                """)


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def render_ajuste_tasas():
    """
    Función principal: render dos tabs (Directo e Indirecto)
    """
    st.title("⚖️ Ajuste de Tasas (Estandarización Epidemiológica)")
    
    # Validación de datos
    if 'df_principal' not in st.session_state or st.session_state.df_principal is None:
        st.warning("⚠️ Carga datos primero en '🧹 Limpieza de Datos'")
        return
    
    df = st.session_state.df_principal
    
    # Inicializar session state
    if 'ultimo_resultado_directo' not in st.session_state:
        st.session_state.ultimo_resultado_directo = None
    if 'ultimo_resultado_indirecto' not in st.session_state:
        st.session_state.ultimo_resultado_indirecto = None
    
    # Info general
    st.info("""
    **¿Qué es el ajuste de tasas?**
    
    Estandarizar tasas permite comparar poblaciones con diferentes estructuras etarias.
    Elegir entre método **Directo** e **Indirecto** según tu n:
    - **Directo:** n > 50 y todos los grupos > 5 casos
    - **Indirecto:** n < 50 o algunos grupos con < 5 casos
    """)
    
    # TABS
    tab1, tab2 = st.tabs(["🟢 Método Directo", "🔵 Método Indirecto (SMR)"])
    
    with tab1:
        render_metodo_directo_mejorado(df)
    
    with tab2:
        render_metodo_indirecto_mejorado(df)


# Punto de entrada
if __name__ == "__main__" or hasattr(st, '_is_running_with_streamlit'):
    render_ajuste_tasas()
