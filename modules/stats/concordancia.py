"""
Módulo de Renderizado: Concordancia y Consistencia Interna
---------------------------------------------------------
Incluye Cohen's Kappa para variables categóricas y Bland-Altman para numéricas.
Refactorizado con Plotly y validaciones estrictas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def render_concordancia(df: pd.DataFrame = None):
    """
    Panel de Concordancia: Cohen's Kappa y Bland-Altman.
    Optimizado con Plotly (sin matplotlib).
    """
    st.subheader("🤝 Análisis de Concordancia Clínica")
    
    if df is None:
        if 'df_principal' in st.session_state:
            df = st.session_state.df_principal
        else:
            st.info("Cargue datos para realizar el análisis de concordancia.")
            return

    all_cols = df.columns.tolist()
    
    # 1. UI Selectores en Columnas
    col_sel_1, col_sel_2 = st.columns(2)
    with col_sel_1:
        var1 = st.selectbox("Observador 1 (Variable A):", all_cols, key='conc_v1')
    with col_sel_2:
        # Excluir la primera selección para evitar redundancia obvia
        opts_v2 = [c for c in all_cols if c != var1]
        var2 = st.selectbox("Observador 2 (Variable B):", opts_v2, key='conc_v2')

    measure = st.radio(
        "Método de Concordancia:", 
        ["Cohen's Kappa (Categóricos)", "Bland-Altman (Numéricos)"],
        horizontal=True,
        help="Use Kappa para clasificaciones (ej: Diagnóstico Sí/No). Use Bland-Altman para mediciones continuas (ej: Presión Arterial)."
    )
    
    st.markdown("---")
    
    if not var1 or not var2:
        st.warning("Seleccione dos variables para comparar.")
        return

    # --------------------------------------------------------------------------
    # MÉTODO 1: COHEN'S KAPPA (CATEGÓRICOS)
    # --------------------------------------------------------------------------
    if "Kappa" in measure:
        st.markdown("#### Cohen's Kappa (Inter-rater Agreement)")
        
        # Validación de Integridad
        data = df[[var1, var2]].dropna()
        if data.empty:
            st.error("❌ No hay datos coincidentes (filas completas) entre las variables seleccionadas.")
            return

        try:
            # Cálculo de Kappa
            k = cohen_kappa_score(data[var1], data[var2])
            
            # Interpretación Landis & Koch
            interp = ""
            color_interp = "black"
            
            if k < 0: 
                interp = "Pobre (Peor que el azar)"
                color_interp = "#e74c3c" # Rojo
            elif k < 0.20: 
                interp = "Leve"
                color_interp = "#e67e22" # Naranja
            elif k < 0.40: 
                interp = "Aceptable"
                color_interp = "#f1c40f" # Amarillo
            elif k < 0.60: 
                interp = "Moderada"
                color_interp = "#3498db" # Azul claro
            elif k < 0.80: 
                interp = "Sustancial"
                color_interp = "#2980b9" # Azul
            else: 
                interp = "Casi perfecta"
                color_interp = "#2ecc71" # Verde

            # KPI Principal
            col_res, col_info = st.columns([1, 2])
            with col_res:
                st.metric("Índice Kappa", f"{k:.3f}")
            with col_info:
                st.markdown(f"**Interpretación (Landis & Koch):**")
                st.markdown(f"<h4 style='color: {color_interp}; margin-top: -10px;'>{interp}</h4>", unsafe_allow_html=True)

            # Matriz de Confusión (Heatmap)
            st.subheader("Matriz de Coincidencia (Heatmap)")
            
            # Obtener etiquetas únicas ordenadas para ejes
            labels = sorted(list(set(data[var1].unique()) | set(data[var2].unique())))
            cm = confusion_matrix(data[var1], data[var2], labels=labels)
            
            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                color_continuous_scale='Blues',
                text_auto=True,
                labels=dict(x="Observador 2", y="Observador 1", color="Conteo"),
                title=f"Coincidencias: {var1} vs {var2}"
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.caption("ℹ️ La **diagonal principal** (cuadros más oscuros) representa los casos en que ambos observadores estuvieron de acuerdo.")

        except Exception as e:
            st.error(f"Error calculando Kappa. Asegúrese de que las variables sean categóricas (texto/factores). Detalles: {str(e)}")

    # --------------------------------------------------------------------------
    # MÉTODO 2: BLAND-ALTMAN (NUMÉRICOS)
    # --------------------------------------------------------------------------
    else:
        st.markdown("#### Gráfico de Bland-Altman")
        
        # 1. Validación Estricta de Tipos Numéricos
        if not pd.api.types.is_numeric_dtype(df[var1]) or not pd.api.types.is_numeric_dtype(df[var2]):
            st.error("❌ **Error Crítico:** Ha seleccionado variables de Texto/Categóricas para un test Numérico.")
            st.info("💡 Solución: Cambie el método a 'Cohen's Kappa' arriba, o seleccione variables numéricas (Edad, Peso, TAS, etc).")
            return

        try:
            # Preparación de datos (cast a float para seguridad)
            d1 = pd.to_numeric(df[var1], errors='coerce')
            d2 = pd.to_numeric(df[var2], errors='coerce')
            
            # DataFrame limpio
            data_ba = pd.DataFrame({'x': d1, 'y': d2}).dropna()
            
            if data_ba.empty:
                st.error("No hay datos numéricos válidos (posiblemente todo es texto o NaN).")
                return
            
            # Cálculos BA
            mean_vals = (data_ba['x'] + data_ba['y']) / 2
            diff_vals = data_ba['x'] - data_ba['y']
            
            md = np.mean(diff_vals)       # Bias
            sd = np.std(diff_vals, axis=0) # SD de la diferencia
            
            upper_loa = md + 1.96 * sd
            lower_loa = md - 1.96 * sd
            
            # Métricas Clave
            c1, c2, c3 = st.columns(3)
            c1.metric("Bias (Dif. Media)", f"{md:.3f}")
            c2.metric("Límite Sup (+1.96 SD)", f"{upper_loa:.3f}")
            c3.metric("Límite Inf (-1.96 SD)", f"{lower_loa:.3f}")

            # Gráfico Plotly
            fig_ba = go.Figure()

            # Puntos
            fig_ba.add_trace(go.Scatter(
                x=mean_vals, y=diff_vals,
                mode='markers',
                marker=dict(color='#3498db', size=7, opacity=0.6),
                name='Pacientes'
            ))

            # Líneas de Referencia
            # 1. Bias (Media)
            fig_ba.add_hline(y=md, line_dash="solid", line_color="red", annotation_text="Bias", annotation_position="top right")
            
            # 2. Límites de Acuerdo
            fig_ba.add_hline(y=upper_loa, line_dash="dash", line_color="gray", annotation_text="+1.96 SD")
            fig_ba.add_hline(y=lower_loa, line_dash="dash", line_color="gray", annotation_text="-1.96 SD")

            # Área Sombreada entre límites (Zona de Acuerdo)
            fig_ba.add_hrect(
                y0=lower_loa, y1=upper_loa,
                line_width=0, fillcolor="gray", opacity=0.1
            )

            fig_ba.update_layout(
                title="Plot de Diferencia vs Promedio",
                xaxis_title="Promedio de las dos mediciones",
                yaxis_title=f"Diferencia ({var1} - {var2})",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_ba, use_container_width=True)
            
            st.info("ℹ️ **Interpretación:** Si los puntos están dispersos aleatoriamente dentro de la banda gris, hay buen acuerdo. Si hay tendencia (embudo/línea), el error depende de la magnitud de la medida.")

        except Exception as e:
            st.error(f"Error calculando Bland-Altman: {str(e)}")
