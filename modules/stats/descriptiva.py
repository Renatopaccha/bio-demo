"""
Módulo de Renderizado: Estadística Descriptiva (Formato Médico/Investigación)
-----------------------------------------------------------------------------
Genera tablas de estadísticas descriptivas con formato profesional (Tabla 1)
y análisis detallado de distribución.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional

import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

# Importamos la utilidad de guardado
from modules.utils import boton_guardar_tabla, card_container

from modules.stats.core import (
    calculate_descriptive_stats,
    detect_outliers_advanced,
    check_normality,
    check_homoscedasticity,
    check_symmetry_kurtosis,
    get_normal_curve_data,
    get_qq_coordinates,
    analyze_outlier_details,
    calculate_group_comparison,
    generate_table_one_structure,
    calculate_frequency_table,
    generate_crosstab_analysis,
    interpret_crosstab
)
from modules.stats.validators import validate_data_for_analysis


# --- HELPER: INTERPRETACIÓN AUTOMÁTICA (MEJORADO) ---
def generar_interpretacion_automatica(df_stats):
    """Genera texto explicativo, soportando análisis global o por grupos."""
    texto = "### 🧠 Análisis Automático de Resultados\n\n"
    
    # Detectar si es tabla segmentada
    tiene_grupos = 'Grupo' in df_stats.columns
    
    for index, row in df_stats.iterrows():
        var = row['Variable']
        grupo_txt = f" (Grupo: **{row['Grupo']}**)" if tiene_grupos else ""
        
        try:
            # Limpieza robusta de strings
            cv_raw = str(row.get('CV (%)', '0')).replace('%', '').strip()
            cv = float(cv_raw) if cv_raw and cv_raw != '-' else 0
            
            p_val = row.get('P-Normalidad', 0)
            is_normal = p_val > 0.05
            
            dist = "Normal" if is_normal else "No Normal"
            dispersión = "baja" if cv < 15 else ("moderada" if cv < 30 else "alta")
            central = f"Media (**{row.get('Media (IC)', row.get('Media (IC 95%)', '-'))}**)" if is_normal else f"Mediana (**{row.get('Mediana (IQR)', '-')}**)"
            
            texto += f"- **{var}**{grupo_txt}: Distribución **{dist}** (p={p_val:.3f}), dispersión {dispersión} (CV={cv}%). Se sugiere: {central}.\n"
        except Exception:
            continue  # Saltar filas con errores de formato
            
    return texto


# --- RENDER PRINCIPAL ---
def render_descriptiva(df: Optional[pd.DataFrame] = None, selected_vars: Optional[List[str]] = None):
    """
    Renderiza la sección de estadísticas descriptivas con formato de publicación médica.
    Organizada en 4 pestañas: Univariado, Comparativa, Gráficos y Tabla Inteligente.
    """
    st.subheader("📊 Estadística Descriptiva")
    
    # 1. Recuperación de Datos
    if df is None:
        if 'df_principal' in st.session_state:
            df = st.session_state.df_principal
        else:
            st.error("⚠️ No hay datos cargados en la sesión.")
            return
    
    # 2. Validación Básica
    valid, msg = validate_data_for_analysis(df)
    if not valid:
        st.error(f"⚠️ {msg}")
        return
    
    # 3. Selección de Variables (Global)
    if selected_vars is None:
        numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numericas:
            st.warning("El dataset no contiene variables numéricas.")
            return
            
        selected_vars = st.multiselect(
            "Seleccione las variables a analizar:",
            options=numericas,
            default=numericas[:5] if len(numericas) > 5 else numericas
        )
    
    if not selected_vars:
        st.info("Seleccione al menos una variable para continuar.")
        return
    
    st.divider()
    
    # 4. Estructura de Pestañas (AHORA SON 4)
    tab_univariado, tab_comparativa, tab_graficos, tab_inteligente = st.tabs([
        "📋 Univariado (Global)", 
        "⚔️ Comparativa (Tabla 1)", 
        "📊 Gráficos Diagnósticos",
        "📑 Tabla Inteligente (Paper)"
    ])
    
    # ==============================================================================
    # PESTAÑA 1: UNIVARIADO (GLOBAL)
    # ==============================================================================
    with tab_univariado:
        data_summary = []
        data_percentiles = []
        
        for var in selected_vars:
            stats = calculate_descriptive_stats(df[var])
            
            mean_val = stats.get('mean')
            ci_low = stats.get('ci95_lower')
            ci_high = stats.get('ci95_upper')
            
            if pd.notna(mean_val) and pd.notna(ci_low) and pd.notna(ci_high):
                mean_str = f"{mean_val:.2f} ({ci_low:.2f} - {ci_high:.2f})"
            else:
                mean_str = f"{mean_val:.2f}" if pd.notna(mean_val) else "-"
            
            median_val = stats.get('median')
            p25 = stats.get('p25')
            p75 = stats.get('p75')
            
            if pd.notna(median_val) and pd.notna(p25) and pd.notna(p75):
                median_str = f"{median_val:.2f} ({p25:.2f} - {p75:.2f})"
            else:
                median_str = f"{median_val:.2f}" if pd.notna(median_val) else "-"
                
            min_val = stats.get('min')
            max_val = stats.get('max')
            if pd.notna(min_val) and pd.notna(max_val):
                range_str = f"{min_val:.2f} - {max_val:.2f}"
            else:
                range_str = "-"
                
            cv_val = stats.get('cv')
            cv_str = f"{cv_val:.2f}%" if pd.notna(cv_val) else "-"
            
            row_summary = {
                'Variable': var,
                'N': stats.get('n', 0),
                'Media (IC 95%)': mean_str,
                'Mediana (P25 - P75)': median_str,
                'D.E.': stats.get('std'), 
                'Mín - Máx': range_str,
                'CV %': cv_str
            }
            data_summary.append(row_summary)
            
            row_perc = {
                'Variable': var,
                'P5': stats.get('p5'),
                'P10': stats.get('p10'),
                'P25 (Q1)': stats.get('p25'),
                'P50 (Mediana)': stats.get('p50'),
                'P75 (Q3)': stats.get('p75'),
                'P90': stats.get('p90'),
                'P95': stats.get('p95')
            }
            data_percentiles.append(row_perc)
            
        df_resumen = pd.DataFrame(data_summary)
        df_percentiles = pd.DataFrame(data_percentiles)
        
        st.markdown("### 📋 Resumen Descriptivo (Global)")
        st.caption("Reporte estándar con Intervalos de Confianza (IC 95%) y Rangos Intercuartílicos.")
        
        st.dataframe(
            df_resumen.style.format({"D.E.": "{:.2f}", "N": "{:.0f}"}),
            use_container_width=True
        )
        
        csv_resumen = df_resumen.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Descargar Tabla Resumen (CSV)", csv_resumen, "tabla1_resumen.csv", "text/csv", key='dl_summary')
        
        st.divider()
        
        with st.expander("📍 Ver Distribución Detallada (Percentiles)", expanded=False):
            cols_perc = ['P5', 'P10', 'P25 (Q1)', 'P50 (Mediana)', 'P75 (Q3)', 'P90', 'P95']
            st.dataframe(
                df_percentiles.style.format("{:.4f}", subset=cols_perc)
                .background_gradient(cmap="Blues", subset=cols_perc),
                use_container_width=True
            )
        
        st.divider()
        
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        with st.expander("📊 Variables Categóricas (Tablas de Frecuencia)", expanded=False):
            if not cat_cols:
                st.info("No se detectaron variables categóricas (texto/categoría) en el dataset.")
            else:
                sel_cats = st.multiselect(
                    "Seleccione variables categóricas para analizar:",
                    options=cat_cols,
                    default=[cat_cols[0]] if len(cat_cols) > 0 else None,
                    key='multiselect_freq_tables'
                )
                
                if sel_cats:
                    col_seg, _ = st.columns([1, 2])
                    with col_seg:
                        segment_var = st.selectbox(
                            "🔹 Segmentar por (opcional):",
                            options=["(Ninguno)"] + [c for c in cat_cols if c not in sel_cats],
                            help="Divide el análisis en subgrupos (ej. por Sexo)."
                        )
                    
                    for i, var_cat in enumerate(sel_cats):
                        st.markdown(f"### 📌 Variable: {var_cat}")
                        
                        if segment_var == "(Ninguno)":
                            freq_df = calculate_frequency_table(df[var_cat])
                            if not freq_df.empty:
                                st.dataframe(freq_df.style.format({'Porcentaje (%)': '{:.2f}%', 'Acumulado (%)': '{:.2f}%'}), use_container_width=True, hide_index=True)
                                df_plot = freq_df[freq_df['Categoría'] != 'TOTAL']
                                fig = px.bar(df_plot, x='Categoría', y='Frecuencia (n)', text='Porcentaje (%)', color='Frecuencia (n)', title=f"Distribución Global: {var_cat}", color_continuous_scale='Teal')
                                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            grupos = df[segment_var].dropna().unique()
                            grupos.sort()
                            tabs = st.tabs([f"📂 {g}" for g in grupos])
                            for idx, grupo in enumerate(grupos):
                                with tabs[idx]:
                                    st.caption(f"Subgrupo: **{segment_var} = {grupo}**")
                                    df_sub = df[df[segment_var] == grupo]
                                    freq_df_sub = calculate_frequency_table(df_sub[var_cat])
                                    if not freq_df_sub.empty:
                                        st.dataframe(freq_df_sub.style.format({'Porcentaje (%)': '{:.2f}%', 'Acumulado (%)': '{:.2f}%'}), use_container_width=True, hide_index=True)
                                        df_plot_sub = freq_df_sub[freq_df_sub['Categoría'] != 'TOTAL']
                                        fig_sub = px.bar(df_plot_sub, x='Categoría', y='Frecuencia (n)', text='Porcentaje (%)', color='Frecuencia (n)', title=f"{var_cat} ({segment_var}={grupo})", color_continuous_scale='Blues')
                                        fig_sub.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                        st.plotly_chart(fig_sub, use_container_width=True)
                                    else:
                                        st.warning("Sin datos para este grupo.")
                        st.divider()
                else:
                    st.info("Seleccione al menos una variable categórica.")
        
        st.divider()
        
        with st.expander("🔀 Tablas de Contingencia (Bivariado)", expanded=False):
            st.caption("Análisis de relación entre dos variables categóricas (Chi-Cuadrado).")
            c1, c2, c3 = st.columns(3)
            with c1:
                row_var = st.selectbox("Variable Fila:", options=cat_cols, index=0, key="ct_row")
            with c2:
                col_opts = [c for c in cat_cols if c != row_var]
                col_var = st.selectbox("Variable Columna:", options=col_opts, index=0, key="ct_col")
            with c3:
                seg_opts = ["(Ninguno)"] + [c for c in cat_cols if c not in [row_var, col_var]]
                seg_var = st.selectbox("🔹 Segmentar por:", options=seg_opts, key="ct_seg")
            
            metrics = st.multiselect(
                "Métricas a visualizar:",
                options=['n', 'row_pct', 'col_pct', 'total_pct'],
                default=['n', 'row_pct'],
                format_func=lambda x: {'n': 'Frecuencia (N)', 'row_pct': '% Fila', 'col_pct': '% Columna', 'total_pct': '% Total'}[x],
                key="ct_metrics"
            )
            
            st.divider()
            
            def render_crosstab_view(sub_df, r_var, c_var, mets, context_key):
                if sub_df.empty:
                    st.warning("No hay datos para este grupo.")
                    return
                res = generate_crosstab_analysis(sub_df, r_var, c_var, mets)
                if res['formatted_df'].empty:
                    st.warning("No se pudo generar la tabla (datos insuficientes).")
                    return
                st.write("📋 **Tabla Cruzada**")
                st.dataframe(res['formatted_df'], use_container_width=True)
                st.info(f"💡 **Análisis Inteligente:**\n\n{res['analysis_text']}")
                with st.expander("🎨 Ver Mapa de Calor", expanded=False):
                    raw_matrix = res['raw_n'].drop(index='TOTAL', columns='TOTAL', errors='ignore')
                    fig = px.imshow(raw_matrix, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title=f"Mapa de Calor: {r_var} vs {c_var}")
                    st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{context_key}")
            
            if row_var and col_var:
                if seg_var == "(Ninguno)":
                    render_crosstab_view(df, row_var, col_var, metrics, "global")
                else:
                    grupos = sorted(df[seg_var].dropna().unique())
                    tabs = st.tabs([f"📂 {g}" for g in grupos])
                    for i, grupo in enumerate(grupos):
                        with tabs[i]:
                            st.caption(f"Análisis para subgrupo: **{seg_var} = {grupo}**")
                            df_filtered = df[df[seg_var] == grupo]
                            render_crosstab_view(df_filtered, row_var, col_var, metrics, f"seg_{i}")
            else:
                st.info("Seleccione variables de Fila y Columna para comenzar.")
        
        with st.expander("🚨 Diagnóstico de Outliers (Detección Avanzada)", expanded=True):
            reporte_outliers = []
            for var in selected_vars:
                out_res = detect_outliers_advanced(df[var])
                for method in ['iqr', 'zscore', 'mad']:
                    indices = out_res.get(f'{method}_outliers', [])
                    if indices:
                        analysis = analyze_outlier_details(df[var], indices, method)
                        reporte_outliers.append({
                            'Variable': var,
                            'Método': method.upper(),
                            'Número': analysis['count'],
                            'Valores': analysis['values_str'],
                            'Acción': analysis['action']
                        })
            if reporte_outliers:
                df_reporte = pd.DataFrame(reporte_outliers)
                def style_action(val):
                    return 'color: red; font-weight: bold' if "Error" in val else ('color: orange; font-weight: bold' if "Revisar" in val else '')
                st.dataframe(df_reporte.style.applymap(style_action, subset=['Acción']), use_container_width=True)
            else:
                st.success("✅ Tus datos están PERFECTAMENTE LIMPIOS (No hay outliers).")
        
        with st.expander("✅ Validación de Supuestos (Normality Checks)", expanded=False):
            assumptions_data = []
            for var in selected_vars:
                norm_res = check_normality(df[var])
                stats_basic = calculate_descriptive_stats(df[var])
                shape_res = check_symmetry_kurtosis(stats_basic.get('skewness'), stats_basic.get('kurtosis'))
                is_normal = norm_res.get('conclusion') == "Normal"
                is_symmetric = shape_res.get('symmetry_eval') == "Simétrica"
                recomendacion = "✅ Paramétrico (T-Test/ANOVA)" if is_normal else ("✅ Paramétrico (Robustez TCL)" if is_symmetric and stats_basic.get('n', 0) > 30 else "⚠️ No Paramétrico (Mann-Whitney/Kruskal)")
                assumptions_data.append({
                    'Variable': var,
                    'Shapiro-Wilk p-val': norm_res.get('shapiro_p'),
                    'Skewness (Simetría)': stats_basic.get('skewness'),
                    'Kurtosis': stats_basic.get('kurtosis'),
                    'Normality Conclusion': norm_res.get('conclusion'),
                    'Recomendación': recomendacion
                })
            df_assumptions = pd.DataFrame(assumptions_data)
            def color_p(val):
                return '' if pd.isna(val) else ('color: green; font-weight: bold' if val > 0.05 else ('color: red; font-weight: bold' if val < 0.01 else 'color: orange; font-weight: bold'))
            st.dataframe(df_assumptions.style.format({'Shapiro-Wilk p-val': '{:.4f}', 'Skewness (Simetría)': '{:.2f}'}, na_rep="-").applymap(color_p, subset=['Shapiro-Wilk p-val']), use_container_width=True)
    
    # ==============================================================================
    # PESTAÑA 2: COMPARATIVA (TABLA 1)
    # ==============================================================================
    with tab_comparativa:
        st.markdown("### ⚔️ Comparativa de Grupos (Tabla 1)")
        cat_vars = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        group_col_comp = st.selectbox(
            "🔀 Agrupar por (Variable Categórica):",
            options=["(Ninguno)"] + cat_vars,
            index=0,
            key="group_selector_tab2",
            help="Seleccione grupos (ej: Tratamiento vs Control) para generar p-values y comparaciones."
        )
        
        if group_col_comp == "(Ninguno)":
            st.info("ℹ️ Seleccione una variable de agrupación arriba para generar la Tabla 1 con P-values.")
        else:
            n_groups = df[group_col_comp].nunique()
            if n_groups > 10:
                st.warning(f"⚠️ La variable '{group_col_comp}' tiene {n_groups} categorías. La tabla será muy ancha.")
            
            st.markdown(f"**Comparando grupos según:** `{group_col_comp}`")
            st.divider()
            
            available_vars = [c for c in df.columns if c != group_col_comp]
            default_mix = [v for v in selected_vars if v in available_vars]
            
            table1_vars = st.multiselect(
                "Seleccione variables para la Tabla 1:",
                options=available_vars,
                default=default_mix,
                key="table1_vars_selector"
            )
            
            if table1_vars:
                with st.spinner("Calculando estadísticas y P-Values..."):
                    df_table1 = generate_table_one_structure(df, table1_vars, group_col_comp)
                
                if not df_table1.empty:
                    def highlight_p(val):
                        return 'font-weight: bold; color: #2c3e50; background-color: #d5dbdb' if isinstance(val, str) and (val.startswith("<") or (val.replace('.','',1).isdigit() and float(val) < 0.05)) else ''
                    
                    st.dataframe(
                        df_table1.style.applymap(highlight_p, subset=['P-Value']).set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption("Nota: Para variables numéricas se muestra Mean ± SD (Paramétrico) o Median (IQR). Para categóricas n (%).")
                    
                    csv_table1 = df_table1.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Descargar Tabla 1 (CSV)",
                        data=csv_table1,
                        file_name=f"tabla1_comparativa_{group_col_comp}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No se pudieron generar resultados. Verifique que las variables tengan datos.")
            else:
                st.info("Seleccione al menos una variable para construir la tabla.")
    
    # ==============================================================================
    # PESTAÑA 3: GRÁFICOS DIAGNÓSTICOS
    # ==============================================================================
    with tab_graficos:
        st.markdown("### 📊 Inspección Visual")
        subtab_panel, subtab_comparar = st.tabs(["🔬 Panel Diagnóstico (Univariado)", "🆚 Comparación de Grupos"])
        
        with subtab_panel:
            st.caption("Diagnóstico de normalidad y evaluación de transformaciones.")
            c_sel, c_chk = st.columns([2, 1])
            with c_sel:
                var_plot = st.selectbox("Seleccione variable a diagnosticar:", options=selected_vars, key="selector_plot_var_panel")
            
            if var_plot:
                original_data = df[var_plot].dropna()
                data_plot = original_data
                is_transformed = False
                
                with c_chk:
                    st.write("")
                    apply_log = st.checkbox("🛠️ Simular Log10", help="Aplica Log10(x+1) para intentar normalizar.")
                
                if apply_log:
                    if (original_data < 0).any():
                        st.warning("⚠️ No se puede aplicar Log10: Hay valores negativos.")
                    else:
                        data_plot = np.log10(original_data + 1)
                        is_transformed = True
                        st.toast("Transformación Logarítmica Aplicada (Datos temporales)", icon="🛠️")
                
                n = len(data_plot)
                norm_res = check_normality(data_plot)
                p_val = norm_res.get('shapiro_p', 0.0)
                status_color = "#2ecc71" if p_val > 0.05 else "#e74c3c"
                status_text = "NORMAL" if p_val > 0.05 else "NO NORMAL"
                msg_trans = " (Transformado Log10)" if is_transformed else ""
                
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: rgba(240,242,246,0.5); border: 1px solid #ddd; margin-bottom: 15px;'>"
                    f"<b>Diagnóstico{msg_trans}:</b> <span style='color:{status_color}; font-weight:bold'>{status_text}</span> "
                    f"(Shapiro-Wilk p = <b>{p_val:.4f}</b>, n={n})</div>",
                    unsafe_allow_html=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    title_hist = f"Distribución: {var_plot}{msg_trans} | SW p={p_val:.3f}"
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=data_plot, histnorm='probability density', name='Histograma', marker_color='#3498db', opacity=0.6))
                    x_norm, y_norm = get_normal_curve_data(data_plot)
                    if x_norm is not None:
                        fig_hist.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Teórica', line=dict(color='orange', width=2)))
                    mean_val = data_plot.mean()
                    median_val = data_plot.median()
                    fig_hist.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color="red", annotation_text="Media")
                    fig_hist.add_vline(x=median_val, line_width=2, line_dash="dot", line_color="green", annotation_text="Med")
                    fig_hist.update_layout(title=title_hist, margin=dict(l=20, r=20, t=40, b=20), height=350)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    qq_data = get_qq_coordinates(data_plot)
                    if qq_data:
                        fig_qq = go.Figure()
                        fig_qq.add_trace(go.Scatter(x=qq_data['theoretical'], y=qq_data['sample'], mode='markers', name='Datos', marker=dict(color='#8e44ad')))
                        min_x, max_x = min(qq_data['theoretical']), max(qq_data['theoretical'])
                        fig_qq.add_trace(go.Scatter(x=[min_x, max_x], y=[qq_data['slope']*min_x + qq_data['intercept'], qq_data['slope']*max_x + qq_data['intercept']], mode='lines', name='Ideal', line=dict(color='red')))
                        fig_qq.update_layout(title="Q-Q Plot (Validación Normalidad)", xaxis_title="Teórico", yaxis_title="Muestral", margin=dict(l=20, r=20, t=40, b=20), height=350)
                        st.plotly_chart(fig_qq, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(x=data_plot, boxpoints='outliers', name=f"{var_plot}", marker_color='#f39c12', orientation='h'))
                    fig_box.update_layout(title="Box Plot (Outliers)", margin=dict(l=20, r=20, t=40, b=20), height=350)
                    st.plotly_chart(fig_box, use_container_width=True)
                
                with col4:
                    fig_viol = go.Figure()
                    fig_viol.add_trace(go.Violin(y=data_plot, box_visible=True, line_color='black', meanline_visible=True, fillcolor='#1abc9c', opacity=0.6, x0=f"{var_plot}", points='all', jitter=0.05, pointpos=0))
                    fig_viol.update_layout(title="Violin Híbrido (Densidad + Puntos)", margin=dict(l=20, r=20, t=40, b=20), height=350)
                    st.plotly_chart(fig_viol, use_container_width=True)
        
        with subtab_comparar:
            st.caption("Visualización comparativa con pruebas de hipótesis automáticas.")
            c1, c2 = st.columns(2)
            with c1:
                var_comp = st.selectbox("Variable Numérica:", options=selected_vars, key="sel_var_comp_tab3")
            with c2:
                cat_options = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                group_comp = st.selectbox("Variable de Agrupación:", options=["(Seleccione)"] + cat_options, key="sel_group_comp_tab3")
            
            if var_comp and group_comp != "(Seleccione)":
                df_c = df[[var_comp, group_comp]].dropna()
                stats_res = calculate_group_comparison(df, var_comp, group_comp)
                test_name = stats_res.get('test_used', 'N/A')
                p_val_str = stats_res.get('p_value_str', '-')
                stats_title = f"{var_comp} por {group_comp} | {test_name}: p={p_val_str}"
                
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_box_c = px.box(df_c, x=group_comp, y=var_comp, color=group_comp, points="all", title=f"Dispersión: {stats_title}", color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_box_c.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_box_c, use_container_width=True)
                
                with col_b:
                    fig_viol_c = px.violin(df_c, x=group_comp, y=var_comp, color=group_comp, box=True, points=False, title=f"Densidad: {stats_title}", color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_viol_c.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_viol_c, use_container_width=True)
                
                st.info(f"ℹ️ **Interpretación:** El gráfico izquierdo muestra cada paciente individual (puntos) y la mediana. El derecho muestra la 'forma' de los datos (densidad). El valor p (**{p_val_str}**) indica inferencia estadística.")
            elif group_comp == "(Seleccione)":
                st.info("👋 Seleccione una variable de agrupación para generar los gráficos comparativos.")
    
    # ==============================================================================
    # PESTAÑA 4: TABLA INTELIGENTE (PERSONALIZABLE PRO)
    # ==============================================================================
    with tab_inteligente:
        st.markdown("### 📑 Tabla de Estadísticos Personalizada")
        st.markdown("Configura exactamente qué métricas deseas calcular, al estilo de software estadístico profesional.")
        
        # --- CONFIGURACIÓN AVANZADA ---
        with st.expander("⚙️ Configuración de Variables y Métricas", expanded=True):
            # 1. Selección de Variables y Segmentación
            c_sel_1, c_sel_2 = st.columns(2)
            with c_sel_1:
                vars_inteligentes = st.multiselect(
                    "Variables a Analizar:", 
                    options=selected_vars, 
                    default=selected_vars[:5] if len(selected_vars) > 5 else selected_vars
                )
            with c_sel_2:
                cat_opts = ["(General - Sin Segmentar)"] + df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                segmento = st.selectbox("🔀 Segmentar Resultados por:", options=cat_opts)
            
            st.divider()
            st.markdown("**Seleccione los estadísticos a calcular:**")
            
            # 2. Panel de Métricas (4 Columnas)
            col_tend, col_disp, col_perc, col_form = st.columns(4)
            
            with col_tend:
                st.markdown("###### 🎯 Tendencia Central")
                check_n = st.checkbox("N (Conteo)", value=True)
                check_media = st.checkbox("Media", value=True)
                check_mediana = st.checkbox("Mediana", value=True)
                check_moda = st.checkbox("Moda")
                check_suma = st.checkbox("Suma")
                check_gmean = st.checkbox("Media Geométrica")
                check_ic = st.checkbox("IC 95% (Media)", value=True)
            
            with col_disp:
                st.markdown("###### 📏 Dispersión")
                check_std = st.checkbox("Desviación Típica", value=True)
                check_var = st.checkbox("Varianza")
                check_cv = st.checkbox("Coef. Variación (%)", value=True)
                check_min = st.checkbox("Mínimo", value=True)
                check_max = st.checkbox("Máximo", value=True)
                check_rango = st.checkbox("Recorrido (Rango)")
                check_iqr = st.checkbox("Rango Intercuartílico", value=True)
                check_se = st.checkbox("Error Est. Media")
            
            with col_perc:
                st.markdown("###### 📍 Percentiles")
                check_cuartiles = st.checkbox("Cuartiles (25, 50, 75)")
                check_deciles = st.checkbox("Deciles (10, 20...90)")
                custom_perc_str = st.text_input("Personalizados (ej: 5, 95, 99)", help="Separa los valores por comas")
            
            with col_form:
                st.markdown("###### 🔔 Forma y Dist.")
                check_asimetria = st.checkbox("Asimetría (Skewness)")
                check_curtosis = st.checkbox("Curtosis")
                check_norm = st.checkbox("Prueba Normalidad (p)", value=True)
        
        # --- FUNCIÓN HELPER LOCAL (Con alias seguro 'ss') ---
        import scipy.stats as ss  # IMPORTACIÓN SEGURA AQUÍ
        
        def calcular_metricas_fila(serie_datos):
            """Calcula solo lo seleccionado para una serie de datos"""
            res = {}
            datos = serie_datos.dropna()
            n = len(datos)
            
            if n == 0:
                return {}
            
            # Tendencia Central
            if check_n:
                res['N'] = n
            if check_media:
                res['Media'] = np.mean(datos)
            if check_mediana:
                res['Mediana'] = np.median(datos)
            if check_moda:
                try:
                    moda_res = ss.mode(datos, keepdims=True)  # Usamos ss
                    res['Moda'] = moda_res.mode[0]
                except:
                    res['Moda'] = np.nan
            if check_suma:
                res['Suma'] = np.sum(datos)
            if check_gmean:
                try:
                    if (datos <= 0).any():
                        res['M. Geom.'] = np.nan
                    else:
                        res['M. Geom.'] = ss.gmean(datos)  # Usamos ss
                except:
                    res['M. Geom.'] = np.nan
            if check_ic:
                try:
                    se = ss.sem(datos)  # Usamos ss (Aquí fallaba antes)
                    h = se * ss.t.ppf((1 + 0.95) / 2., n-1)
                    m = np.mean(datos)
                    res['IC 95%'] = f"[{m-h:.2f} - {m+h:.2f}]"
                except:
                    res['IC 95%'] = "-"
            
            # Dispersión
            if check_std:
                res['D.E.'] = np.std(datos, ddof=1)
            if check_var:
                res['Varianza'] = np.var(datos, ddof=1)
            if check_cv:
                mu = np.mean(datos)
                res['CV %'] = (np.std(datos, ddof=1) / mu * 100) if mu != 0 else 0
            if check_min:
                res['Mín'] = np.min(datos)
            if check_max:
                res['Máx'] = np.max(datos)
            if check_rango:
                res['Rango'] = np.max(datos) - np.min(datos)
            if check_iqr:
                res['IQR'] = np.percentile(datos, 75) - np.percentile(datos, 25)
            if check_se:
                res['E.E.M.'] = ss.sem(datos)  # Usamos ss
            
            # Forma
            if check_asimetria:
                res['Asimetría'] = ss.skew(datos)  # Usamos ss
            if check_curtosis:
                res['Curtosis'] = ss.kurtosis(datos)  # Usamos ss
            if check_norm:
                try:
                    if n < 3:
                        res['P-Normalidad'] = np.nan
                    elif n < 50:
                        _, p = ss.shapiro(datos)  # Usamos ss
                        res['P-Normalidad'] = p
                    else:
                        # KS contra normal estandarizada
                        _, p = ss.kstest((datos - np.mean(datos))/np.std(datos, ddof=1), 'norm')
                        res['P-Normalidad'] = p
                except:
                    res['P-Normalidad'] = np.nan
            
            # Percentiles
            if check_cuartiles:
                res['P25'] = np.percentile(datos, 25)
                res['P50'] = np.percentile(datos, 50)
                res['P75'] = np.percentile(datos, 75)
            
            if check_deciles:
                for d in range(10, 100, 10):
                    res[f'P{d}'] = np.percentile(datos, d)
            
            if custom_perc_str:
                try:
                    vals = [float(x.strip()) for x in custom_perc_str.split(',') if x.strip()]
                    for v in vals:
                        if 0 <= v <= 100:
                            res[f'P{int(v)}'] = np.percentile(datos, v)
                except:
                    pass
            
            return res
        
        # --- MOTOR DE RENDERIZADO AISLADO (IFRAME) ---
        def renderizar_tabla_tesis_aislada(df_plano):
            """
            Genera un documento HTML independiente con el CSS del usuario.
            Al usarse en un IFrame, Streamlit NO puede sobrescribir los estilos.
            """
            df_indexed = df_plano.set_index('Variable')
            # Re-estructurar MultiIndex
            grupos_map = {
                'Tendencia Central': ['N', 'Media', 'Mediana', 'Moda', 'Suma', 'M. Geom.', 'IC 95%'],
                'Dispersión': ['D.E.', 'Varianza', 'CV %', 'Mín', 'Máx', 'Rango', 'IQR', 'E.E.M.'],
                'Forma y Dist.': ['Asimetría', 'Curtosis', 'P-Normalidad'],
                'Percentiles': [c for c in df_indexed.columns if c.startswith('P') and c[1:].isdigit()]
            }
            new_cols = []
            for col in df_indexed.columns:
                grupo = next((g for g, cs in grupos_map.items() if col in cs), 'Otros')
                new_cols.append((grupo, col))
            df_indexed.columns = pd.MultiIndex.from_tuples(new_cols)
            # --- HTML + CSS EXACTO DEL USUARIO ---
            html_content = f"""
            <html>
            <head>
            <style>
                body {{ background-color: transparent; margin: 0; padding: 0; }}
                .tabla-contenedor {{
                    overflow-x: auto;
                    width: 100%;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 1.5rem;
                    font-size: 0.95rem;
                    background-color: white;
                }}
                th, td {{
                    padding: 0.75rem;
                    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                    text-align: center;
                    border-right: 1px solid rgba(0, 0, 0, 0.05);
                }}
                th {{
                    font-weight: 600;
                    color: #002676; /* AZUL ACADÉMICO */
                    background-color: #f9f9f9;
                    border-bottom: 2px solid rgba(0, 38, 118, 0.2);
                }}
                tr:first-child th {{
                    background-color: #eef2f8;
                    text-transform: uppercase;
                    font-size: 0.8rem;
                    letter-spacing: 1px;
                }}
                tbody th {{
                    text-align: left;
                    background-color: #fcfcfc;
                    color: #333;
                    font-weight: 600;
                }}
                tr:hover {{
                    background-color: rgba(0, 38, 118, 0.02);
                }}
            </style>
            </head>
            <body>
                <div class="tabla-contenedor">
                    {df_indexed.to_html(float_format="%.2f", border=0)}
                </div>
            </body>
            </html>
            """
            return html_content

        # --- LÓGICA DE VISUALIZACIÓN EN LA PESTAÑA ---
        if vars_inteligentes:
            st.divider()
            import scipy.stats as ss
            
            col_v, _ = st.columns([1, 3])
            vista_p = col_v.toggle("📰 Vista Académica (Tesis)", value=True)
            
            # Caso General
            if segmento == "(General - Sin Segmentar)":
                filas = []
                for v in vars_inteligentes:
                    m = calcular_metricas_fila(df[v])
                    m['Variable'] = v
                    filas.append(m)
                df_res = pd.DataFrame(filas)
                
                if not df_res.empty:
                    st.markdown(f"##### 📊 Resultados Globales (N={len(df)})")
                    if vista_p:
                        import streamlit.components.v1 as components
                        html_completo = renderizar_tabla_tesis_aislada(df_res)
                        # Ajustamos la altura dinámicamente según el número de variables (aprox 50px por fila + cabecera)
                        altura_tabla = (len(df_res) * 55) + 150 
                        components.html(html_completo, height=min(altura_tabla, 800), scrolling=True)
                    else:
                        st.dataframe(df_res, use_container_width=True)
                    boton_guardar_tabla(df_res, "Descriptiva_Global", "btn_dg")
            
            # Caso Segmentado
            else:
                grupos = sorted(df[segmento].dropna().unique())
                st.info(f"📂 Segmentado por: **{segmento}**")
                tbs = st.tabs([f"{g}" for g in grupos])
                for i, g in enumerate(grupos):
                    with tbs[i]:
                        df_sub = df[df[segmento] == g]
                        filas_g = []
                        for v in vars_inteligentes:
                            m = calcular_metricas_fila(df_sub[v])
                            m['Variable'] = v
                            filas_g.append(m)
                        df_res_g = pd.DataFrame(filas_g)
                        if not df_res_g.empty:
                            if vista_p:
                                import streamlit.components.v1 as components
                                html_completo = renderizar_tabla_tesis_aislada(df_res_g)
                                # Ajustamos la altura dinámicamente según el número de variables (aprox 50px por fila + cabecera)
                                altura_tabla = (len(df_res_g) * 55) + 150 
                                components.html(html_completo, height=min(altura_tabla, 800), scrolling=True)
                            else:
                                st.dataframe(df_res_g, use_container_width=True)
                            boton_guardar_tabla(df_res_g, f"Desc_{g}", f"btn_{i}")
        else:
            st.info("👈 Selecciona variables numéricas arriba para comenzar.")
