
"""
Módulo de Renderizado: Pruebas de Hipótesis (Inferencia)
--------------------------------------------------------
Interfaz unificada para pruebas de hipótesis paramétricas y no paramétricas,
comparaciones de grupos y análisis de correlación/riesgo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

# Importar lógica estadística desde backend
from modules.stats.inference import (
    # Paramétricas Indep
    ttest_independiente, anova_oneway, 
    # Paramétricas Pareadas
    ttest_paired, 
    # No Paramétricas Indep
    mann_whitney_u, kruskal_wallis, 
    # No Paramétricas Pareadas
    wilcoxon_paired, friedman_test,
    # Cualitativas
    chi_cuadrado, fisher_exact, mcnemar_test, cochran_q_test,
    # Correlaciones
    pearson_correlation, spearman_correlation, kendall_tau, odds_ratio_risk,
    # Supuestos
    shapiro_wilk_test, levene_test,
    # Herramientas Tier 1
    calculate_mean_ci, calculate_power_analysis, pairwise_welch_games_howell, get_hypothesis_text
)
from modules.stats.validators import validate_data_for_analysis

def plot_qq_normality(data: pd.Series, title: str):
    """Genera un gráfico Q-Q para inspección visual de normalidad."""
    q_theoretical, q_sample = stats.probplot(data, dist="norm", plot=None)[0]
    
    fig = px.scatter(x=q_theoretical, y=q_sample, labels={'x': 'Cuantiles Teóricos', 'y': 'Cuantiles Muestra'}, title=title)
    # Agregar línea de referencia (45 grados aprox, ajustada por regresión interna de probplot o extremos)
    # Mejor: Línea ajustada por mínimos cuadrados de los puntos
    slope, intercept, r, p, err = stats.linregress(q_theoretical, q_sample)
    
    # Puntos para la línea
    line_x = np.array([min(q_theoretical), max(q_theoretical)])
    line_y = slope * line_x + intercept
    
    fig.add_shape(type="line", x0=line_x[0], y0=line_y[0], x1=line_x[1], y1=line_y[1], line=dict(color="red", dash="dash"))
    return fig

def render_inferencia(df: pd.DataFrame = None, 
                     var1: str = None, 
                     var2: str = None, 
                     test_type: str = 'auto'):
    """
    Panel de pruebas de hipótesis con pestañas organizadas.
    """
    st.header("🧪 Inferencia Estadística")

    if df is None:
        if 'df_principal' in st.session_state:
            df = st.session_state.df_principal
        else:
            st.error("⚠️ No hay datos disponibles.")
            return

    valid, msg = validate_data_for_analysis(df)
    if not valid: return

    # --- CONFIGURACIÓN GLOBAL ---
    show_thesis_mode = st.toggle("🎓 Activar Modo Tesis (Avanzado)", value=False, help="Muestra opciones avanzadas como Alpha configurable, Hipótesis Nula/Alterna y Potencia.")
    st.divider()

    alpha = 0.05
    if show_thesis_mode:
        with st.expander("⚙️ Configuración Global (Nivel de Significancia)", expanded=True):
            alpha = st.slider("Nivel de Significancia (Alpha):", 0.01, 0.10, 0.05, 0.01, help="Define el umbral para rechazar la Hipótesis Nula.")

    # Estructura de Pestañas
    tab_cuant, tab_cuali, tab_corr = st.tabs([
        "📊 Comparación Cuantitativa", 
        "🧬 Comparación Cualitativa", 
        "🔗 Correlaciones & Riesgo"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: COMPARACIÓN CUANTITATIVA (Medias/Medianas)
    # --------------------------------------------------------------------------
    with tab_cuant:
        st.markdown("#### Comparación de Grupos (Variables Numéricas)")
        
        # 0. Inicialización
        group_var = None
        second_var = None
        is_paired = False
        n_groups = 0
        data_groups = []
        group_labels = []

        # 1. Configuración de Variables (Inputs)
        col_c1, col_c2 = st.columns(2)
        
        numericas = df.select_dtypes(include=['number']).columns.tolist()
        todas_cols = df.columns.tolist()
        
        with col_c1:
            var_interes = st.selectbox("Variable Numérica (Dependiente):", numericas, key='inf_q_var')
            
        with col_c2:
            modo_comparacion = st.radio("Modo:", ["Por Grupos (ej. Hombres vs Mujeres)", "Entre Variables (ej. Pre vs Post)"], horizontal=True)
            
        # Lógica de Inputs Específicos
        if modo_comparacion.startswith("Por Grupos"):
            group_var = st.selectbox("Variable de Agrupación (Independiente):", [c for c in todas_cols if c != var_interes], key='inf_q_group')
            is_paired = False 
        else:
            second_var = st.selectbox("Segunda Variable Numérica (Comparación):", [c for c in numericas if c != var_interes], key='inf_q_var2')
            is_paired = st.checkbox("¿Son muestras pareadas / medidas repetidas?", value=True, help="Ej: Mismo paciente medido dos veces.")

        # 2. Preparación de Datos (Lógica después de Inputs)
        if modo_comparacion.startswith("Por Grupos"):
            if group_var:
                unique_groups = df[group_var].dropna().unique()
                n_groups = len(unique_groups)
                
                # Check de seguridad
                if n_groups < 2:
                    st.warning("⚠️ La variable de agrupación debe tener al menos 2 niveles para comparar.")
                elif n_groups > 20:
                     st.warning(f"⚠️ La variable '{group_var}' tiene {n_groups} niveles únicos. ¿Seguro?")
                
                if n_groups >= 2:
                    for g in unique_groups:
                        data_groups.append(df[df[group_var] == g][var_interes].dropna())
                        group_labels.append(str(g))
        else:
            if second_var:
                n_groups = 2
                group_labels = [var_interes, second_var]
                # Para efectos de conteo UI:
                data_groups = [df[var_interes], df[second_var]] 

        # 3. Configuración del Método (Auto/Manual)
        st.markdown("##### ⚙️ Configuración del Análisis")
        mode_test = st.radio("Método de Selección:", ["Automática (Inteligente)", "Manual"], horizontal=True, key='mode_test_q')
        
        selected_manual_test = None
        
        if mode_test == "Manual":
            options = []
            st.caption(f"ℹ️ Se han detectado **{n_groups}** grupos/series para comparar.")
            
            if n_groups == 2:
                if is_paired: options = ["T-Student Pareada", "Wilcoxon Signed-Rank"]
                else: options = ["T-Student Independiente", "U de Mann-Whitney"]
            elif n_groups > 2:
                if is_paired: options = ["Friedman Test"] 
                else: options = ["ANOVA One-Way", "Kruskal-Wallis"]
            else:
                st.warning("No hay suficientes grupos definidos para mostrar opciones.")
            
            if options:
                selected_manual_test = st.selectbox("Seleccione la prueba:", options)
                
            # Mostrar Hipótesis Manual
            if selected_manual_test and show_thesis_mode:
                h_text = get_hypothesis_text(selected_manual_test, var_interes, group_var if group_var else "Grupos")
                st.info(f"**H₀:** {h_text['H0']}\n\n**H₁:** {h_text['H1']}")
                
        # --- CONFIGURACIÓN AVANZADA ---
        if show_thesis_mode:
            with st.expander("⚙️ Verificación de Potencia (A Priori)"):
                st.caption(f"Cálculo basado en Alpha Global: **{alpha}**")
                target_power = st.number_input("Tamaño del Efecto Esperado (d):", 0.2, 2.0, 0.5, 0.1)
                est_n = len(df) # Aprox
                current_pow = calculate_power_analysis(target_power, int(est_n/n_groups) if n_groups else est_n, alpha)
                st.metric("Potencia Estimada (1-Beta)", f"{current_pow:.2f}")

        # Caja Educativa
        with st.expander("ℹ️ Ayuda para elegir la prueba", expanded=False):
            st.markdown("""
            **Guía Rápida:**
            - **T-Student / ANOVA**: Para datos con distribución Normal (Paramétricos).
            - **Mann-Whitney / Kruskal**: Para datos No Normales (No Paramétricos).
            - **Paired T / Wilcoxon**: Si son los mismos sujetos medidos dos veces.
            - **Friedman**: Si son 3+ mediciones en los mismos sujetos.
            """)
            
        # 4. Botón de Acción
        if st.button("Calcular Diferencia", type="primary", key='btn_calc_cuant'):
            st.divider()
            
            # --- VALIDACIÓN FINAL Y EXTRACCIÓN REAL DE DATOS ---
            real_data_groups = []
            
            if modo_comparacion.startswith("Por Grupos"):
                if n_groups < 2: 
                    st.error("No se puede calcular: Menos de 2 grupos.")
                    st.stop()
                real_data_groups = data_groups 
                st.write(f"Comparando **{var_interes}** entre grupos de **{group_var}**: {', '.join(group_labels)}")
            else:
                if not second_var:
                    st.error("Falta segunda variable.")
                    st.stop()
                
                # Extracción real al momento de ejecutar (para manejar pareado vs indep si hiciera falta lógica extra)
                # Backend ttest_paired ya hace inner join. Backend indep hace dropna.
                real_data_groups = [
                    df[var_interes].dropna(),
                    df[second_var].dropna()
                ]
                st.write(f"Comparando variables: **{var_interes}** vs **{second_var}**")

            # --- VERIFICACIÓN DE SUPUESTOS ---
            st.markdown("##### 🔍 Verificación de Supuestos")
            
            # Normalidad (Shapiro)
            normality_data = [] 
            all_normal = True
            
            for i, d in enumerate(real_data_groups):
                n_curr = len(d)
                # Calcular IC de la Media
                ci_low, ci_high, mean_val = calculate_mean_ci(d, alpha)
                
                if n_curr < 3:
                     normality_data.append({
                         "Grupo": group_labels[i], "N": n_curr, "Media": f"{mean_val:.2f}", "IC 95%": "N/A",
                         "Shapiro p": "-", "¿Normal?": "Error (N<3)"
                     })
                     all_normal = False 
                else:
                    shapiro = shapiro_wilk_test(d)
                    is_norm = shapiro.get('is_normal', False) and shapiro.get('p_value', 0) > alpha # Usar alpha usuario
                    if not is_norm: all_normal = False
                    
                    normality_data.append({
                        "Grupo": group_labels[i], 
                        "N": n_curr, 
                        "Media": f"{mean_val:.2f}",
                        "IC 95%": f"[{ci_low:.2f} - {ci_high:.2f}]",
                        "Shapiro p": f"{shapiro['p_value']:.4f}", 
                        "¿Normal?": "✅ Sí" if is_norm else "❌ No"
                    })
            
            col_sup1, col_sup2 = st.columns(2)
            with col_sup1:
                st.caption(f"Descriptivos y Normalidad (Alpha={alpha})")
                st.dataframe(pd.DataFrame(normality_data), hide_index=True, use_container_width=True)
                
                # --- GRÁFICOS Q-Q (THESIS MODE) ---
                if show_thesis_mode and len(real_data_groups) <= 6: # Mostrar hasta 6 grupos
                    st.markdown("**Verificación Visual (Q-Q Plots):**")
                    cols_qq = st.columns(len(real_data_groups))
                    for idx, grp_data in enumerate(real_data_groups):
                         with cols_qq[idx]:
                             try:
                                 fig_qq = plot_qq_normality(grp_data, f"Q-Q: {group_labels[idx]}")
                                 st.plotly_chart(fig_qq, use_container_width=True)
                             except: pass

            # Homocedasticidad (Levene)
            all_homo = True
            if not is_paired and len(real_data_groups) >= 2:
                with col_sup2:
                    st.caption("Homocedasticidad (Levene)")
                    try:
                        clean_arrays = [x.values for x in real_data_groups if len(x) > 0]
                        levene = levene_test(clean_arrays)
                        if "error" not in levene:
                            p_levene = levene['p_value']
                            is_homo = p_levene > alpha # Usar alpha usuario
                            icon = "✅ Varianzas Iguales" if is_homo else "⚠️ Varianzas Distintas"
                            st.info(f"{icon}\n\n**p-value:** {p_levene:.4f}")
                            all_homo = is_homo
                    except:
                        st.caption("No se pudo calcular Levene.")

            # --- EJECUCIÓN DEL TEST ---
            res = {}
            test_name_final = ""
            
            if mode_test == "Manual" and selected_manual_test:
                test_name_final = selected_manual_test
                try:
                    if selected_manual_test == "T-Student Pareada": res = ttest_paired(real_data_groups[0], real_data_groups[1])
                    elif selected_manual_test == "Wilcoxon Signed-Rank": res = wilcoxon_paired(real_data_groups[0], real_data_groups[1])
                    elif selected_manual_test == "T-Student Independiente": res = ttest_independiente(real_data_groups[0], real_data_groups[1])
                    elif selected_manual_test == "U de Mann-Whitney": res = mann_whitney_u(real_data_groups[0], real_data_groups[1])
                    elif selected_manual_test == "Friedman Test": res = friedman_test(*real_data_groups)
                    elif selected_manual_test == "ANOVA One-Way": res = anova_oneway(real_data_groups)
                    elif selected_manual_test == "Kruskal-Wallis": res = kruskal_wallis(*real_data_groups)
                except Exception as e:
                    res = {"error": f"Error manual: {str(e)}"}
            else:
                # Automático
                reason = "Selección Automática Basada en Supuestos."
                if n_groups == 2:
                    if is_paired:
                         if all_normal: 
                             res = ttest_paired(real_data_groups[0], real_data_groups[1])
                             test_name_final = "T-Student Pareada"
                             reason = "Se eligió **T-Student Pareada** (Normalidad Cumplida)."
                         else: 
                             res = wilcoxon_paired(real_data_groups[0], real_data_groups[1])
                             test_name_final = "Wilcoxon Signed-Rank"
                             reason = "Se eligió **Wilcoxon** (No Normal)."
                    else:
                        if all_normal: 
                            res = ttest_independiente(real_data_groups[0], real_data_groups[1])
                            test_name_final = "T-Student Independiente"
                            reason = f"Se eligió **T-Student** ({'Var.Homogéneas' if all_homo else 'Welch'})."
                        else: 
                            res = mann_whitney_u(real_data_groups[0], real_data_groups[1])
                            test_name_final = "U de Mann-Whitney"
                            reason = "Se eligió **Mann-Whitney** (No Normal)."
                else: 
                    # 3+ Grupos
                     if is_paired:
                         res = friedman_test(*real_data_groups)
                         test_name_final = "Friedman Test"
                         reason = "Se eligió **Friedman** (3+ Muestras Relacionadas)."
                     else:
                        if all_normal and all_homo: 
                            res = anova_oneway(real_data_groups)
                            test_name_final = "ANOVA One-Way"
                            reason = "Se eligió **ANOVA** (Supuestos Cumplidos)."
                        else: 
                            res = kruskal_wallis(*real_data_groups)
                            test_name_final = "Kruskal-Wallis"
                            reason = "Se eligió **Kruskal-Wallis** (No Paramétrico)."
                
                if "error" not in res:
                    st.info(f"🤖 **Auto:** {reason}")

            # --- MOSTRAR RESULTADOS ---
            if "error" in res:
                st.error(f"❌ Error en el cálculo: {res['error']}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Estadístico", f"{res.get('statistic',0):.3f}")
                
                pval = res.get('p_value', 1.0)
                sig_label = "Significativo (p<0.05)" if pval < 0.05 else "No Significativo"
                sig_color = "inverse" if pval < 0.05 else "normal"
                c2.metric("Valor P", f"{pval:.4f}", delta=sig_label, delta_color=sig_color)
                
                eff_name = res.get('effect_size_name', 'Tamaño Efecto')
                eff_val = res.get('effect_size')
                if eff_val is not None:
                    c3.metric(eff_name, f"{eff_val:.3f}")
                else:
                    c3.metric("Tamaño Efecto", "N/A")
                    
                st.success(f"📌 **Conclusión:** {res.get('interpretation', '')}")
                
                if 'effect_interpretation' in res:
                    st.caption(f"Interpretación del Efecto: {res['effect_interpretation']}")
                    
                # --- REPORTE TESIS ---
                with st.expander("📝 Interpretación para Tesis (Generado Automáticamente)", expanded=True):
                    test_clean = res.get('test_name', 'Prueba Estadística')
                    conclusion_thesis = "existe una diferencia estadísticamente significativa" if pval < alpha else "no se encontró evidencia suficiente para rechazar la hipótesis nula de igualdad"
                    
                    # Generar String APA
                    apa_str = ""
                    try:
                        stat_val = res.get('statistic', 0)
                        # Intentar inferir DF o usar datos genéricos
                        # T-Test
                        if "Student" in test_clean or "Welch" in test_clean:
                            # DF aprox = N1 + N2 - 2 (indep) o N - 1 (paired). Backend a veces lo retorna, sino aprox.
                            if 'dof' in res:
                                 apa_str = f"t({res['dof']:.1f}) = {stat_val:.2f}, p = {pval:.3f}"
                            else:
                                 # Fallback genérico si no hay DOF
                                 apa_str = f"t = {stat_val:.2f}, p = {pval:.3f}"
                            if eff_val is not None: apa_str += f", d = {eff_val:.2f}"
                                
                        # Mann-Whitney / Wilcoxon
                        elif "Mann" in test_clean or "Wilcoxon" in test_clean:
                            u_val = stat_val
                            apa_str = f"U = {u_val:.1f}, p = {pval:.3f}"
                            
                        # ANOVA / Kruskal
                        elif "ANOVA" in test_clean:
                            # F suele requerir 2 DF (between, within), aquí simplificamos si no están
                            apa_str = f"F = {stat_val:.2f}, p = {pval:.3f}"
                            if eff_val is not None: apa_str += f", η² = {eff_val:.2f}"
                        elif "Kruskal" in test_clean:
                             apa_str = f"H = {stat_val:.2f}, p = {pval:.3f}"
                        else:
                             apa_str = f"Estadístico = {stat_val:.2f}, p = {pval:.3f}"
                             
                    except:
                        apa_str = "No se pudo generar formato APA exacto."

                    st.markdown(f"**Reporte APA (Sugerido):**\n> {apa_str}")

                    reporte = (
                        f"Para analizar la relación entre las variables, se aplicó la prueba estadística **{test_clean}**. "
                        f"El análisis arrojó un valor estadístico de {res.get('statistic',0):.3f} y un valor p de {pval:.4f}. \n\n"
                        f"Considerando un nivel de significancia de {alpha}, se concluye que **{conclusion_thesis}** entre los grupos analizados. "
                    )
                    if eff_val is not None:
                        reporte += f"Adicionalmente, el tamaño del efecto ({eff_name}) fue de {eff_val:.3f}, considerado {res.get('effect_interpretation', 'desconocido')}."
                        
                    st.info(reporte)
                    
                # --- GRÁFICO ---
                st.write("#### Visualización")
                try:
                    df_viz = pd.DataFrame()
                    # Reconstruir un DF tidy para plotting
                    if group_var:
                       df_viz = df[[group_var, var_interes]].dropna()
                       fig = px.box(df_viz, x=group_var, y=var_interes, points="all", 
                                    title=f"Distribución de {var_interes} por {group_var}",
                                    color=group_var)
                    else:
                       # Caso 2 Variables (Paired o Indep)
                       # Transformar a formato largo: [Valor, Variable]
                       # Asegurar misma longitud para visualización concatenada
                       # Pero data_groups ya tiene series limpias indep
                       # Para visualización mejor usar el original o crear DataFrames con label
                       d1 = pd.DataFrame({ 'Valor': data_groups[0].values, 'Variable': group_labels[0] })
                       d2 = pd.DataFrame({ 'Valor': data_groups[1].values, 'Variable': group_labels[1] })
                       df_viz = pd.concat([d1, d2])
                       
                       fig = px.box(df_viz, x='Variable', y='Valor', points="all",
                                    title=f"Comparación: {group_labels[0]} vs {group_labels[1]}",
                                    color='Variable')
                                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"No se pudo generar el gráfico: {e}")


    # --------------------------------------------------------------------------
    # TAB 2: COMPARACIÓN CUALITATIVA (Proporciones)
    # --------------------------------------------------------------------------
    with tab_cuali:
        st.markdown("#### Comparación de Variables Categóricas (Proporciones)")
        
        # 1. Configuración de Variables
        c1, c2 = st.columns(2)
        cats = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        with c1:
            cat_1 = st.selectbox("Variable 1 (Filas):", cats, key='inf_c_v1')
        with c2:
            cat_2 = st.selectbox("Variable 2 (Columnas):", [c for c in cats if c != cat_1], key='inf_c_v2')
            
        is_paired_cat = st.checkbox("¿Son muestras pareadas? (McNemar)", help="Ej: Diagnóstico Antes vs Diagnóstico Después (mismos sujetos).")

        # 2. Preparación de Datos (Cálculo Previo)
        s1 = pd.Series(dtype='object')
        s2 = pd.Series(dtype='object')
        crosstab_preview = pd.DataFrame()
        ct_shape = (0,0)
        has_low_counts = False
        
        if cat_1 and cat_2:
            s1 = df[cat_1].dropna()
            s2 = df[cat_2].dropna()
            
            # Alinear para pareado
            if is_paired_cat:
                df_temp = df[[cat_1, cat_2]].dropna()
                s1, s2 = df_temp[cat_1], df_temp[cat_2]
            
            # Verificar Counts para Fisher/Chi
            try:
                crosstab_preview = pd.crosstab(s1, s2)
                ct_shape = crosstab_preview.shape
                # Verificar celdas < 5
                has_low_counts = (crosstab_preview < 5).any().any()
            except Exception:
                pass

        # 3. Configuración del Método
        st.markdown("##### ⚙️ Configuración del Análisis")
        mode_test_c = st.radio("Método de Selección:", ["Automática (Inteligente)", "Manual"], horizontal=True, key='mode_test_c')
        
        selected_manual_test_c = None
        
        if mode_test_c == "Manual":
            options_c = []
            if ct_shape == (2,2):
                if is_paired_cat: options_c = ["McNemar Test"]
                else: options_c = ["Chi-Cuadrado Pearson", "Fisher Exact Test"]
            else:
                # Tablas mayores a 2x2
                if is_paired_cat: options_c = ["Cochran Q Test"] # Si hubiera >2 vars pareadas, pero aquí solo hay 2 inputs... McNemar
                else: options_c = ["Chi-Cuadrado Pearson"]
            
            # Ajuste manual si el usuario se empeña
            if is_paired_cat and "McNemar Test" not in options_c: options_c.append("McNemar Test")

            selected_manual_test_c = st.selectbox("Seleccione la prueba:", options_c)
            
            if selected_manual_test_c and show_thesis_mode:
                h_text_c = get_hypothesis_text(selected_manual_test_c, cat_1, cat_2)
                st.info(f"**H₀:** {h_text_c['H0']}\n\n**H₁:** {h_text_c['H1']}")
        
        # Caja Educativa
        with st.expander("ℹ️ Ayuda para elegir la prueba", expanded=False):
            st.markdown("""
            - **Chi-Cuadrado**: Para ver si dos variables categóricas son independientes.
            - **Fisher Exact**: Igual que Chi2 pero para muestras pequeñas (celdas < 5) en tablas 2x2.
            - **McNemar**: Para cambios en variables binarias en los mismos sujetos (Antes vs Después).
            """)
            
        # 4. Botón de Acción
        if st.button("Calcular Asociación", type="primary", key='btn_calc_cuali'):
            st.divider()
            
            if not cat_1 or not cat_2:
                st.error("Seleccione dos variables.")
            else:
                res = {}
                test_name_used = ""
                
                # --- EJECUCIÓN DEL TEST ---
                if mode_test_c == "Manual" and selected_manual_test_c:
                    # Manual Logic
                    test_name_used = selected_manual_test_c
                    try:
                        if selected_manual_test_c == "McNemar Test":
                            res = mcnemar_test(s1, s2)
                        elif selected_manual_test_c == "Fisher Exact Test":
                            if ct_shape != (2,2):
                                res = {"error": f"Fisher Exact requiere tabla 2x2. Tu tabla es {ct_shape}."}
                            else:
                                res = fisher_exact(crosstab_preview)
                        elif selected_manual_test_c == "Chi-Cuadrado Pearson":
                            res = chi_cuadrado(s1, s2)
                        elif selected_manual_test_c == "Cochran Q Test":
                             res = {"error": "Cochran Q requiere >2 grupos pareados. Usa McNemar para 2."}
                    except Exception as e:
                         res = {"error": f"Error manual: {str(e)}"}
                         
                else:
                    # Auto Logic
                    if is_paired_cat:
                        res = mcnemar_test(s1, s2)
                        st.info("🤖 **Auto:** Se eligió **McNemar** por ser muestras pareadas.")
                        test_name_used = "McNemar (Pareado)"
                    else:
                        if ct_shape == (2,2):
                            if has_low_counts:
                                res = fisher_exact(crosstab_preview)
                                st.info("🤖 **Auto:** Se eligió **Fisher Exact** (detectadas celdas con valores < 5).")
                                test_name_used = "Fisher Exact Test"
                            else:
                                res = chi_cuadrado(s1, s2)
                                st.info("🤖 **Auto:** Se eligió **Chi-Cuadrado** (Muestras grandes, celdas >= 5).")
                                test_name_used = "Chi-Cuadrado Pearson"
                        else:
                            res = chi_cuadrado(s1, s2)
                            reason_add = "(Fisher no disponible para >2x2)" if has_low_counts else ""
                            st.info(f"🤖 **Auto:** Se eligió **Chi-Cuadrado** para tabla {ct_shape}. {reason_add}")
                            test_name_used = "Chi-Cuadrado Pearson"

                # --- MOSTRAR RESULTADOS ---
                if "error" in res:
                    st.error(f"❌ Error: {res['error']}")
                else:
                    col_res1, col_res2 = st.columns(2)
                    col_res1.metric("Estadístico", f"{res.get('statistic', 0):.3f}")
                    
                    pval = res.get('p_value', 1.0)
                    sig_label = "Significativo (p<0.05)" if pval < 0.05 else "No Significativo"
                    col_res2.metric("Valor P", f"{pval:.4f}", delta=sig_label, delta_color="inverse" if pval < 0.05 else "normal")
                    
                    eff = res.get('effect_size_name')
                    if eff:
                        st.metric(eff, f"{res.get('effect_size', 0):.3f}")
                        
                    st.success(f"📌 **Conclusión:** {res.get('interpretation', '')}")
                    
                    # --- REPORTE TESIS ---
                    with st.expander("📝 Interpretación para Tesis (Generado Automáticamente)", expanded=True):
                         conclusion_txt = "existe una asociación estadísticamente significativa" if pval < alpha else "no se encontró evidencia suficiente de asociación"
                         
                         # Generar APA
                         apa_str_c = ""
                         try:
                             chi_val = res.get('statistic', 0)
                             n_total = len(df) # aprox
                             apa_str_c = f"χ²(N={n_total}) = {chi_val:.2f}, p = {pval:.3f}"
                             if res.get('odds_ratio'):
                                 apa_str_c += f", OR = {res['odds_ratio']:.2f}"
                         except:
                             apa_str_c = "Formato APA no disponible."
                             
                         st.markdown(f"**Reporte APA (Sugerido):**\n> {apa_str_c}")

                         reporte = (
                             f"Para analizar la relación entre **{cat_1}** y **{cat_2}**, se aplicó la prueba de **{test_name_used}**. "
                             f"Los resultados mostraron un p-valor de {pval:.4f}. \n\n"
                             f"Considerando un nivel de significancia de {alpha}, se concluye que **{conclusion_txt}** entre las variables estudiadas."
                         )
                         # Agregar Odds Ratio si existe (Fisher)
                         if 'odds_ratio' in res:
                             reporte += f" El Odds Ratio calculado fue de {res['odds_ratio']:.2f}."
                         
                         st.info(reporte)

                    # Mostrar tabla contingencia visual (Stacked Bar)
                    st.write("#### Visualización de Frecuencias")
                    c_viz1, c_viz2 = st.columns([1, 1])
                    
                    with c_viz1:
                        st.caption("Tabla Cruzada")
                        st.dataframe(crosstab_preview)
                    
                    with c_viz2:
                        try:
                            # Plotly Stacked Bar
                            # Necesitamos DataFrame tidy count
                            df_counts = crosstab_preview.reset_index()
                            # Melt para plotly
                            df_long = df_counts.melt(id_vars=cat_1, var_name=cat_2, value_name='Frecuencia')
                            
                            fig = px.bar(df_long, x=cat_1, y='Frecuencia', color=cat_2, 
                                         title="Frecuencias Observadas", barmode='group')
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.caption(f"No graficable: {e}")

    # --------------------------------------------------------------------------
    # TAB 3: CORRELACIONES & RIESGO
    # --------------------------------------------------------------------------
    with tab_corr:
        st.markdown("#### Correlación y Riesgo")
        
        mode_corr = st.radio("Tipo de Análisis:", ["Correlación (Numérica)", "Riesgo (Odds Ratio)"], horizontal=True)
        
        if mode_corr == "Correlación (Numérica)":
            col_cr1, col_cr2 = st.columns(2)
            numericas = df.select_dtypes(include=['number']).columns.tolist()
            
            with col_cr1:
                var_x = st.selectbox("Variable X:", numericas, key='corr_x')
            with col_cr2:
                var_y = st.selectbox("Variable Y:", [n for n in numericas if n != var_x], key='corr_y')
                
            # --- 2. VERIFICACIÓN DE NORMALIDAD (PRE-CÁLCULO) ---
            st.markdown("##### 🔍 Verificación de Normalidad")
            
            norm_stats = []
            is_both_normal = True
            
            if var_x and var_y:
                 # Check X
                 sx = df[var_x].dropna()
                 nx = len(sx)
                 if nx > 3:
                     sh_x = shapiro_wilk_test(sx)
                     is_norm_x = sh_x.get('is_normal', False)
                     norm_stats.append({
                         "Variable": var_x, 
                         "Shapiro P-Value": f"{sh_x.get('p_value',0):.4f}", 
                         "Conclusión (p>0.05)": "✅ Normal" if is_norm_x else "⚠️ No Normal"
                     })
                     if not is_norm_x: is_both_normal = False
                 
                 # Check Y
                 sy = df[var_y].dropna()
                 ny = len(sy)
                 if ny > 3:
                     sh_y = shapiro_wilk_test(sy)
                     is_norm_y = sh_y.get('is_normal', False)
                     norm_stats.append({
                         "Variable": var_y, 
                         "Shapiro P-Value": f"{sh_y.get('p_value',0):.4f}", 
                         "Conclusión (p>0.05)": "✅ Normal" if is_norm_y else "⚠️ No Normal"
                     })
                     if not is_norm_y: is_both_normal = False

            if norm_stats:
                st.dataframe(pd.DataFrame(norm_stats), hide_index=True, use_container_width=True)

                if show_thesis_mode and st.toggle("Ver Gráficos Q-Q (Normalidad)", value=False):
                    cqq1, cqq2 = st.columns(2)
                    with cqq1: 
                        if var_x: st.plotly_chart(plot_qq_normality(df[var_x].dropna(), f"Q-Q: {var_x}"), use_container_width=True)
                    with cqq2:
                        if var_y: st.plotly_chart(plot_qq_normality(df[var_y].dropna(), f"Q-Q: {var_y}"), use_container_width=True)
                
                # Smart Recommendation
                if is_both_normal:
                    st.success("💡 Ambas variables siguen una distribución normal. Se recomienda: **Pearson**.")
                else:
                    st.warning("💡 Al menos una variable no es normal. Se recomienda: **Spearman** (más robusto).")
            
            # --- 3. CONFIGURACIÓN DEL MÉTODO ---
            st.markdown("##### ⚙️ Configuración del Análisis")
            mode_corr_settings = st.radio("Método de Selección:", ["Automática", "Manual"], horizontal=True, key='corr_mode_settings')
            
            selected_corr_method = None
            
            if mode_corr_settings == "Manual":
                selected_corr_method = st.selectbox("Prueba de Correlación:", ["Pearson (Lineal)", "Spearman (Rangos)", "Kendall (Muestras Pequeñas)"])
                
                # Advertencia inteligente
                if "Pearson" in selected_corr_method and not is_both_normal:
                    st.warning("⚠️ Precaución: Ha seleccionado Pearson pero sus datos no parecen seguir una distribución normal. Se sugiere Spearman.")
                
                if selected_corr_method and show_thesis_mode:
                    h_text_corr = get_hypothesis_text(selected_corr_method, var_x, var_y)
                    st.info(f"**H₀:** {h_text_corr['H0']}\n\n**H₁:** {h_text_corr['H1']}")
            
            st.info("ℹ️ Recuerde: **Correlación no implica causalidad**.")
            
            # --- 4. EJECUCIÓN ---
            if st.button("Calcular Correlación", type="primary"):
                st.divider()
                
                vx = df[var_x]
                vy = df[var_y]
                
                res = {}
                method_used = ""
                
                if mode_corr_settings == "Automática":
                    # Lógica Auto
                    if is_both_normal:
                        res = pearson_correlation(vx, vy)
                        method_used = "Pearson (Paramétrico)"
                        st.info("🤖 **Auto:** Se eligió **Pearson** porque ambas variables parecen normales.")
                    else:
                        # Si N es muy pequeño podría ser Kendall, pero por estandar usamos Spearman
                        res = spearman_correlation(vx, vy)
                        method_used = "Spearman (No Paramétrico)"
                        st.info("🤖 **Auto:** Se eligió **Spearman** por falta de normalidad en una o ambas variables.")
                else:
                    # Lógica Manual
                    method_used = selected_corr_method
                    if "Pearson" in selected_corr_method: res = pearson_correlation(vx, vy)
                    elif "Spearman" in selected_corr_method: res = spearman_correlation(vx, vy)
                    elif "Kendall" in selected_corr_method: res = kendall_tau(vx, vy)
                    
                if "error" in res:
                    st.error(res['error'])
                else:
                    c1, c2, c3 = st.columns(3)
                    r_val = res.get('statistic',0)
                    pval = res.get('p_value',1.0)
                    
                    c1.metric("Coeficiente (r)", f"{r_val:.3f}")
                    
                    sig_label = "Significativo (p<0.05)" if pval < 0.05 else "No Significativo"
                    c2.metric("Valor P", f"{pval:.4f}", delta=sig_label, delta_color="inverse" if pval < 0.05 else "normal")
                    
                    interpretation_short = res.get('effect_interpretation', '-')
                    c3.metric("Interpretación", interpretation_short)
                    
                    st.success(f"📌 **Conclusión:** {res.get('interpretation', '')}")
                    
                    # --- SENSITIVITY ANALYSIS (ATÍPICOS) ---
                    st.markdown("##### ⚖️ Análisis de Sensibilidad (Atípicos)")
                    
                    # Calcular Outliers IQR
                    def get_clean_mean(s):
                         q1 = s.quantile(0.25)
                         q3 = s.quantile(0.75)
                         iqr = q3 - q1
                         lower = q1 - 1.5 * iqr
                         upper = q3 + 1.5 * iqr
                         clean = s[(s >= lower) & (s <= upper)]
                         return clean.mean(), len(s) - len(clean)

                    raw_mean_x, outliers_x = vx.mean(), get_clean_mean(vx)[1]
                    clean_mean_x = get_clean_mean(vx)[0]
                    
                    raw_mean_y, outliers_y = vy.mean(), get_clean_mean(vy)[1]
                    clean_mean_y = get_clean_mean(vy)[0]
                    
                    sens_data = [
                        {"Variable": var_x, "Promedio (Todos)": f"{raw_mean_x:.2f}", "Promedio (Sin Atípicos)": f"{clean_mean_x:.2f}", "N Atípicos": outliers_x},
                        {"Variable": var_y, "Promedio (Todos)": f"{raw_mean_y:.2f}", "Promedio (Sin Atípicos)": f"{clean_mean_y:.2f}", "N Atípicos": outliers_y}
                    ]
                    
                    st.dataframe(pd.DataFrame(sens_data), hide_index=True, use_container_width=True)
                    
                    if "Pearson" in method_used:
                         st.warning("⚠️ Nota: Pearson es sensible a valores atípicos. Si observa una gran diferencia en los promedios, el resultado podría estar sesgado.")
                    else:
                         st.info("✅ Nota: Spearman/Kendall trabajan con rangos, por lo que estos valores atípicos NO afectan drásticamente su resultado. Esta tabla es informativa.")

                    # --- REPORTE TESIS ---
                    with st.expander("📝 Interpretación para Tesis (Generado Automáticamente)", expanded=True):
                        # Lógica de texto
                        direction = "positiva" if r_val > 0 else "negativa"
                        abs_r = abs(r_val)
                        magnitude = "muy baja"
                        if abs_r >= 0.9: magnitude = "muy alta"
                        elif abs_r >= 0.7: magnitude = "alta"
                        elif abs_r >= 0.5: magnitude = "moderada"
                        elif abs_r >= 0.3: magnitude = "baja"
                        
                        sig_txt = "estadísticamente significativa" if pval < alpha else "no significativa"
                        
                        # APA String
                        apa_str_r = f"r = {r_val:.2f}, p = {pval:.3f}"
                        st.markdown(f"**Reporte APA (Sugerido):**\n> {apa_str_r}")
                        
                        reporte = (
                            f"Se analizó la relación entre **{var_x}** y **{var_y}** mediante el coeficiente de correlación de **{method_used}**. "
                            f"Se obtuvo un coeficiente r = {r_val:.3f}, lo que indica una correlación **{direction} {magnitude}** entre las variables "
                            f"(p = {pval:.4f}). Considerando un alpha de {alpha}, esta relación es **{sig_txt}**."
                        )
                        st.info(reporte)

                    # Gráfico Scatter
                    st.write("#### Diagrama de Dispersión")
                    try:
                        # Usar vars limpias para plot
                        df_viz = pd.concat([vx, vy], axis=1).dropna()
                        # Visualización con plots marginales
                        fig = px.scatter(df_viz, x=var_x, y=var_y, trendline="ols",
                                         marginal_x="box", marginal_y="box",
                                         title=f"Correlación: {var_x} vs {var_y} (r={r_val:.2f})")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.caption(f"No visualizable: {e}")

        else: # Odds Ratio
            st.markdown("##### Odds Ratio (Riesgo)")
            st.caption("Requiere tabla 2x2. Seleccione variables binarias (0/1 o Si/No).")
            
            c1, c2 = st.columns(2)
            cats = df.select_dtypes(include=['object','category','bool','number']).columns.tolist() # A veces binarias son num
            
            with c1: v_case = st.selectbox("Variable Caso/Control (Filas):", cats, key='or_v1')
            with c2: v_exp = st.selectbox("Variable Exposición (Columnas):", [c for c in cats if c != v_case], key='or_v2')
            
            if st.button("Calcular Odds Ratio"):
                # Crear tabla 2x2
                try:
                    ct = pd.crosstab(df[v_case], df[v_exp])
                    if ct.shape != (2,2):
                        st.error(f"La tabla generada no es 2x2 ({ct.shape}). Verifique que las variables sean binarias.")
                        st.write(ct)
                    else:
                        st.write("Tabla de Contingencia (Base para OR):")
                        st.dataframe(ct)
                        
                        # Llamar backend
                        res = odds_ratio_risk(ct)
                        
                        if "error" in res:
                            st.error(res['error'])
                        else:
                            col1, col2 = st.columns(2)
                            col1.metric("Odds Ratio", f"{res['statistic']:.2f}")
                            col2.metric("IC 95%", f"[{res['ci_lower']:.2f} - {res['ci_upper']:.2f}]")
                            st.success(res['interpretation'])
                            
                except Exception as e:
                    st.error(f"Error procesando datos: {e}")
