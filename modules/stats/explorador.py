import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats

def render_explorador():
    """
    Módulo de Exploración de Datos (Reescrito desde Cero).
    Simple, Robusto y Manual.
    """
    # 1. Obtención de Datos
    if 'df_principal' not in st.session_state:
        st.error("❌ No se encontró el dataset principal. Por favor, carga un archivo primero.")
        return

    df = st.session_state.df_principal
    
    if df is None or df.empty:
        st.error("❌ El dataset está vacío.")
        return

    st.title("🔍 Explorador de Datos")

    tab1, tab2 = st.tabs(["📊 Estudio Gráfico (Manual)", "📋 Constructor de Tablas"])

    # -----------------------------------------------------------------------------
    # PESTAÑA 1: GRÁFICOS
    # -----------------------------------------------------------------------------
    with tab1:
        st.subheader("Generador Gráfico Manual")
        
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        
        with col_sel1:
            x_val = st.selectbox("Eje X (Variable):", df.columns, key="x_val_manual")
        
        with col_sel2:
            opciones_y = ["Ninguno"] + list(df.columns)
            y_val = st.selectbox("Eje Y (Opcional):", opciones_y, key="y_val_manual")
            
        with col_sel3:
            opciones_color = ["Ninguno"] + list(df.columns)
            color_val = st.selectbox("Color (Agrupación):", opciones_color, key="color_val_manual")
            
        # -- Algoritmo de Decisión --
        tipos_validos = []
        msg_debug = "" # Para depuración si es necesario

        # Detectar Tipos
        is_x_num = pd.api.types.is_numeric_dtype(df[x_val])
        
        is_y_active = y_val != "Ninguno"
        is_y_num = False
        if is_y_active:
            is_y_num = pd.api.types.is_numeric_dtype(df[y_val])
            
        # Lógica de Gráficos
        if not is_y_active:
            # Caso Univariado
            if is_x_num:
                tipos_validos = ["Histograma", "Boxplot", "Violin", "Densidad"]
            else:
                tipos_validos = ["Barras (Conteo)", "Pastel"]
        else:
            # Caso Bivariado
            if is_x_num and is_y_num:
                tipos_validos = ["Scatter Plot", "Línea", "Área"]
            elif not is_x_num and is_y_num:
                # X Cat, Y Num
                tipos_validos = ["Boxplot", "Violin", "Barras (Promedio)"]
            elif is_x_num and not is_y_num:
                # X Num, Y Cat
                tipos_validos = ["Boxplot Horizontal", "Violin Horizontal"]
            else:
                # X Cat, Y Cat
                tipos_validos = ["Heatmap", "Barras Apiladas"]

        # Selector de Tipo
        if tipos_validos:
            tipo = st.selectbox("🎨 Tipo de Gráfico:", tipos_validos)
            
            # Preparar argumento de color
            color_arg = color_val if color_val != "Ninguno" else None
            
            fig = None
            
            try:
                # -- Renderizado --
                if tipo == "Histograma":
                    fig = px.histogram(df, x=x_val, color=color_arg, marginal="box", title=f"Distribución de {x_val}")
                elif tipo == "Boxplot":
                    # Puede ser univariado o bivariado (X Cat, Y Num) -> pero aquí entramos por nombre, cuidado
                    # Si estamos en univariado:
                    if not is_y_active:
                         fig = px.box(df, y=x_val, color=color_arg, title=f"Boxplot de {x_val}")
                    else:
                        # Bivariado X Cat, Y Num
                        fig = px.box(df, x=x_val, y=y_val, color=color_arg, title=f"{y_val} por {x_val}")

                elif tipo == "Violin":
                    if not is_y_active:
                        fig = px.violin(df, y=x_val, box=True, color=color_arg, title=f"Violin de {x_val}")
                    else:
                        fig = px.violin(df, x=x_val, y=y_val, box=True, color=color_arg, title=f"{y_val} por {x_val}")
                        
                elif tipo == "Densidad":
                    # Solo univariado numérico realmente 'lindo' con histograma o distplot, 
                    # pero px no tiene kde puro simple, se usa histograma con prob.
                    fig = px.histogram(df, x=x_val, color=color_arg, histnorm='probability density', title=f"Densidad de {x_val}")
                    
                elif tipo == "Barras (Conteo)":
                    fig = px.histogram(df, x=x_val, color=color_arg, title=f"Conteo de {x_val}")
                    
                elif tipo == "Pastel":
                    fig = px.pie(df, names=x_val, title=f"Proporción de {x_val}")
                    
                elif tipo == "Scatter Plot":
                    fig = px.scatter(df, x=x_val, y=y_val, color=color_arg, title=f"{x_val} vs {y_val}")
                    
                elif tipo == "Línea":
                    fig = px.line(df, x=x_val, y=y_val, color=color_arg, title=f"{x_val} vs {y_val}")
                    
                elif tipo == "Área":
                    fig = px.area(df, x=x_val, y=y_val, color=color_arg, title=f"{x_val} vs {y_val}")
                    
                elif tipo == "Barras (Promedio)":
                    # X Cat, Y Num
                    df_agg = df.groupby(x_val)[y_val].mean().reset_index()
                    fig = px.bar(df_agg, x=x_val, y=y_val, title=f"Promedio de {y_val} por {x_val}")
                    
                elif tipo == "Boxplot Horizontal":
                    # X Num, Y Cat
                    fig = px.box(df, x=x_val, y=y_val, color=color_arg, orientation='h', title=f"{x_val} por {y_val}")
                    
                elif tipo == "Violin Horizontal":
                    fig = px.violin(df, x=x_val, y=y_val, color=color_arg, orientation='h', box=True, title=f"{x_val} por {y_val}")
                    
                elif tipo == "Heatmap":
                    # X Cat, Y Cat - Densidad 2D o Crosstab
                    ft = pd.crosstab(df[x_val], df[y_val])
                    fig = px.imshow(ft, text_auto=True, title=f"Mapa de Calor: {x_val} vs {y_val}")
                    
                elif tipo == "Barras Apiladas":
                    # X Cat, Y Cat -> Conteo
                    fig = px.histogram(df, x=x_val, color=y_val, barmode='stack', title=f"{x_val} apilado por {y_val}")

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al generar gráfico: {str(e)}")
                
        else:
            st.info("⚠️ La combinación de variables seleccionada no tiene gráficos predefinidos simples.")

    # -----------------------------------------------------------------------------
    # PESTAÑA 2: TABLAS
    # -----------------------------------------------------------------------------
    with tab2:
        st.subheader("Constructor de Tablas Resumen")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            vars_tabla = st.multiselect("Seleccionar Variables (Filas):", df.columns)
        
        with col_t2:
            opciones_grupo_tab = ["Ninguno"] + list(df.columns)
            grupo_tabla = st.selectbox("Agrupar por (Columnas):", opciones_grupo_tab)
            
        st.markdown("##### Métricas a calcular")
        col_m1, col_m2, col_m3, col_m4, col_m5, col_m6, col_m7 = st.columns(7)
        
        met_n = col_m1.checkbox("N", value=True)
        met_pct = col_m2.checkbox("%")
        met_mean = col_m3.checkbox("Media", value=True)
        met_sd = col_m4.checkbox("DE")
        met_med = col_m5.checkbox("Mediana")
        met_minmax = col_m6.checkbox("Min/Max")
        met_p = col_m7.checkbox("Valor P")
        
        if st.button("🛠️ Construir Tabla"):
            if not vars_tabla:
                st.warning("Selecciona al menos una variable.")
            else:
                datos_tabla = []
                
                # Definir grupos
                if grupo_tabla != "Ninguno":
                    grupos_unicos = df[grupo_tabla].dropna().unique()
                    grupos_unicos.sort()
                
                for var in vars_tabla:
                    row = {"Variable": var}
                    es_num = pd.api.types.is_numeric_dtype(df[var])
                    
                    # --- Lógica SIN GRUPO ---
                    if grupo_tabla == "Ninguno":
                        series = df[var].dropna()
                        if met_n: row["N"] = len(series)
                        
                        if es_num:
                            if met_mean: row["Media"] = round(series.mean(), 2)
                            if met_sd: row["DE"] = round(series.std(), 2)
                            if met_med: row["Mediana"] = round(series.median(), 2)
                            if met_minmax: row["Rango"] = f"{round(series.min(), 2)} - {round(series.max(), 2)}"
                        else:
                            # Categórica
                            if met_n or met_pct:
                                top = series.mode()[0] if not series.empty else "N/A"
                                count = series.value_counts().max() if not series.empty else 0
                                if met_pct: row["Top Cat (%)"] = f"{top} ({round((count/len(series))*100, 1)}%)"
                    
                    # --- Lógica CON GRUPO ---
                    else:
                        # Calcular P-Value Global
                        p_val_str = "-"
                        if met_p:
                            try:
                                datos_grupos = [df[df[grupo_tabla] == g][var].dropna() for g in grupos_unicos]
                                # Filtrar vacíos
                                datos_grupos = [d for d in datos_grupos if len(d) > 0]
                                
                                if len(datos_grupos) >= 2:
                                    if es_num:
                                        # Numérico -> ANOVA o Kruskal ?? (Usamos Kruskal por robustez en exploratorio simple)
                                        # O One-way ANOVA si preferimos. Usaremos Kruskal para evitar supuestos fuertes aquí.
                                        stat, p = stats.kruskal(*datos_grupos)
                                        p_val_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
                                    else:
                                        # Categórico -> Chi2
                                        contingency = pd.crosstab(df[var], df[grupo_tabla])
                                        c, p, dof, expected = stats.chi2_contingency(contingency)
                                        p_val_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
                            except:
                                p_val_str = "Err"
                        
                        if met_p: row["Valor P"] = p_val_str

                        # Métricas por grupo
                        for g in grupos_unicos:
                            sub_s = df[df[grupo_tabla] == g][var].dropna()
                            pref = f"{g}"
                            
                            if met_n: row[f"{pref} (N)"] = len(sub_s)
                            
                            if es_num:
                                if met_mean: row[f"{pref} (Media)"] = round(sub_s.mean(), 2) if not sub_s.empty else 0
                                if met_sd: row[f"{pref} (DE)"] = round(sub_s.std(), 2) if not sub_s.empty else 0
                            else:
                                if met_pct and not sub_s.empty:
                                    top = sub_s.mode()[0]
                                    cnt = sub_s.value_counts().max()
                                    row[f"{pref} (Modo %)"] = f"{top} ({round((cnt/len(sub_s))*100, 0)}%)"

                    datos_tabla.append(row)
                
                df_res = pd.DataFrame(datos_tabla)
                st.dataframe(df_res, use_container_width=True)
