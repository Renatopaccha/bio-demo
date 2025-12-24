import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from modules.utils import boton_guardar_grafico, boton_guardar_tabla

# ================================================
# CONSTANTES DE ESTILO GLOBAL PARA GRÁFICOS
# ================================================

# Configuración global de matplotlib
plt.style.use('default')

# CONSTANTES DE ESTILO
COLORS_PRIMARY = {
    'primary': '#208080',      # Teal profesional
    'secondary': '#FF6B6B',    # Rojo suave
    'success': '#51CF66',      # Verde
    'warning': '#FFD43B',      # Amarillo
    'error': '#FF6B6B',        # Rojo
    'info': '#74C0FC'          # Azul
}

COLORS_PALETTE = sns.color_palette("husl", 8)  # Paleta moderna

FONTS = {
    'family': 'sans-serif',
    'size': 11,
    'weight': 'normal'
}

TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10

# Tamaño de figuras por tipo
FIGSIZE = {
    'small': (8, 5),
    'medium': (12, 6),
    'large': (14, 8),
    'wide': (16, 6),
    'square': (10, 10)
}

# Estilo seaborn
sns.set_palette("husl")
sns.set_style("whitegrid")

def apply_style_to_ax(ax, title="", xlabel="", ylabel="", grid=True):
    """
    Aplica estilo consistente a un eje.
    """
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight='bold')
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight='bold')
    
    # Tamaño de ticks
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    # Grid
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Spines (bordes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax

# ================================================
# FUNCIONES HELPER DE GRÁFICOS
# ================================================

@st.cache_data
def plot_distribucion_numerica(df, variable, tipo='hist', show_normal=False):
    """
    Gráfico de distribución de variable numérica.
    """
    try:
        data = df[variable].dropna()
        if len(data) == 0: raise ValueError(f"Variable {variable} vacía")
        
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        
        if tipo == 'hist':
            sns.histplot(data, kde=True, ax=ax, color=COLORS_PRIMARY['primary'], 
                        alpha=0.6, edgecolor='white', linewidth=0.5, label='Datos')
            
            if show_normal:
                mu, std = norm.fit(data)
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                # Escalar normal a la altura del histograma (densidad)
                # Para esto necesitamos que el hist sea densidad o usar eje secundario.
                # Simplificación: Usar stat='density' en histplot si se pide normal
                ax.clear()
                sns.histplot(data, kde=True, stat="density", ax=ax, color=COLORS_PRIMARY['primary'], alpha=0.6, label='Datos')
                ax.plot(x, p, 'r--', linewidth=2, label=f"Normal ($\mu$={mu:.2f}, $\sigma$={std:.2f})")
        
        elif tipo == 'kde':
            sns.kdeplot(data, fill=True, ax=ax, color=COLORS_PRIMARY['primary'], linewidth=2)
        
        # Líneas de referencia
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color=COLORS_PRIMARY['error'], linestyle='--', linewidth=1.5, label=f'Media: {mean_val:.2f}')
        ax.axvline(median_val, color=COLORS_PRIMARY['success'], linestyle='--', linewidth=1.5, label=f'Mediana: {median_val:.2f}')
        
        ax = apply_style_to_ax(ax, title=f'Distribución de {variable}', xlabel=variable, ylabel='Frecuencia/Densidad')
        ax.legend(loc='best', fontsize=LEGEND_SIZE, framealpha=0.95)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_distribucion_numerica: {str(e)}")

@st.cache_data
def plot_boxplot_univariado(df, variable):
    try:
        data = df[variable].dropna()
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        sns.boxplot(x=data, ax=ax, color=COLORS_PRIMARY['info'], width=0.5,
                   showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
        ax = apply_style_to_ax(ax, title=f'Diagrama de Caja de {variable}', xlabel=variable)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_boxplot_univariado: {str(e)}")

@st.cache_data
def plot_barras_categorico(df, variable, orient='Vertical', order_freq=False):
    try:
        data = df[variable].dropna()
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        
        order = data.value_counts().index if order_freq else None
        
        if orient == 'Vertical':
            sns.countplot(x=data, order=order, ax=ax, palette="viridis")
            plt.xticks(rotation=45, ha='right')
        else:
            sns.countplot(y=data, order=order, ax=ax, palette="viridis")
            
        ax = apply_style_to_ax(ax, title=f'Frecuencias de {variable}', xlabel=variable if orient=='Vertical' else 'Conteo', ylabel='Conteo' if orient=='Vertical' else variable)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_barras_categorico: {str(e)}")

@st.cache_data
def plot_pie_chart(df, variable):
    try:
        data = df[variable].dropna()
        counts = data.value_counts()
        fig, ax = plt.subplots(figsize=FIGSIZE['small'])
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax.axis('equal')
        ax.set_title(f"Distribución de {variable}", fontsize=TITLE_SIZE, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_pie_chart: {str(e)}")

@st.cache_data
def plot_scatter_con_regresion(df, x_var, y_var, color_var=None, add_reg=False):
    try:
        data = df[[x_var, y_var]].dropna()
        if color_var and color_var != "Ninguno":
            data[color_var] = df[color_var]
            data = data.dropna(subset=[color_var])
            hue = color_var
        else:
            hue = None
            
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        
        if add_reg:
            sns.regplot(data=df, x=x_var, y=y_var, scatter=False, color=COLORS_PRIMARY['error'], ax=ax, label='Tendencia')
        
        sns.scatterplot(data=df, x=x_var, y=y_var, hue=hue, ax=ax, alpha=0.7, s=60, edgecolor='white')
        
        ax = apply_style_to_ax(ax, title=f'{y_var} vs {x_var}', xlabel=x_var, ylabel=y_var)
        if hue or add_reg: ax.legend(loc='best', fontsize=LEGEND_SIZE)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_scatter_con_regresion: {str(e)}")

@st.cache_data
def plot_lineplot(df, x_var, y_var, color_var=None):
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
        hue = color_var if color_var != "Ninguno" else None
        sns.lineplot(data=df, x=x_var, y=y_var, hue=hue, markers=True, ax=ax, linewidth=2)
        if len(df[x_var].unique()) > 10: plt.xticks(rotation=45, ha='right')
        ax = apply_style_to_ax(ax, title=f'Tendencia de {y_var} por {x_var}', xlabel=x_var, ylabel=y_var)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_lineplot: {str(e)}")

@st.cache_data
def plot_boxplot_bivariado(df, x_var, y_var, color_var=None):
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        hue = color_var if color_var != "Ninguno" else None
        sns.boxplot(data=df, x=x_var, y=y_var, hue=hue, palette="viridis", ax=ax)
        if len(df[x_var].unique()) > 5: plt.xticks(rotation=45, ha='right')
        ax = apply_style_to_ax(ax, title=f'Distribución de {y_var} por {x_var}', xlabel=x_var, ylabel=y_var)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_boxplot_bivariado: {str(e)}")

@st.cache_data
def plot_violin_bivariado(df, x_var, y_var, color_var=None):
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        hue = color_var if color_var != "Ninguno" else None
        split = False
        if hue and df[hue].nunique() == 2: split = True
        
        sns.violinplot(data=df, x=x_var, y=y_var, hue=hue, split=split, inner="quart", palette="viridis", ax=ax)
        if len(df[x_var].unique()) > 5: plt.xticks(rotation=45, ha='right')
        ax = apply_style_to_ax(ax, title=f'Distribución de {y_var} por {x_var}', xlabel=x_var, ylabel=y_var)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_violin_bivariado: {str(e)}")

@st.cache_data
def plot_barplot_bivariado(df, x_var, y_var, color_var=None):
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        hue = color_var if color_var != "Ninguno" else None
        sns.barplot(data=df, x=x_var, y=y_var, hue=hue, palette="viridis", ax=ax, capsize=.1)
        if len(df[x_var].unique()) > 5: plt.xticks(rotation=45, ha='right')
        ax = apply_style_to_ax(ax, title=f'Promedio de {y_var} por {x_var}', xlabel=x_var, ylabel=y_var)
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_barplot_bivariado: {str(e)}")

@st.cache_data
def plot_heatmap_multivariado(df, vars_num, triangular=False):
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE['square'])
        corr = df[vars_num].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool)) if triangular else None
        sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, vmin=-1, vmax=1, square=True)
        ax.set_title("Matriz de Correlación", fontsize=TITLE_SIZE, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        raise Exception(f"Error en plot_heatmap_multivariado: {str(e)}")

@st.cache_data
def plot_pairplot(df, vars_num, hue_opt=None):
    try:
        h = hue_opt if hue_opt != "Ninguno" else None
        # Pairplot devuelve un Grid, no una Fig directamente, pero podemos acceder a g.fig
        g = sns.pairplot(df, vars=vars_num, hue=h, diag_kind="kde", corner=True, palette="viridis", height=2.5)
        g.fig.suptitle("Matriz de Dispersión (Pairplot)", fontsize=TITLE_SIZE, fontweight='bold', y=1.02)
        return g.fig
    except Exception as e:
        raise Exception(f"Error en plot_pairplot: {str(e)}")

@st.cache_data
def plot_facetgrid(df, x_val, y_val, col_val, row_val=None, tipo_inner="Scatter"):
    try:
        r = row_val if row_val != "Ninguno" else None
        g = sns.FacetGrid(df, col=col_val, row=r, margin_titles=True, height=3.5, aspect=1.2)
        
        if tipo_inner == "Scatter":
            g.map(sns.scatterplot, x_val, y_val, alpha=0.7)
        elif tipo_inner == "Boxplot":
            g.map(sns.boxplot, x_val, y_val, order=sorted(df[x_val].unique()) if df[x_val].dtype == 'object' else None)
        elif tipo_inner == "Barplot":
            g.map(sns.barplot, x_val, y_val, order=sorted(df[x_val].unique()) if df[x_val].dtype == 'object' else None)
        
        g.add_legend()
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f"FacetGrid: {y_val} vs {x_val} por {col_val}", fontsize=TITLE_SIZE, fontweight='bold')
        return g.fig
    except Exception as e:
        raise Exception(f"Error en plot_facetgrid: {str(e)}")

# ================================================
# FUNCIÓN PRINCIPAL
# ================================================

def render_graficos():
    """
    Renderiza la suite gráfica avanzada.
    Incluye validaciones de tipos de datos antes de graficar.
    """
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Suite Gráfica Avanzada")
    df = st.session_state.df_principal
    
    cat_graf = st.selectbox("Selecciona la Categoría de Análisis:", 
                            ["1. Univariado (Una Variable)", "2. Bivariado (Dos Variables)", "3. Multivariado (Complejo)"])
    
    # --- 1. UNIVARIADO ---
    if cat_graf == "1. Univariado (Una Variable)":
        st.subheader("Análisis de una sola variable")
        var = st.selectbox("Selecciona Variable", df.columns)
        
        if var:
            # DETECTAR TIPO
            is_num = pd.api.types.is_numeric_dtype(df[var])
            
            if is_num:
                tipo = st.selectbox("Tipo de Gráfico", ["Histograma Avanzado", "Boxplot (Caja)", "Densidad (Density Plot)"])
                
                # --- VALIDACIÓN UNIVARIADA (NUMÉRICA) ---
                if not pd.api.types.is_numeric_dtype(df[var]):
                    st.warning("⚠️ La variable seleccionada es de Texto. Para estos gráficos necesitas una variable Numérica (ej. Edad). Prueba con Gráfico de Barras.")
                    st.stop()
                
                fig = None
                if tipo == "Histograma Avanzado":
                    show_norm = st.checkbox("¿Superponer Curva Normal Teórica?")
                    fig = plot_distribucion_numerica(df, var, tipo='hist', show_normal=show_norm)
                    
                elif tipo == "Boxplot (Caja)":
                    fig = plot_boxplot_univariado(df, var)
                    # Estadísticos
                    data = df[var].dropna()
                    q1, q3 = data.quantile(0.25), data.quantile(0.75)
                    st.caption(f"📍 **Estadísticos:** Media: {data.mean():.2f} | Mediana: {data.median():.2f} | Q1: {q1:.2f} | Q3: {q3:.2f}")
                    
                elif tipo == "Densidad (Density Plot)":
                    fig = plot_distribucion_numerica(df, var, tipo='kde')
                
                if fig:
                    st.pyplot(fig)
                    boton_guardar_grafico(fig, f"Gráfico Univariado: {var}", f"uni_{var}_btn")
                
            else: # CATEGÓRICA
                tipo = st.selectbox("Tipo de Gráfico", ["Gráfico de Barras", "Gráfico de Pastel"])
                
                # --- VALIDACIÓN UNIVARIADA (CATEGÓRICA) ---
                if df[var].nunique() > 50:
                    st.warning(f"⚠️ Esta variable tiene demasiadas categorías únicas ({df[var].nunique()}). El gráfico será ilegible.")
                
                fig = None
                if tipo == "Gráfico de Barras":
                    orient = st.radio("Orientación", ["Vertical", "Horizontal"], horizontal=True)
                    order = st.checkbox("¿Ordenar por Frecuencia?")
                    fig = plot_barras_categorico(df, var, orient, order)
                    
                elif tipo == "Gráfico de Pastel":
                    fig = plot_pie_chart(df, var)
                
                if fig:
                    st.pyplot(fig)
                    boton_guardar_grafico(fig, f"Gráfico Univariado: {var}", f"uni_cat_{var}_btn")

    # --- 2. BIVARIADO ---
    elif cat_graf == "2. Bivariado (Dos Variables)":
        st.subheader("Comparación de dos variables")
        tipo_bi = st.selectbox("Tipo de Gráfico", ["Scatterplot (Dispersión)", "Lineplot (Tendencias)", "Boxplot (Cajas)", "Violinplot (Distribución)", "Barplot (Promedios)"])
        
        # --- A) SCATTERPLOT ---
        if tipo_bi == "Scatterplot (Dispersión)":
            c1, c2, c3 = st.columns(3)
            with c1: x_val = st.selectbox("Eje X", df.columns, key="bi_scat_x")
            with c2: y_val = st.selectbox("Eje Y", df.columns, key="bi_scat_y")
            with c3: hue_val = st.selectbox("Color (Hue)", ["Ninguno"] + df.columns.tolist(), key="bi_scat_hue")
            reg = st.checkbox("Agregar Línea de Regresión")
            
            if x_val and y_val:
                # --- VALIDACIÓN SCATTERPLOT ---
                x_is_num = pd.api.types.is_numeric_dtype(df[x_val])
                y_is_num = pd.api.types.is_numeric_dtype(df[y_val])
                
                if not (x_is_num and y_is_num):
                    st.error("⛔ Error Lógico: Para un Scatterplot necesitas dos números (ej. Peso vs Talla). Has seleccionado texto.")
                    st.stop()
                
                fig = plot_scatter_con_regresion(df, x_val, y_val, hue_val, reg)
                st.pyplot(fig)
                boton_guardar_grafico(fig, f"Scatter: {x_val} vs {y_val}", "bi_scat_btn")

        # --- B) LINEPLOT ---
        elif tipo_bi == "Lineplot (Tendencias)":
            c1, c2, c3 = st.columns(3)
            with c1: x_val = st.selectbox("Tiempo/Secuencia (X)", df.columns, key="bi_line_x")
            with c2: y_val = st.selectbox("Valor (Y)", df.select_dtypes(include=np.number).columns, key="bi_line_y")
            with c3: hue_val = st.selectbox("Color (Hue)", ["Ninguno"] + df.columns.tolist(), key="bi_line_hue")
            
            if x_val and y_val:
                fig = plot_lineplot(df, x_val, y_val, hue_val)
                st.pyplot(fig)
                boton_guardar_grafico(fig, f"Lineplot: {y_val} por {x_val}", "bi_line_btn")

        # --- C) BOXPLOT ---
        elif tipo_bi == "Boxplot (Cajas)":
            c1, c2, c3 = st.columns(3)
            with c1: x_val = st.selectbox("Grupo (X)", df.columns, key="bi_box_x")
            with c2: y_val = st.selectbox("Numérica (Y)", df.columns, key="bi_box_y")
            with c3: hue_val = st.selectbox("Subgrupo/Color (Opcional)", ["Ninguno"] + df.columns.tolist(), key="bi_box_hue")
            
            if x_val and y_val:
                # --- VALIDACIÓN BOXPLOT ---
                x_is_num = pd.api.types.is_numeric_dtype(df[x_val])
                y_is_num = pd.api.types.is_numeric_dtype(df[y_val])
                
                if not y_is_num:
                    st.error("⛔ Imposible graficar: El Eje Y debe ser numérico para calcular promedios o medianas.")
                    st.stop()
                
                if x_is_num and y_is_num:
                    st.warning("⚠️ Estás cruzando dos números en un Boxplot. El gráfico intentará tratar el Eje X como grupos, lo cual puede ser muy lento.")
                
                if x_is_num and not y_is_num:
                    st.warning("💡 Aviso: Parece que tus variables están invertidas. Usualmente el Grupo va en X y el Valor en Y. Se graficará horizontalmente.")

                fig = plot_boxplot_bivariado(df, x_val, y_val, hue_val)
                st.pyplot(fig)
                boton_guardar_grafico(fig, f"Boxplot: {y_val} por {x_val}", "bi_box_btn")

        # --- D) VIOLINPLOT ---
        elif tipo_bi == "Violinplot (Distribución)":
            c1, c2, c3 = st.columns(3)
            with c1: x_val = st.selectbox("Grupo (X)", df.columns, key="bi_vio_x")
            with c2: y_val = st.selectbox("Numérica (Y)", df.columns, key="bi_vio_y")
            with c3: hue_val = st.selectbox("Subgrupo/Color (Opcional)", ["Ninguno"] + df.columns.tolist(), key="bi_vio_hue")
            
            if x_val and y_val:
                # --- VALIDACIÓN VIOLINPLOT ---
                x_is_num = pd.api.types.is_numeric_dtype(df[x_val])
                y_is_num = pd.api.types.is_numeric_dtype(df[y_val])
                
                if x_is_num and not y_is_num:
                    st.warning("💡 Aviso: Parece que tus variables están invertidas. Usualmente el Grupo va en X y el Valor en Y.")
                
                if x_is_num and y_is_num:
                    st.warning("⚠️ Estás cruzando dos números. El gráfico intentará tratar el Eje X como grupos.")
                
                if not y_is_num:
                    st.error("⛔ Imposible graficar: El Eje Y debe ser numérico.")
                    st.stop()

                fig = plot_violin_bivariado(df, x_val, y_val, hue_val)
                st.pyplot(fig)
                boton_guardar_grafico(fig, f"Violinplot: {y_val} por {x_val}", "bi_vio_btn")

        # --- E) BARPLOT ---
        elif tipo_bi == "Barplot (Promedios)":
            c1, c2, c3 = st.columns(3)
            with c1: x_val = st.selectbox("Grupo (X)", df.columns, key="bi_bar_x")
            with c2: y_val = st.selectbox("Numérica (Y)", df.columns, key="bi_bar_y")
            with c3: hue_val = st.selectbox("Subgrupo/Color (Opcional)", ["Ninguno"] + df.columns.tolist(), key="bi_bar_hue")
            
            if x_val and y_val:
                # --- VALIDACIÓN BARPLOT ---
                x_is_num = pd.api.types.is_numeric_dtype(df[x_val])
                y_is_num = pd.api.types.is_numeric_dtype(df[y_val])
                
                if x_is_num and not y_is_num:
                    st.warning("💡 Aviso: Parece que tus variables están invertidas. Usualmente el Grupo va en X y el Valor en Y.")
                
                if x_is_num and y_is_num:
                    st.warning("⚠️ Estás cruzando dos números. El gráfico intentará tratar el Eje X como grupos.")
                
                if not y_is_num:
                    st.error("⛔ Imposible graficar: El Eje Y debe ser numérico.")
                    st.stop()

                fig = plot_barplot_bivariado(df, x_val, y_val, hue_val)
                st.pyplot(fig)
                boton_guardar_grafico(fig, f"Barplot: {y_val} por {x_val}", "bi_bar_btn")

    # --- 3. MULTIVARIADO ---
    elif cat_graf == "3. Multivariado (Complejo)":
        st.subheader("Análisis Multidimensional")
        tipo_multi = st.selectbox("Tipo de Gráfico", ["Matriz de Calor (Heatmap)", "Matriz de Dispersión (Pairplot)", "Grid de Facetas (FacetGrid)"])
        
        # --- A) HEATMAP ---
        if tipo_multi == "Matriz de Calor (Heatmap)":
            vars_num = st.multiselect("Selecciona Variables Numéricas", df.select_dtypes(include=np.number).columns, default=df.select_dtypes(include=np.number).columns[:5])
            
            if len(vars_num) < 2:
                st.warning("⚠️ Selecciona al menos 2 variables numéricas.")
            else:
                triangular = st.checkbox("Ver solo mitad inferior (Triangular)")
                fig = plot_heatmap_multivariado(df, vars_num, triangular)
                st.pyplot(fig)
                boton_guardar_grafico(fig, "Heatmap Multivariado", "multi_heat_btn")
                
                corr = df[vars_num].corr()
                boton_guardar_tabla(corr, "Matriz de Correlación", "corr_matrix_table_btn")

        # --- B) PAIRPLOT ---
        elif tipo_multi == "Matriz de Dispersión (Pairplot)":
            vars_num = st.multiselect("Variables Numéricas (Máximo 6 sugeridas)", df.select_dtypes(include=np.number).columns)
            hue_opt = st.selectbox("Dividir por Grupo (Hue)", ["Ninguno"] + df.columns.tolist())
            
            if len(vars_num) > 6:
                st.warning("⚠️ Muchos gráficos pueden hacer lenta la app. Considera reducir variables.")
            
            if len(vars_num) >= 2:
                if st.button("Generar Pairplot"):
                    fig = plot_pairplot(df, vars_num, hue_opt)
                    st.pyplot(fig)
                    boton_guardar_grafico(fig, "Pairplot", "multi_pair_btn")

        # --- C) FACET GRID ---
        elif tipo_multi == "Grid de Facetas (FacetGrid)":
            st.info("Construye gráficos de 4 dimensiones: X, Y, Columnas y Filas.")
            c1, c2 = st.columns(2)
            with c1: x_val = st.selectbox("Eje X", df.columns, key="facet_x")
            with c2: y_val = st.selectbox("Eje Y", df.select_dtypes(include=np.number).columns, key="facet_y")
            
            c3, c4 = st.columns(2)
            with c3: col_val = st.selectbox("Dividir Columnas por...", df.select_dtypes(include=['object', 'category']).columns, key="facet_col")
            with c4: row_val = st.selectbox("Dividir Filas por... (Opcional)", ["Ninguno"] + df.select_dtypes(include=['object', 'category']).columns.tolist(), key="facet_row")
            
            tipo_inner = st.selectbox("Tipo de Gráfico Interno", ["Scatter", "Boxplot", "Barplot"])
            
            if x_val and y_val and col_val:
                if st.button("Generar FacetGrid"):
                    fig = plot_facetgrid(df, x_val, y_val, col_val, row_val, tipo_inner)
                    st.pyplot(fig)
                    boton_guardar_grafico(fig, "FacetGrid", "multi_facet_btn")

    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data
def plot_bland_altman(df, col_a, col_b):
    """
    Genera un gráfico de Bland-Altman profesional para concordancia clínica.
    Calcula Bias y Límites de Acuerdo (LoA) al 95%.
    """
    try:
        # 1. Preparación de datos
        data = df[[col_a, col_b]].dropna()
        x = data[col_a]
        y = data[col_b]
        
        # 2. Cálculos Estadísticos
        mean = (x + y) / 2
        diff = x - y
        md = np.mean(diff)                   # Bias (Sesgo)
        sd = np.std(diff, axis=0)            # Desviación estándar de diferencias
        
        # Límites de acuerdo (LoA 95%)
        loa_upper = md + 1.96 * sd
        loa_lower = md - 1.96 * sd
        
        # 3. Configuración del Gráfico
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        
        # Scatter plot (Diferencias vs Promedios)
        sns.scatterplot(x=mean, y=diff, ax=ax, color=COLORS_PRIMARY['primary'], alpha=0.6, s=50, label='Observaciones')
        
        # Líneas de referencia
        ax.axhline(md, color='black', linestyle='-', lw=2, label=f'Bias (Media Dif): {md:.2f}')
        ax.axhline(loa_upper, color=COLORS_PRIMARY['error'], linestyle='--', lw=1.5, label=f'+1.96 SD: {loa_upper:.2f}')
        ax.axhline(loa_lower, color=COLORS_PRIMARY['error'], linestyle='--', lw=1.5, label=f'-1.96 SD: {loa_lower:.2f}')
        
        # Intervalo de confianza visual (Sombreado entre límites)
        ax.fill_between([mean.min(), mean.max()], loa_lower, loa_upper, color='gray', alpha=0.05)
        
        # Estilizado Médico
        ax = apply_style_to_ax(ax, 
                              title=f"Gráfico de Bland-Altman: {col_a} vs {col_b}", 
                              xlabel=f"Promedio de mediciones (({col_a} + {col_b}) / 2)", 
                              ylabel=f"Diferencia ({col_a} - {col_b})")
        
        # Leyenda fuera del gráfico para no tapar datos
        ax.legend(loc='upper right', frameon=True, fontsize='small')
        plt.tight_layout()
        
        return fig, md, loa_lower, loa_upper
        
    except Exception as e:
        raise Exception(f"Error generando Bland-Altman: {str(e)}")
