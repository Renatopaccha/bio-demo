"""
Módulo de Renderizado: Asociaciones y Correlaciones
---------------------------------------------------
Muestra matrices de correlación (heatmap) y tablas de Chi2.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Opcional, usaremos matplotlib si seaborn falla o simple

from modules.stats.multivariate import correlation_matrix, covariance_matrix
from modules.stats.inference import chi_cuadrado
from modules.stats.validators import validate_and_truncate_variables
from modules.stats.utils import get_numeric_columns

def render_asociaciones(df: pd.DataFrame = None, 
                       selected_vars: list = None, 
                       analysis_type: str = 'correlation'):
    """
    Renderiza matrices de asociación.
    """
    st.subheader("🔗 Asociaciones y Correlaciones")
    
    if df is None:
        if 'df_principal' in st.session_state:
            df = st.session_state.df_principal
        else:
            return

    # Selector de tipo
    if not analysis_type: # if passed as None default
         analysis_type = st.radio("Tipo de Análisis:", ["Correlación (Numéricas)", "Chi-Cuadrado (Categóricas)"], horizontal=True)

    if "Correlación" in analysis_type: # Flexible matching
        numericas = get_numeric_columns(df)
        if len(numericas) < 2:
            st.warning("Se requieren al menos 2 variables numéricas.")
            return
            
        if selected_vars is None:
            selected_vars = st.multiselect("Variables:", numericas, default=numericas[:10])
            
        selected_vars, msg = validate_and_truncate_variables(selected_vars, 20, "Matriz Correlación")
        if msg: st.warning(msg)
        
        if len(selected_vars) < 2: return

        # Matriz
        matriz = correlation_matrix(df[selected_vars])
        
        # Plot Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matriz, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(selected_vars)))
        ax.set_yticks(np.arange(len(selected_vars)))
        ax.set_xticklabels(selected_vars, rotation=45, ha="right")
        ax.set_yticklabels(selected_vars)
        plt.colorbar(im)
        
        # Annotate
        for i in range(len(selected_vars)):
            for j in range(len(selected_vars)):
                text = ax.text(j, i, f"{matriz.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black" if abs(matriz.iloc[i,j])<0.5 else "white")
                               
        st.pyplot(fig)
        
        with st.expander("Ver Tabla Numérica"):
            st.dataframe(matriz.style.background_gradient(cmap='coolwarm', axis=None))
            
    else: # Chi-Square Matrix (Simulated via pairwise)
        categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categoricas) < 2:
            st.warning("Menos de 2 variables categóricas.")
            return

        if selected_vars is None:
             selected_vars = st.multiselect("Variables Categóricas:", categoricas)
             
        if len(selected_vars) < 2: return
        
        # Compute pairwise chi2 p-values
        res_matrix = pd.DataFrame(index=selected_vars, columns=selected_vars)
        
        for v1 in selected_vars:
            for v2 in selected_vars:
                if v1 == v2:
                    res_matrix.loc[v1, v2] = 1.0 # Diagonal ?? P-value 0 or 1? Identity is perfect association -> p ~ 0
                    res_matrix.loc[v1, v2] = 0.0 
                else:
                    res = chi_cuadrado(df[v1], df[v2])
                    if 'p_value' in res:
                        res_matrix.loc[v1, v2] = res['p_value']
                    else:
                         res_matrix.loc[v1, v2] = np.nan
        
        st.write("Matriz de p-valores (Chi-Cuadrado):")
        st.dataframe(res_matrix.astype(float).style.background_gradient(cmap='Reds_r', vmin=0, vmax=0.05))
        st.caption("Rojo intenso = P-valor bajo (Asociación significativa).")
