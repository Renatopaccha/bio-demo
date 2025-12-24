"""
Módulo de Renderizado: Análisis Avanzado (Multivariado)
------------------------------------------------------
PCA, MANOVA y Clustering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.stats.multivariate import pca_analysis, kmeans_clustering
from modules.stats.utils import get_numeric_columns

def render_multivariado(df: pd.DataFrame = None, 
                   analysis_type: str = 'pca', 
                   selected_vars: list = None):
    """Panel multivariado."""
    st.subheader("🚀 Análisis Multivariado")
    
    if df is None:
        if 'df_principal' in st.session_state:
            df = st.session_state.df_principal
        else: return

    # Selector de tipo
    analysis_type = st.radio("Método:", ["PCA (Componentes Principales)", "Clustering (K-Means)"], horizontal=True)
    
    numericas = get_numeric_columns(df)
    if len(numericas) < 2:
        st.error("Se requieren variables numéricas.")
        return
        
    if selected_vars is None:
         selected_vars = st.multiselect("Variables a incluir:", numericas, default=numericas[:5] if len(numericas)>5 else numericas)
         
    if len(selected_vars) < 2: return
    
    # --------------------------------------------------------------------------
    # PCA
    # --------------------------------------------------------------------------
    if "PCA" in analysis_type:
        if st.button("Ejecutar PCA"):
            res = pca_analysis(df[selected_vars])
            
            if "error" in res:
                st.error(res['error'])
                return
                
            # Varianza Explicada
            var_exp = res['explained_variance_ratio']
            cum_var = res['cumulative_variance']
            
            st.write("### Varianza Explicada")
            col1, col2 = st.columns(2)
            col1.dataframe(pd.DataFrame({
                "Componente": [f"PC{i+1}" for i in range(len(var_exp))],
                "Varianza (%)": [f"{v*100:.2f}%" for v in var_exp],
                "Acumulada (%)": [f"{v*100:.2f}%" for v in cum_var]
            }))
            
            # Scree Plot
            fig, ax = plt.subplots()
            ax.plot(range(1, len(var_exp)+1), var_exp, 'o-')
            ax.set_title("Scree Plot")
            ax.set_xlabel("Componente")
            ax.set_ylabel("Varianza Explicada")
            col2.pyplot(fig)
            
            # Loadings
            st.write("### Loadings (Contribución de Variables)")
            st.dataframe(res['loadings'].style.background_gradient(cmap='coolwarm'))
            
            # Biplot (PC1 vs PC2)
            if len(var_exp) >= 2:
                data_trans = res['transformed_data']
                fig2, ax2 = plt.subplots(figsize=(8,6))
                ax2.scatter(data_trans['PC1'], data_trans['PC2'], alpha=0.5)
                ax2.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
                ax2.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
                ax2.set_title("Score Plot (PC1 vs PC2)")
                st.pyplot(fig2)

    # --------------------------------------------------------------------------
    # CLUSTERING
    # --------------------------------------------------------------------------
    elif "Clustering" in analysis_type:
        n_clusters = st.slider("Número de Clusters (k):", 2, 10, 3)
        if st.button("Ejecutar K-Means"):
            res = kmeans_clustering(df[selected_vars], n_clusters)
            
            if "error" in res:
                st.error(res['error'])
                return
                
            st.success(f"Clustering completado. Silhouette Score: {res.get('silhouette_score',0):.3f}")
            
            # Gráfico simple (usando las 2 primeras vars o PCA reducido interno)
            # Para visualización rápida, usamos las dos primeras vars seleccionadas
            if len(selected_vars) >= 2:
                v1, v2 = selected_vars[0], selected_vars[1]
                labels = res['labels']
                
                fig, ax = plt.subplots()
                scatter = ax.scatter(df[v1], df[v2], c=labels, cmap='viridis', alpha=0.6)
                ax.set_xlabel(v1)
                ax.set_ylabel(v2)
                ax.set_title("Clusters (proyección 2 variables)")
                plt.colorbar(scatter, label='Cluster')
                st.pyplot(fig)
