"""
Módulo de Renderizado: Psicometría
----------------------------------
Alpha de Cronbach y análisis de ítems.
"""

import streamlit as st
import pandas as pd
import numpy as np

from modules.stats.psychometrics import cronbach_alpha, item_total_correlation, item_analysis
from modules.stats.utils import get_numeric_columns

def render_psicometria(df: pd.DataFrame = None, 
                      items_vars: list = None,
                      criterion_var: str = None):
    """Panel de Psicometría."""
    st.subheader("🧠 Análisis Psicométrico")
    
    if df is None:
        if 'df_principal' in st.session_state:
            df = st.session_state.df_principal
        else: return

    numericas = get_numeric_columns(df)
    
    if items_vars is None:
        items_vars = st.multiselect("Seleccione Ítems de la Escala:", numericas)
        
    if len(items_vars) < 2:
        st.info("Seleccione al menos 2 ítems.")
        return
        
    # Ejecutar análisis
    if st.button("Calcular Confiabilidad"):
        res = cronbach_alpha(df[items_vars])
        
        if "error" in res:
             st.error(res['error'])
             return
             
        # Mostrar Alpha
        st.metric("Alpha de Cronbach", f"{res.get('alpha',0):.3f}", help=res.get('interpretation'))
        st.write(f"Interpretación: **{res.get('interpretation')}** (N={res.get('n_obs')}, k={res.get('n_items')})")
        
        # Análisis detallado
        st.write("### Análisis de Ítems")
        
        # Checkbox para incluir criterio?
        # Simplemente item analysis completo
        detail = item_analysis(df[items_vars])
        
        # 1. Asegurar tipos numéricos antes de formatear
        cols_to_convert = ['Discrimination (D)', 'Item-Total Corr']
        # Also include other potentially numeric columns if they exist in detail
        # Item analysis usually returns: Mean, SD, Difficulty, Discrimination, Item-Total Corr, Alpha if Deleted
        # Let's convert all except 'Item' name if it was the index (though detail usually has items as index)
        
        # Safe explicit conversion for known columns to target
        for col in detail.columns:
             detail[col] = pd.to_numeric(detail[col], errors='coerce')
             
        # Use subset for background gradient only on specific cols if they exist
        subset_cols = [c for c in ['Discrimination (D)', 'Item-Total Corr'] if c in detail.columns]
        
        st.dataframe(detail.style.format("{:.3f}", na_rep="").background_gradient(cmap='RdYlGn', subset=subset_cols))
        
        st.caption("Nota: 'Difficulty Index' cercano a 1 es fácil. 'Discrimination' > 0.3 es bueno.")
