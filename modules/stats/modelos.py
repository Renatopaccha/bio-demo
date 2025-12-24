"""
Módulo de Renderizado: Modelos de Regresión
-------------------------------------------
Regresión Lineal Simple, Múltiple y Logística con diagnósticos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.stats.regression import (
    linear_regression_simple, linear_regression_multiple, logistic_regression_binary,
    verify_regression_assumptions
)
from modules.stats.utils import format_p_value, format_ci

def render_regression(df: pd.DataFrame = None, 
                   dependent_var: str = None, 
                   independent_vars: list = None,
                   model_type: str = 'linear'):
    """Panel de Modelos de Regresión."""
    st.subheader("📈 Modelos de Regresión")
    
    if df is None:
        if 'df_principal' in st.session_state:
            df = st.session_state.df_principal
        else: return

    # Configuración del Modelo
    c1, c2 = st.columns(2)
    with c1:
        model_type = st.selectbox("Tipo de Modelo:", ["Lineal (OLS)", "Logística (Binaria)"], index=0)
    
    all_cols = df.columns.tolist()
    
    # Selectores
    with c1: dependent_var = st.selectbox("Variable Dependiente (Y):", all_cols)
    with c2: independent_vars = st.multiselect("Variables Independientes (X):", [c for c in all_cols if c!=dependent_var])
    
    if not dependent_var or not independent_vars:
        st.info("Configure las variables para ajustar el modelo.")
        return
        
    if st.button("Ajustar Modelo", type="primary"):
        with st.spinner("Calculando..."):
            try:
                # LINEAL
                if "Lineal" in model_type:
                    if len(independent_vars) == 1:
                        # Simple
                        st.info("Usando Regresión Lineal Simple.")
                        x_series = df[independent_vars[0]]
                        y_series = df[dependent_var]
                        res = linear_regression_simple(x_series, y_series)
                        
                        # Pack simple result into similar structure as multiple for display code reuse?
                        # Or display simple specific
                        st.metric("R-Cuadrado", f"{res.get('r_squared', 0):.3f}")
                        st.write(f"Ecuación: Y = {res.get('intercept',0):.2f} + {res.get('slope',0):.2f}*X")
                        
                        # Plot
                        fig, ax = plt.subplots()
                        ax.scatter(x_series, y_series, alpha=0.5)
                        # Line
                        x_range = np.linspace(x_series.min(), x_series.max(), 100)
                        y_pred = res.get('intercept',0) + res.get('slope',0)*x_range
                        ax.plot(x_range, y_pred, color='red')
                        st.pyplot(fig)
                        
                    else:
                        # Multiple
                        st.info("Usando Regresión Lineal Múltiple.")
                        X = df[independent_vars]
                        
                        # 1. Primero creamos las Dummies (esto puede generar nombres numéricos)
                        if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
                            st.info("ℹ️ Aplicando codificación automática a variables categóricas...")
                            X = pd.get_dummies(X, drop_first=True)

                        # 2. INMEDIATAMENTE DESPUÉS, convertimos TODOS los nombres de columnas a string
                        # (Esta línea es la que soluciona el error, debe ir DESPUÉS del get_dummies)
                        X.columns = X.columns.astype(str)

                        # 3. Aseguramos que los datos sean numéricos
                        X = X.astype(float)

                        y = df[dependent_var]
                        res = linear_regression_multiple(X, y)
                        
                        if "error" in res:
                            st.error(res['error'])
                            return
                            
                        # Métricas Globales
                        m1, m2, m3 = st.columns(3)
                        m1.metric("R²", f"{res.get('r_squared',0):.3f}")
                        m2.metric("R² Ajustado", f"{res.get('adj_r_squared',0):.3f}")
                        m3.metric("F p-valor", format_p_value(res.get('f_pvalue')))
                        
                        # Coeficientes
                        st.subheader("Coeficientes")
                        coefs = res.get('coefficients', [])
                        if coefs:
                            cdf = pd.DataFrame(coefs)
                            # Convertir a numeric antes de formatear
                            for col_f in ["coef", "std_err", "p_value"]:
                                if col_f in cdf.columns:
                                    cdf[col_f] = pd.to_numeric(cdf[col_f], errors='coerce')
                                    
                            st.dataframe(cdf.style.format({
                                "coef": "{:.3f}", "std_err": "{:.3f}", "p_value": "{:.4f}"
                            }, na_rep=""))
                            
                        # Residuos Check rápido
                        if 'residuals' in res:
                            resids = res['residuals']
                            check = verify_regression_assumptions(resids, res.get('fitted', []))
                            with st.expander("Diagnóstico de Residuos (Normalidad)"):
                                st.write(f"Test Shapiro-Wilk: p={format_p_value(check.get('normality_p'))}")
                                if check.get('residuals_normally_distributed'):
                                    st.success("Residuos normales (OK).")
                                else:
                                    st.warning("Posible violación de normalidad en residuos.")

                # LOGISTICA
                else:
                    st.info("Usando Regresión Logística.")
                    X = df[independent_vars]
                    
                    # 1. Primero creamos las Dummies (esto puede generar nombres numéricos)
                    if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
                        st.info("ℹ️ Aplicando codificación automática a variables categóricas...")
                        X = pd.get_dummies(X, drop_first=True)

                    # 2. INMEDIATAMENTE DESPUÉS, convertimos TODOS los nombres de columnas a string
                    # (Esta línea es la que soluciona el error, debe ir DESPUÉS del get_dummies)
                    X.columns = X.columns.astype(str)

                    # 3. Aseguramos que los datos sean numéricos
                    X = X.astype(float)

                    # Ensure Y is binary numeric?
                    y = pd.to_numeric(df[dependent_var], errors='coerce')
                    
                    res = logistic_regression_binary(X, y)
                    
                    if "error" in res:
                         st.error(res['error'])
                         return
                         
                    st.metric("Pseudo R²", f"{res.get('pseudo_r2',0):.3f}")
                    st.metric("LLR p-valor", format_p_value(res.get('llr_pvalue')))
                    
                    st.subheader("Coeficientes (Odds Ratios)")
                    coefs = res.get('coefficients', [])
                    if coefs:
                        cdf = pd.DataFrame(coefs)
                        # Ensure numeric
                        numeric_targets = ['odds_ratio', 'p_value', 'or_ci_lower', 'or_ci_upper']
                        for c in numeric_targets:
                            if c in cdf.columns:
                                cdf[c] = pd.to_numeric(cdf[c], errors='coerce')
                                
                        st.dataframe(cdf[['variable', 'odds_ratio', 'p_value', 'or_ci_lower', 'or_ci_upper']].style.format({
                            "odds_ratio": "{:.3f}", "p_value": "{:.4f}"
                        }, na_rep=""))
                        
            except Exception as e:
                st.error(f"Error en ajuste: {str(e)}")
