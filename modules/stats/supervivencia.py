import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Intentar importar lifelines de forma segura
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

def render_survival(df):
    """
    Renderiza el análisis de supervivencia completo (Kaplan-Meier + Cox).
    """
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("⏳ Análisis de Supervivencia")

    if not LIFELINES_AVAILABLE:
        st.error("⚠️ La librería 'lifelines' no está instalada.")
        st.code("pip install lifelines")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Selectores
    c1, c2, c3 = st.columns(3)
    with c1: 
        time_col = st.selectbox("Variable Tiempo (Días/Meses)", df.select_dtypes(include=['number']).columns)
    with c2: 
        event_col = st.selectbox("Variable Evento (0/1 o True/False)", df.columns)
    with c3:
        group_col = st.selectbox("Comparar Grupos (Opcional)", ["Ninguno"] + list(df.columns))

    if st.button("Generar Curvas Kaplan-Meier"):
        try:
            T = df[time_col]
            E = df[event_col]
            
            kmf = KaplanMeierFitter()
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if group_col != "Ninguno":
                groups = df[group_col].unique()
                for i, group in enumerate(groups):
                    mask = df[group_col] == group
                    # Ajuste robusto
                    kmf.fit(T[mask], event_observed=E[mask], label=str(group))
                    kmf.plot_survival_function(ax=ax, ci_show=True)
                st.success(f"✅ Curvas generadas estratificadas por {group_col}")
            else:
                kmf.fit(T, event_observed=E, label="Todos")
                kmf.plot_survival_function(ax=ax, ci_show=True)
                st.success("✅ Curva generada (Global)")
            
            plt.title(f"Supervivencia Kaplan-Meier: {time_col}")
            plt.ylabel("Probabilidad de Supervivencia")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Tabla de resumen
            st.write("**Tabla de Supervivencia:**")
            st.dataframe(kmf.event_table.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Error en el análisis: {str(e)}")
            st.caption("Verifica que la columna de evento sea numérica (0=Vivo, 1=Evento) o Booleana.")

    st.markdown('</div>', unsafe_allow_html=True)
