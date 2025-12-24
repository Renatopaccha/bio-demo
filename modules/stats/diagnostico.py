import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def render_diagnostico():
    """
    Módulo de Pruebas Diagnósticas y Curvas ROC.
    Reescrito para ser independiente y robusto.
    """
    # 1. Obtención de Datos
    if 'df_principal' not in st.session_state:
        st.error("❌ No se encontró el dataset principal.")
        return

    df = st.session_state.df_principal
    
    if df is None or df.empty:
        st.error("❌ El dataset está vacío.")
        return

    st.header("🏥 Validación de Pruebas Diagnósticas (ROC & AUC)")
    st.markdown("Analiza la capacidad discriminativa de una variable continua frente a un 'Gold Standard' binario.")

    # 2. Selectores de Variables
    col1, col2 = st.columns(2)
    
    with col1:
        # Filtrar posibles variables binarias (2 valores únicos o tipo bool/category con 2 niveles)
        # Para ser prácticos, dejamos que el usuario elija y validamos después.
        gold_std = st.selectbox("1. Estándar de Oro (Binaria):", df.columns, key="diag_gold")
    
    with col2:
        # Filtrar numéricas
        numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        test_var = st.selectbox("2. Variable de Prueba (Numérica):", numericas, key="diag_test")

    # 3. Validación y Preparación de Datos
    if gold_std and test_var:
        # Limpieza de NAs local para este análisis
        data = df[[gold_std, test_var]].dropna()
        
        if len(data) == 0:
            st.warning("No hay datos completos para las variables seleccionadas.")
            return

        # Validar Gold Standard Binario
        # --- CORRECCIÓN DE EMERGENCIA (NUCLEAR) ---
        # 1. Si gold_std es lista, forzamos a string tomando el primer elemento
        if isinstance(gold_std, list):
            gold_std = gold_std[0]
        
        # 2. Extraemos los datos
        temp_data = data[gold_std]
        
        # 3. Si sigue siendo DataFrame (pasa a veces con pandas), forzamos a Serie usando iloc
        if isinstance(temp_data, pd.DataFrame):
            unique_vals = temp_data.iloc[:, 0].unique()
        else:
            unique_vals = temp_data.unique()
        # ------------------------------------------
        if len(unique_vals) != 2:
            st.error(f"❌ La variable '{gold_std}' debe ser estrictamente binaria (tiene {len(unique_vals)} valores únicos: {unique_vals}).")
            return
        
        # Mapeo de Positivo
        col_map1, col_map2 = st.columns(2)
        with col_map1:
            st.info(f"Valores detectados en '{gold_std}': {unique_vals}")
        with col_map2:
            # Convertir a string para el selectbox para evitar errores de tipo
            val_pos = st.selectbox("¿Cuál valor indica 'Enfermedad/Evento' (Positivo)?", list(unique_vals))
        
        if st.button("🚀 Calcular Curva ROC"):
            try:
                # Binarizar Gold Standard (1 = Positivo, 0 = Negativo)
                y_true = (data[gold_std] == val_pos).astype(int)
                y_score = data[test_var]
                                
                # 4. Cálculos ROC y AUC
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)
                
                # 5. Índice de Youden (J = Sensibility + Specificity - 1)
                # Specificity = 1 - FPR
                # J = TPR + (1 - FPR) - 1 = TPR - FPR
                youden_j = tpr - fpr
                best_idx = np.argmax(youden_j)
                best_cut = thresholds[best_idx]
                best_tpr = tpr[best_idx] # Sensibilidad
                best_fpr = fpr[best_idx] # 1 - Especificidad
                
                # 6. Gráfico ROC (Plotly)
                fig = go.Figure()
                
                # Curva ROC
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC (AUC = {auc_score:.3f})',
                    line=dict(color='#2E86C1', width=3)
                ))
                
                # Línea Diagonal (Azar)
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Azar (AUC = 0.5)',
                    line=dict(color='gray', dash='dash')
                ))
                
                # Punto de Corte Óptimo
                fig.add_trace(go.Scatter(
                    x=[best_fpr], y=[best_tpr],
                    mode='markers',
                    name=f'Corte Óptimo ({best_cut:.2f})',
                    marker=dict(color='red', size=12, symbol='star')
                ))

                fig.update_layout(
                    title=f"Curva ROC: {test_var} vs {gold_std}",
                    xaxis_title="1 - Especificidad (Falsos Positivos)",
                    yaxis_title="Sensibilidad (Verdaderos Positivos)",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    legend=dict(x=0.02, y=0.02),
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 7. Tabla de Métricas al Corte Óptimo
                st.subheader("📊 Rendimiento Diagnóstico (En Corte Óptimo)")
                
                # Calcular métricas exactas usando el corte
                # Predicciones binarias basadas en el corte
                # Si la correlación es positiva (mayor valor = más probabilidad), y_pred = score >= create
                # roc_curve maneja esto, pero para la matriz de confusión necesitamos ser explicito.
                # Asumimos dirección estándar.
                
                y_pred = (y_score >= best_cut).astype(int)
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
                especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                met1, met2, met3, met4, met5 = st.columns(5)
                met1.metric("AUC Global", f"{auc_score:.3f}")
                met2.metric("Sensibilidad", f"{sensibilidad:.1%}")
                met3.metric("Especificidad", f"{especificidad:.1%}")
                met4.metric("VPP (+)", f"{ppv:.1%}")
                met5.metric("VPN (-)", f"{npv:.1%}")
                
                st.info(f"**Punto de Corte Óptimo sugerido:** `{best_cut:.4f}` (Maximiza Youden).")
                
                # Matriz de Confusión Simple
                st.markdown("###### Matriz de Confusión al Corte")
                df_cm = pd.DataFrame(
                    [[tp, fp], [fn, tn]],
                    columns=["Pred +", "Pred -"],
                    index=["Real +", "Real -"]
                )
                st.dataframe(df_cm)

            except Exception as e:
                st.error(f"Error en el cálculo: {str(e)}")
