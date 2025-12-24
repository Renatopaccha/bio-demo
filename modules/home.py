"""
Módulo de página de inicio para BioStat Easy (Biometric)
Landing Page moderna con componentes visuales
"""

import streamlit as st
from modules.ui import hero_section, features_section, comparison_section, onboarding_section


def render_home():
    """
    Renderiza la landing page completa con diseño moderno.
    
    Incluye:
    - Hero section con título principal
    - Call to Action centrado y prominente
    - Grid de características
    - Comparación de flujos de trabajo
    - Pasos de onboarding
    """
    
    # Espaciado superior
    st.write("")
    
    # --- LOGO CENTRADO ---
    col_izq, col_centro, col_der = st.columns([1, 2, 1])
    with col_centro:
        st.image("assets/logo.png", use_container_width=True)
    
    st.write("")  # Espacio entre logo y título
    
    # 1. Hero Section (HTML)
    st.markdown(hero_section(), unsafe_allow_html=True)
    
    # 2. Call to Action (Botón Nativo Centrado)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        # Botón grande y prominente
        if st.button("Comenzar Ahora ➔", type="primary", use_container_width=True):
            st.session_state['menu_option'] = "Limpieza de Datos"
            st.rerun()
    
    # Espaciadores
    st.write("")
    st.write("")
    
    # 3. Propuesta de Valor (Grid de Features)
    st.markdown(features_section(), unsafe_allow_html=True)
    
    # 4. Comparación (Por qué Biometric)
    st.markdown(comparison_section(), unsafe_allow_html=True)
    
    # 5. Onboarding (Comienza en 3 Pasos)
    st.markdown(onboarding_section(), unsafe_allow_html=True)
    
    # Footer con información adicional
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h4 style="color: #2E86C1; margin-bottom: 0.5rem;">🔬 Para Investigadores</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Enfocado en ciencias de la salud y epidemiología
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h4 style="color: #2E86C1; margin-bottom: 0.5rem;">💻 Open Source</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Código abierto, transparente y auditable
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h4 style="color: #2E86C1; margin-bottom: 0.5rem;">🚀 En Desarrollo</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Nuevas funcionalidades cada semana
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mensaje final motivacional
    st.write("")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #E8F4F8 0%, #FFFFFF 100%); border-radius: 12px; margin-top: 2rem;">
        <p style="font-size: 1.1rem; color: #2C3E50; font-weight: 500; margin: 0;">
            "La estadística no tiene que ser complicada. Con las herramientas correctas, puede ser poderosa y accesible."
        </p>
        <p style="font-size: 0.9rem; color: #5D6D7E; margin-top: 0.5rem;">
            — Equipo BioStat Easy
        </p>
    </div>
    """, unsafe_allow_html=True)
