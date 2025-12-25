"""
Módulo de página de inicio para Biometric
Landing Page moderna con diseño premium health-tech
"""

import streamlit as st
from modules.ui import hero_section, features_section, comparison_section, onboarding_section


def render_home():
    """
    Renderiza la landing page completa con diseño premium SaaS 2025.
    
    Incluye:
    - Logo centrado
    - Hero section con glassmorphism y mini-KPIs
    - CTAs principales (Comenzar + Ver Ejemplo)
    - Grid de características con hover effects
    - Comparación tradicional vs Biometric
    - Onboarding en 3 pasos tipo stepper
    """
    
    # --- LOGO CENTRADO ---
    st.write("")
    col_izq, col_centro, col_der = st.columns([1, 1.5, 1])
    with col_centro:
        st.image("assets/logo.png", use_container_width=True)
    
    st.write("")
    
    # --- HERO SECTION (HTML con glassmorphism) ---
    st.markdown(hero_section(app_name="Biometric"), unsafe_allow_html=True)
    
    # --- CALL TO ACTIONS (Botones Nativos Estilizados) ---
    st.markdown('<div class="bm-cta-wrap">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Comenzar Ahora", type="primary", use_container_width=True, key="cta_main"):
            st.session_state['menu_option'] = "Limpieza de Datos"
            st.rerun()
    
    with col2:
        if st.button("📊 Ver Ejemplo (Descriptiva)", use_container_width=True, key="cta_demo"):
            st.session_state['menu_option'] = "Estadística Descriptiva"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Espaciador
    st.write("")
    st.write("")
    
    # --- FEATURES SECTION ---
    st.markdown(features_section(), unsafe_allow_html=True)
    
    # --- COMPARISON SECTION ---
    st.markdown(comparison_section(), unsafe_allow_html=True)
    
    # --- ONBOARDING SECTION ---
    st.markdown(onboarding_section(), unsafe_allow_html=True)
    
    # --- FOOTER CON HIGHLIGHTS ---
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🔬</div>
            <h4 style="color: #0B3A82; margin-bottom: 0.8rem; font-weight: 700;">Para Investigadores</h4>
            <p style="color: #6B7280; font-size: 0.95rem; line-height: 1.6;">
                Enfocado en ciencias de la salud, epidemiología y ensayos clínicos
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">💻</div>
            <h4 style="color: #0B3A82; margin-bottom: 0.8rem; font-weight: 700;">Open Source</h4>
            <p style="color: #6B7280; font-size: 0.95rem; line-height: 1.6;">
                Código abierto, transparente y completamente auditable
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🚀</div>
            <h4 style="color: #0B3A82; margin-bottom: 0.8rem; font-weight: 700;">En Desarrollo</h4>
            <p style="color: #6B7280; font-size: 0.95rem; line-height: 1.6;">
                Nuevas funcionalidades y mejoras cada semana
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- MENSAJE FINAL ---
    st.write("")
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 2.5rem; 
        background: linear-gradient(135deg, #F0F9FF 0%, #FFFFFF 100%); 
        border-radius: 20px; 
        margin-top: 3rem;
        border: 1px solid #E0F2FE;
    ">
        <p style="
            font-size: 1.2rem; 
            color: #0B3A82; 
            font-weight: 600; 
            margin: 0 0 0.8rem 0;
            line-height: 1.6;
        ">
            "La estadística no tiene que ser intimidante.<br>
            Con las herramientas correctas, puede ser poderosa y accesible."
        </p>
        <p style="font-size: 0.95rem; color: #6B7280; margin: 0;">
            — Equipo Biometric
        </p>
    </div>
    """, unsafe_allow_html=True)


# Alias para compatibilidad con diferentes convenciones de nombres
def render():
    """Alias para render_home()"""
    render_home()


def main():
    """Alias para render_home()"""
    render_home()
