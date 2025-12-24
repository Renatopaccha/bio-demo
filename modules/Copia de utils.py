import streamlit as st
import os
import io
import requests
import matplotlib.pyplot as plt
from pathlib import Path

# --- FUNCIÓN ROBUSTA DE CARGA DE CSS ---
def load_custom_css():
    """
    Busca e inyecta el archivo style.css.
    Prioriza la búsqueda dentro de la carpeta 'assets'.
    """
    # Definimos dónde buscar el CSS.
    # Usamos os.path.join para asegurar compatibilidad entre Windows/Mac/Linux
    ruta_assets = os.path.join("assets", "style.css")
    ruta_raiz = "style.css"
    
    css_content = None
    
    # 1. Intentar buscar en la carpeta assets (Ubicación preferida)
    if os.path.exists(ruta_assets):
        try:
            with open(ruta_assets, "r", encoding="utf-8") as f:
                css_content = f.read()
            # st.toast("✅ Estilos cargados desde assets/style.css", icon="🎨")  # Descomentar para debug
        except Exception as e:
            print(f"Error leyendo CSS de assets: {e}")
    
    # 2. Si no estaba en assets, intentar en la raíz (Fallback)
    elif os.path.exists(ruta_raiz):
        try:
            with open(ruta_raiz, "r", encoding="utf-8") as f:
                css_content = f.read()
        except Exception as e:
            print(f"Error leyendo CSS de raíz: {e}")
    
    # 3. Si encontramos contenido, inyectarlo
    if css_content:
        # Esta es la línea mágica que aplica el diseño
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    else:
        # Si falla todo, un aviso discreto en consola (no en la UI)
        print("⚠️ ALERTA: No se pudo encontrar el archivo style.css ni en 'assets/' ni en la raíz.")

# --- CONSTANTES DE GESTIÓN DE MEMORIA ---
MAX_REPORT_ITEMS = 50

# --- FUNCIONES AUXILIARES GLOBALES (REPORTES) ---
def boton_guardar_grafico(fig, titulo_grafico, key_unica):
    """
    Guarda un gráfico en el reporte, verificando límites de memoria.
    """
    if st.button(f"➕ Añadir Gráfico al Reporte", key=key_unica):
        # Verificación de Límite de Memoria
        if len(st.session_state.get('reporte_items', [])) < MAX_REPORT_ITEMS:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            
            if 'reporte_items' not in st.session_state:
                st.session_state['reporte_items'] = []
                
            st.session_state['reporte_items'].append({'tipo': 'img', 'titulo': titulo_grafico, 'data': buf})
            st.toast(f"Gráfico '{titulo_grafico}' guardado!", icon="✅")
        else:
            st.warning(f"⚠️ El reporte está lleno ({MAX_REPORT_ITEMS} ítems). Por favor, descarga o vacía el reporte antes de añadir más.")

def boton_guardar_tabla(df, titulo_tabla, key_unica):
    """
    Guarda una tabla en el reporte, verificando límites de memoria.
    """
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button(f"➕ Añadir Tabla", key=key_unica):
            # Verificación de Límite de Memoria
            if len(st.session_state.get('reporte_items', [])) < MAX_REPORT_ITEMS:
                if 'reporte_items' not in st.session_state:
                    st.session_state['reporte_items'] = []
                    
                st.session_state['reporte_items'].append({'tipo': 'df', 'titulo': titulo_tabla, 'data': df})
                st.toast(f"Tabla '{titulo_tabla}' guardada!", icon="✅")
            else:
                st.warning(f"⚠️ El reporte está lleno ({MAX_REPORT_ITEMS} ítems). Por favor, descarga o vacía el reporte antes de añadir más.")
                
    with c2:
        # Opción de descarga individual inmediata
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(f"⬇️ Descargar CSV", csv, f"{titulo_tabla}.csv", "text/csv", key=f"dl_{key_unica}")

@st.cache_data
def load_lottieurl(url: str):
    """
    Carga una animación Lottie desde una URL.
    """
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

def cargar_estilo_medico():
    """
    Aplica estilos CSS personalizados para una apariencia médica/profesional.
    LEGACY: Ahora es un alias de load_custom_css() para mantener compatibilidad.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* 1. FUENTE GLOBAL */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        /* 2. FONDO DE LA APP */
        .stApp {
            background: linear-gradient(120deg, #e0f2fe 0%, #f0f9ff 100%) !important;
        }
        
        /* 3. SISTEMA DE TARJETAS (Card UI) */
        .card {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            border: 1px solid rgba(0,0,0,0.02);
        }
        /* Compatibilidad con código anterior */
        .card-container {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            border: 1px solid rgba(0,0,0,0.02);
        }

        /* 4. BOTONES MODERNOS */
        .stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 50px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3) !important;
        }

        /* 5. INPUTS Y SELECTORES */
        div[data-baseweb="select"] > div {
            background-color: white !important;
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
        }
        .stTextInput input, .stNumberInput input {
            background-color: white !important;
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
        }

        /* 6. OCULTAR ELEMENTOS STREAMLIT */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* header {visibility: hidden;} COMENTADO PARA NO OCULTAR BOTÓN DE SIDEBAR */
        
        /* Asegurar que el botón de toggle SIEMPRE sea visible */
        [data-testid="collapsedControl"] { 
            display: block !important; 
            visibility: visible !important; 
            z-index: 1000000 !important;
            color: #1e293b !important;
        }

        /* Ajuste para que el header no ocupe espacio si se desea, 
           pero permitiendo click en el botón (opcional, ajustado para seguridad) */
        header {
            background: transparent !important;
        }
        
        /* Títulos */
        h1, h2, h3 {
            color: #1e293b !important;
        }
        </style>
    """, unsafe_allow_html=True)
