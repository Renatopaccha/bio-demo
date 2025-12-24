import streamlit as st
import os
import io
import pandas as pd
import requests

# ==========================================
# 1. GESTIÓN DE DISEÑO (CSS ROBUSTO)
# ==========================================
def load_custom_css():
    """Carga el estilo CSS usando rutas absolutas."""
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_css = os.path.join(directorio_actual, '..', 'assets', 'style.css')
    ruta_css = os.path.abspath(ruta_css)
    
    if os.path.exists(ruta_css):
        with open(ruta_css, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback silencioso o log
        print(f"Advertencia: No se encontró CSS en {ruta_css}")

# Alias
cargar_estilo_medico = load_custom_css

# --- HELPER DE DISEÑO: TARJETAS ---
def card_container(titulo=None):
    """Contenedor visual tipo tarjeta."""
    if titulo:
        st.markdown(f"""<h3 style="margin-bottom: 1rem;">{titulo}</h3>""", unsafe_allow_html=True)
    return st.container()


# ==========================================
# 2. MOTOR DE REPORTES EXCEL (PREMIUM)
# ==========================================
MAX_REPORT_ITEMS = 50

def agregar_al_reporte(tipo, titulo, data):
    """Añade items a la memoria de la sesión (Carrito)."""
    if 'reporte_items' not in st.session_state:
        st.session_state['reporte_items'] = []
    
    if len(st.session_state['reporte_items']) < MAX_REPORT_ITEMS:
        st.session_state['reporte_items'].append({'tipo': tipo, 'titulo': titulo, 'data': data})
        st.toast(f"✅ '{titulo}' añadido al reporte.", icon="📋")
    else:
        st.warning("⚠️ El reporte está lleno. Descárgalo para limpiar.")

def boton_guardar_grafico(fig, titulo_grafico, key_unica):
    """Botón para guardar gráficos (Matplotlib/Seaborn) en el reporte."""
    if st.button(f"➕ Añadir Gráfica", key=key_unica, help="Agrega esta imagen al Excel final"):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        buf.seek(0)
        agregar_al_reporte('img', titulo_grafico, buf)

def boton_guardar_tabla(df, titulo_tabla, key_unica):
    """Botón dual: Añadir al reporte O Descargar Excel individualmente ahora."""
    c1, c2 = st.columns([1, 1])
    
    # Botón 1: Añadir al carrito
    with c1:
        if st.button(f"➕ Añadir al Reporte", key=key_unica):
            agregar_al_reporte('df', titulo_tabla, df)
            
    # Botón 2: Descarga Inmediata (Excel Nativo)
    with c2:
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Datos')
                # Autoajuste de columnas
                worksheet = writer.sheets['Datos']
                for i, col in enumerate(df.columns):
                    worksheet.set_column(i, i, 20)
                    
            st.download_button(
                label="⬇️ Descargar Excel",
                data=output.getvalue(),
                file_name=f"{titulo_tabla}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{key_unica}"
            )
        except Exception as e:
            st.error(f"Error generando Excel: {e}")


# ==========================================
# 3. UTILIDADES ADICIONALES
# ==========================================
@st.cache_data
def load_lottieurl(url: str):
    """Carga animaciones Lottie desde URL."""
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None
