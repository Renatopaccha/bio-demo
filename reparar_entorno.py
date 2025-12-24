import streamlit as st
import sys
import subprocess
import importlib

st.set_page_config(page_title="Reparador de Entorno", layout="centered")

st.title("🛠️ Reparador de Entorno BioStat")

st.info(f"📍 **Ubicación exacta de este Python:**\n\n`{sys.executable}`")

def instalar(paquete):
    try:
        st.write(f"⏳ Instalando **{paquete}**...")
        # Este comando usa el MISMO python que está corriendo la app
        subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])
        st.success(f"✅ {paquete} instalado correctamente.")
    except Exception as e:
        st.error(f"❌ Error instalando {paquete}: {e}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Estado Actual")
    
    # Chequeo Sklearn
    try:
        import sklearn
        st.success(f"✅ Scikit-learn: Detectado (v{sklearn.__version__})")
    except ImportError:
        st.error("❌ Scikit-learn: NO DETECTADO")
        
    # Chequeo Statsmodels
    try:
        import statsmodels
        st.success(f"✅ Statsmodels: Detectado (v{statsmodels.__version__})")
    except ImportError:
        st.error("❌ Statsmodels: NO DETECTADO")

with col2:
    st.subheader("Acciones")
    if st.button("🚑 FORZAR INSTALACIÓN EN ESTE ENTORNO", type="primary"):
        with st.spinner("Instalando dependencias... por favor espera..."):
            instalar("scikit-learn")
            instalar("statsmodels")
            instalar("patsy")  # Dependencia clave a veces olvidada
            instalar("scipy")
            st.balloons()
            st.success("🎉 PROCESO TERMINADO. Por favor REINICIA la app original.")

st.markdown("---")
st.caption("Si ves los checks verdes a la izquierda, tu app original YA DEBERÍA funcionar.")
