import streamlit as st
import importlib
from streamlit_option_menu import option_menu

# --- CONFIGURACIÓN UI (Debe ser lo primero) ---
st.set_page_config(
    page_title="Biometric - Tesis Simplificada",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNCIÓN HELPER DE EJECUCIÓN DINÁMICA ---
def ejecutar_modulo(modulo):
    """
    Busca inteligentemente el punto de entrada de un módulo y lo ejecuta.
    Orden de búsqueda: render(), main(), app(), o cualquier función render_*.
    """
    if modulo is None:
        st.error("Error Interno: El módulo es None (falló la importación).")
        return

    # 1. Búsquedas Estándar
    if hasattr(modulo, 'render'):
        modulo.render()
        return
    if hasattr(modulo, 'main'):
        modulo.main()
        return
    if hasattr(modulo, 'app'):
        modulo.app()
        return

    # 2. Búsqueda Inteligente de 'render_*'
    for attr in dir(modulo):
        if attr.startswith("render_") and callable(getattr(modulo, attr)):
            func = getattr(modulo, attr)
            # Verificamos si acepta argumento 'df' introspectivamente o probamos
            # Hack simple: la mayoría de nuestros módulos reciben (df) o nada.
            # Como no podemos inspeccionar fácil la firma sin inspect module,
            # usaremos un try-catch en la ejecución.
            try:
                if 'df_principal' in st.session_state:
                    try:
                        func(st.session_state.df_principal)
                    except TypeError:
                        # Si falla por argumentos, probamos sin argumentos
                        func()
                else:
                    func()
            except Exception as e:
                st.error(f"Error ejecutando {attr} en {modulo.__name__}: {e}")
            return

    st.error(f"No se encontró función de inicio (render_*, main, app) en el módulo {modulo.__name__}")

# --- IMPORTADOR ROBUSTO ---
def safe_import(module_name_stats, module_name_root=None):
    """
    Intenta importar módulos desde múltiples ubicaciones (robustez para Cloud).
    Orden: modules.stats → modules → raíz del proyecto
    """
    candidates = []

    # 1. Construir lista de candidatos
    candidates.append(f"modules.stats.{module_name_stats}")

    if module_name_root:
        candidates.append(f"modules.{module_name_root}")
        candidates.append(module_name_root)  # Fallback a raíz
    else:
        candidates.append(module_name_stats)  # Si no hay root, probar stats en raíz

    # 2. Intentar importar en orden
    for path in candidates:
        try:
            return importlib.import_module(path)
        except ImportError as e:
            # Si el error NO es "No module named", es un error de código (CRÍTICO)
            if "No module named" not in str(e):
                st.error(f"🚨 Error de Código en '{path}': {e}")
                return None
            # Si es "No module named", continuar probando otros candidatos
            continue
        except Exception as e:
            # Otro tipo de error (sintaxis, etc.)
            st.error(f"🚨 Error cargando '{path}': {e}")
            return None

    # 3. Si ninguno funcionó, retornar None
    return None

# --- CARGA DE MÓDULOS ---
# Priorizamos Español y Rutas Stats
mod_home        = safe_import("home", "home")              # modules/home.py (NUEVO)
mod_ai_chat     = safe_import("ai_chat", "ai_chat")        # modules/ai_chat.py
mod_limpieza    = safe_import("cleaning", "cleaning")      # modules/cleaning.py
mod_explorador  = safe_import("explorador")                # modules/stats/explorador.py
mod_descriptiva = safe_import("descriptiva")               # modules/stats/descriptiva.py
mod_tabla1      = safe_import("tabla1")                    # modules/stats/tabla1.py
mod_inferencia  = safe_import("inferencia", "inference")   # Prioriza español
mod_modelos     = safe_import("modelos")                   # modules/stats/modelos.py
mod_multivariado= safe_import("multivariado", "multivariate")
mod_survival    = safe_import("supervivencia", "survival")
mod_psicometria = safe_import("psicometria", "psychometrics")
mod_asociaciones= safe_import("asociaciones")
mod_concordancia= safe_import("concordancia")
mod_diagnostico = safe_import("diagnostico")               # modules/stats/diagnostico.py
mod_graficos    = safe_import("plots", "plots")            # modules/plots.py
mod_utils       = safe_import("utils", "utils")
mod_ajuste_tasas= safe_import("ajuste_tasas")
mod_reporte     = safe_import("reporte", "reporte")        # modules/reporte.py

def main():
    # --- CARGA DE DISEÑO (MODO DEBUG) ---
    try:
        # Intentamos importar directamente para ver si el archivo existe y es válido
        from modules import utils as mod_utils
        
        # Intentamos ejecutar la carga de CSS
        print("Intentando cargar CSS...")
        mod_utils.load_custom_css()
        
        # Si llega aquí, funcionó
        st.toast("✅ Diseño cargado correctamente", icon="🎨")
        
    except ImportError as e:
        st.error(f"❌ ERROR CRÍTICO: No se pudo importar 'modules.utils'. \nDetalle: {e}")
        st.info("Verifica que exista el archivo 'modules/utils.py' y que tenga un '__init__.py' válido en la carpeta modules.")
        mod_utils = None
        
    except Exception as e:
        st.error(f"❌ ERROR EJECUTANDO CSS: {e}")
        mod_utils = None

    # Estado Inicial Data
    if 'df_principal' not in st.session_state:
        st.session_state.df_principal = None

    # --- SIDEBAR ---
    with st.sidebar:
        # --- ENCABEZADO DE MARCA (Nativo + CSS) ---
        # Usamos columnas para alinear imagen y texto de forma segura
        col_logo, col_texto = st.columns([0.35, 0.65])
        
        with col_logo:
            # La imagen carga nativamente. Si tiene fondo blanco, se verá como un icono cuadrado.
            st.image("assets/logo.png", use_container_width=True)
            
        with col_texto:
            # Estilizamos el texto para que coincida con el diseño deseado
            st.markdown("""
                <div style="
                    font-size: 1.6rem;
                    font-weight: 700;
                    color: #111827;
                    font-family: 'Outfit', sans-serif;
                    margin-top: 10px;
                    letter-spacing: -0.5px;
                ">
                    Biometric
                </div>
            """, unsafe_allow_html=True)
        
        # Divisor sutil antes del menú
        st.markdown('<hr style="margin: 1rem 0; border-color: rgba(255,255,255,0.2);">', unsafe_allow_html=True)
        st.write("")
        
        # Definir opciones e iconos del menú
        opciones_menu = [
            "Inicio",
            "Mi Reporte",
            "Asistente IA",
            "Limpieza de Datos",
            "Modo Explorador",
            "Estadística Descriptiva",
            "Ajuste de Tasas",
            "Tabla 1 (Paper)",
            "Pruebas de Hipótesis",
            "Modelos de Regresión",
            "Análisis Multivariado",
            "Análisis de Supervivencia",
            "Psicometría",
            "Asociaciones",
            "Concordancia",
            "Diagnóstico (ROC)",
            "Suite Gráfica"
        ]
        
        iconos_menu = [
            "house",
            "box-seam",
            "robot",
            "brush",
            "search",
            "clipboard-data",
            "scales",
            "file-earmark-text",
            "ui-checks",
            "graph-up",
            "diagram-3",
            "activity",
            "person-bounding-box",
            "link",
            "check2-circle",
            "bullseye",
            "palette"
        ]
        
        # --- LÓGICA DE NAVEGACIÓN (CORREGIDA) ---
        
        # 1. Función callback (Acepta la key que envía la librería)
        def on_change_callback(key):
            # Actualizamos el estado usando la key recibida
            st.session_state['menu_option'] = st.session_state[key]
        
        # 2. Inicializar estado si no existe (página por defecto: Inicio)
        if 'menu_option' not in st.session_state:
            st.session_state['menu_option'] = "Inicio"
        
        # 3. Calcular índice visual correcto
        try:
            ind_actual = opciones_menu.index(st.session_state.get('menu_option', 'Inicio'))
        except ValueError:
            ind_actual = 0
        
        # 4. Renderizar Menú
        option_menu(
            menu_title=None,
            options=opciones_menu,
            icons=iconos_menu,
            menu_icon="cast",
            default_index=ind_actual,
            orientation="vertical",
            key='nav_key',                 # Llave interna
            on_change=on_change_callback   # Ejecuta la función sin pasar argumentos extra
        )

        st.divider()

        # Chat Sidebar Persistente (Si existe el módulo)
        # DESHABILITADO: El asistente IA solo se muestra en su página dedicada
        # if mod_ai_chat and hasattr(mod_ai_chat, 'render_ia_sidebar'):
        #     mod_ai_chat.render_ia_sidebar()

    # --- ENRUTADOR ---
    
    # 1. Página de Inicio (NO requiere datos)
    if st.session_state['menu_option'] == "Inicio":
        ejecutar_modulo(mod_home)
        return  # Fin
    
    # 2. Mi Reporte (NO requiere datos)
    elif st.session_state['menu_option'] == "Mi Reporte":
        ejecutar_modulo(mod_reporte)
        return  # Fin
    
    # 2. Módulos que NO requieren Datos cargados
    elif st.session_state['menu_option'] == "Asistente IA":
        if mod_ai_chat and hasattr(mod_ai_chat, "render_asistente_completo"):
            mod_ai_chat.render_asistente_completo()
        else:
            st.error("No se encontró el módulo de IA o falta render_asistente_completo() en ai_chat.py.")
        return
        
    elif st.session_state['menu_option'] == "Limpieza de Datos":
        ejecutar_modulo(mod_limpieza)
        return # Fin

    elif st.session_state['menu_option'] == "Ajuste de Tasas":
        ejecutar_modulo(mod_ajuste_tasas)
        return # Fin

    # 2. Validación Global de Datos
    if st.session_state.df_principal is None:
        st.warning("⚠️ No hay datos en memoria.")
        st.subheader("Por favor, inicia cargando tu Excel en 'Limpieza de Datos'")
        if st.button("Ir a Limpieza"):
            ejecutar_modulo(mod_limpieza)
        return

    # 3. Módulos que REQUIEREN Datos
    if st.session_state['menu_option'] == "Modo Explorador":
        ejecutar_modulo(mod_explorador)
        
    elif st.session_state['menu_option'] == "Estadística Descriptiva":
        ejecutar_modulo(mod_descriptiva)
        
    elif st.session_state['menu_option'] == "Tabla 1 (Paper)":
        ejecutar_modulo(mod_tabla1)
        
    elif st.session_state['menu_option'] == "Pruebas de Hipótesis":
        ejecutar_modulo(mod_inferencia)
        
    elif st.session_state['menu_option'] == "Modelos de Regresión":
        # Caso especial: Queremos Tabs para Ajuste y Diagnóstico
        # Podríamos hacerlo manual aquí o dejar que mod_modelos maneje todo si lo tuviera.
        # Por robustez, hacemos lo manual aquí usando las funciones directas si existen.
        st.subheader("📉 Regresión y Diagnósticos Avanzados")
        tab1, tab2 = st.tabs(["Ajuste de Modelo", "Diagnósticos (Linearidad/VIF)"])
        
        with tab1:
            ejecutar_modulo(mod_modelos)
            
        with tab2:
            # Aquí llamamos directamente a diagnostico.py pero específicamente a las funciones avanzadas si no tiene un render principal adecuado
            if mod_diagnostico:
                # Intentamos buscar render_diagnostico que sabemos que existe
                if hasattr(mod_diagnostico, 'render_diagnostico'):
                    mod_diagnostico.render_diagnostico()
                else:
                    ejecutar_modulo(mod_diagnostico)
            else:
                st.error("Módulo Diagnóstico no cargado.")
                
    elif st.session_state['menu_option'] == "Análisis Multivariado":
        ejecutar_modulo(mod_multivariado)
        
    elif st.session_state['menu_option'] == "Análisis de Supervivencia":
        ejecutar_modulo(mod_survival)
        
    elif st.session_state['menu_option'] == "Psicometría":
        ejecutar_modulo(mod_psicometria)
        
    elif st.session_state['menu_option'] == "Asociaciones":
        ejecutar_modulo(mod_asociaciones)
        
    elif st.session_state['menu_option'] == "Concordancia":
        ejecutar_modulo(mod_concordancia)
        
    elif st.session_state['menu_option'] == "Diagnóstico (ROC)":
        # Si diagnostico.py se usó para VIF/OLS, quizas ROC no esté ahí.
        # Fallback genérico
        ejecutar_modulo(mod_diagnostico)
        
    elif st.session_state['menu_option'] == "Suite Gráfica":
        ejecutar_modulo(mod_graficos)

if __name__ == "__main__":
    main()
