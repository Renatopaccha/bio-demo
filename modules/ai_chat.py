import streamlit as st
import google.generativeai as genai
from modules.ai_logic import generar_resumen_tecnico

def render_asistente_completo():
    """
    Renderiza la interfaz de chat con IA en página completa.
    """
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Encabezado con Créditos
    c1, c2 = st.columns([3, 1])
    with c1:
        st.header("🤖 Asistente de Investigación BioStat")
        st.caption("Tu experto metodológico personal. Pregunta sobre tus datos o estadística.")
    with c2:
        st.markdown(
            """
            <div style="text-align: right; opacity: 0.7;">
                <small>Powered by</small><br>
                <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Google_Gemini_logo.svg" width="80">
            </div>
            """, unsafe_allow_html=True
        )
    
    # Configuración de API Key (Si no está en session_state)
    if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
        st.warning("🔒 Para comenzar, necesitamos configurar tu acceso.")
        api_key = st.text_input("Ingresa tu Google Gemini API Key:", type="password")
        if api_key:
            st.session_state['gemini_api_key'] = api_key
            st.success("¡Conectado! Ya puedes chatear.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Contenedor de Chat
    st.markdown("---")
    
    # Inicializar historial
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input de Usuario
    if prompt := st.chat_input("Escribe tu consulta aquí (ej: ¿Qué prueba uso para comparar estos grupos?)..."):
        # 1. Mostrar mensaje usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generar Contexto
        contexto_datos = "No hay un dataset cargado actualmente."
        if st.session_state.df_principal is not None:
            contexto_datos = generar_resumen_tecnico(st.session_state.df_principal)

        # 3. Llamada a la IA
        try:
            genai.configure(api_key=st.session_state['gemini_api_key'])
            # Usar la versión 'latest' para asegurar compatibilidad
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            system_prompt = f"""
            Eres un Asistente Senior en Bioestadística para Tesis Médicas.
            
            DATOS DEL USUARIO (CONTEXTO REAL):
            {contexto_datos}
            
            INSTRUCCIONES:
            - Responde de forma didáctica, empática y rigurosa.
            - Basa tus sugerencias en las variables disponibles en el CONTEXTO.
            - Si el usuario pregunta "qué prueba usar", analiza los tipos de variables (numérica vs categórica).
            - Usa formato Markdown para tablas o negritas.
            """
            
            full_prompt = f"{system_prompt}\n\nPregunta del usuario: {prompt}"
            
            with st.chat_message("assistant"):
                with st.spinner("Analizando tus datos y metodología..."):
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                    
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"Error de conexión con Gemini: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
