import os
import streamlit as st
import google.generativeai as genai

from modules.ai_logic import generar_resumen_tecnico, configurar_gemini


def _get_gemini_key() -> str | None:
    # 1) session_state
    key = st.session_state.get("gemini_api_key")
    if key:
        return key

    # 2) secrets
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # 3) env vars
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _get_or_create_model():
    # Cache del modelo en session_state
    if "gemini_model" in st.session_state and st.session_state["gemini_model"] is not None:
        return st.session_state["gemini_model"]

    api_key = _get_gemini_key()
    if not api_key:
        return None

    model = configurar_gemini(api_key)
    st.session_state["gemini_model"] = model
    return model


def render_ia_sidebar():
    """Chat mini persistente en la barra lateral (no rompe la UI principal)."""
    with st.sidebar:
        st.markdown("### 🤖 Asistente IA")
        st.caption("Bioestadística aplicada (salud). Respuestas didácticas y académicas.")

        # API Key UI
        api_key = _get_gemini_key()
        if not api_key:
            st.warning("🔒 Ingresa tu API Key de Gemini para activar el asistente.")
            k = st.text_input("Gemini API Key", type="password", key="gemini_api_key_input")
            if k:
                st.session_state["gemini_api_key"] = k
                st.session_state.pop("gemini_model", None)
                st.rerun()
            return

        # Model
        model = _get_or_create_model()
        if model is None:
            st.error("No se pudo inicializar Gemini. Verifica tu API Key o el acceso a modelos.")
            if st.button("Reintentar", use_container_width=True):
                st.session_state.pop("gemini_model", None)
                st.rerun()
            return

        # Estado chat
        if "ai_sidebar_messages" not in st.session_state:
            st.session_state["ai_sidebar_messages"] = []

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("🧹 Limpiar chat", use_container_width=True):
                st.session_state["ai_sidebar_messages"] = []
                st.rerun()
        with colB:
            if st.button("🔄 Reiniciar IA", use_container_width=True):
                st.session_state.pop("gemini_model", None)
                st.rerun()

        # Mostrar últimas interacciones (compacto)
        for m in st.session_state["ai_sidebar_messages"][-6:]:
            role = "🧑‍🎓" if m["role"] == "user" else "🤖"
            st.markdown(f"**{role}** {m['content']}")

        prompt = st.text_area("Pregunta", height=80, placeholder="Ej: ¿Qué prueba usar para comparar dos grupos?")
        if st.button("Enviar", use_container_width=True) and prompt.strip():
            # contexto técnico (sin datos sensibles)
            df = st.session_state.get("df_principal")
            contexto = "No hay dataset cargado actualmente."
            if df is not None:
                contexto = generar_resumen_tecnico(df)

            system_prompt = f"""
Eres un Asistente Senior en Bioestadística para investigación en salud.
No inventes datos. Si falta información, pide lo mínimo necesario.
Usa lenguaje claro (para estudiante) pero riguroso (para investigador).

CONTEXTO DE DATOS (sin filas crudas):
{contexto}
"""
            full_prompt = f"{system_prompt}\n\nPregunta del usuario: {prompt}"

            st.session_state["ai_sidebar_messages"].append({"role": "user", "content": prompt})

            try:
                resp = model.generate_content(full_prompt)
                txt = (resp.text or "").strip()
                if not txt:
                    txt = "No se recibió texto del modelo. Intenta nuevamente."
                st.session_state["ai_sidebar_messages"].append({"role": "assistant", "content": txt})
            except Exception as e:
                st.session_state["ai_sidebar_messages"].append(
                    {"role": "assistant", "content": f"Error al conectar con Gemini: {e}"}
                )

            st.rerun()


def render_asistente_completo():
    """Página completa de chat IA (la que ya usas en tu UI)."""
    st.markdown('<div class="card">', unsafe_allow_html=True)

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

    model = _get_or_create_model()
    if model is None:
        st.warning("🔒 Para usar el asistente, ingresa tu API Key de Gemini aquí:")
        k = st.text_input("Gemini API Key", type="password", key="gemini_api_key_page")

        if k:
            st.session_state["gemini_api_key"] = k
            st.session_state.pop("gemini_model", None)  # forzar recreación del modelo
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        return

    # mensajes
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("---")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Escribe tu pregunta...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        df = st.session_state.get("df_principal")
        contexto_datos = "No hay un dataset cargado actualmente."
        if df is not None:
            contexto_datos = generar_resumen_tecnico(df)

        system_prompt = f"""
Eres un Asistente Senior en Bioestadística para Tesis Médicas y salud pública.
No inventes datos ni números. Si falta información, dilo y sugiere qué falta.
Primero clínico, luego estadístico. Explica para estudiante e investigador.

CONTEXTO (sin datos sensibles):
{contexto_datos}
"""
        full_prompt = f"{system_prompt}\n\nPregunta del usuario: {prompt}"

        try:
            with st.chat_message("assistant"):
                with st.spinner("Analizando..."):
                    response = model.generate_content(full_prompt)
                    txt = (response.text or "").strip()
                    st.markdown(txt)
            st.session_state.messages.append({"role": "assistant", "content": txt})
        except Exception as e:
            st.error(f"Error de conexión con Gemini: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
