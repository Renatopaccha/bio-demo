import os
import streamlit as st
import google.generativeai as genai
from datetime import datetime
import uuid

# Import robusto con fallback para diferentes estructuras de proyecto
try:
    from modules.ai_logic import generar_resumen_tecnico, configurar_gemini
except ImportError:
    try:
        from ai_logic import generar_resumen_tecnico, configurar_gemini
    except ImportError as e:
        # Si falla todo, mostrar error claro
        import streamlit as st
        st.error(f"❌ No se pudo importar ai_logic.py: {e}")
        st.stop()

# ==========================================
# CONFIGURACIÓN DE API KEY (Prioridades):
# ==========================================
# 1. Streamlit Secrets (RECOMENDADO para producción)
#    - Cloud: Settings → Secrets en Streamlit Community Cloud
#    - Local: .streamlit/secrets.toml (NO subir a GitHub - agregar a .gitignore)
# 2. Session State (fallback manual del usuario)
# 3. Variables de entorno (GEMINI_API_KEY / GOOGLE_API_KEY)
# ==========================================


def _get_gemini_key() -> str | None:
    # 1) Streamlit Secrets (PROD en Community Cloud)
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # 2) Session state (fallback manual)
    key = st.session_state.get("gemini_api_key")
    if key:
        return key

    # 3) Env vars (otros hostings)
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


# ==========================================
# SISTEMA MULTI-CHAT (helpers)
# ==========================================

def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _init_chat_state():
    if "ai_conversations" not in st.session_state:
        st.session_state["ai_conversations"] = {}

    if "ai_active_chat_id" not in st.session_state or st.session_state["ai_active_chat_id"] not in st.session_state["ai_conversations"]:
        # Crear chat inicial
        _new_chat(set_active=True)


def _new_chat(set_active: bool = True):
    chat_id = uuid.uuid4().hex[:10]
    st.session_state["ai_conversations"][chat_id] = {
        "title": "Nuevo chat",
        "messages": [],
        "created_at": _now_str(),
        "updated_at": _now_str(),
    }
    if set_active:
        st.session_state["ai_active_chat_id"] = chat_id
    return chat_id


def _delete_chat(chat_id: str):
    convos = st.session_state.get("ai_conversations", {})
    if chat_id in convos:
        del convos[chat_id]

    # Si borraste el activo, elegir otro o crear uno nuevo
    if st.session_state.get("ai_active_chat_id") == chat_id:
        remaining = list(convos.keys())
        if remaining:
            st.session_state["ai_active_chat_id"] = remaining[0]
        else:
            _new_chat(set_active=True)


def _get_active_chat():
    _init_chat_state()
    active_id = st.session_state["ai_active_chat_id"]
    return active_id, st.session_state["ai_conversations"][active_id]


def _auto_title_from_prompt(prompt: str) -> str:
    # 6-8 palabras como título, estilo ChatGPT
    words = (prompt or "").strip().split()
    title = " ".join(words[:8]).strip()
    if len(words) > 8:
        title += "…"
    return title or "Nuevo chat"


def render_ia_sidebar():
    """Chat mini persistente en la barra lateral (no rompe la UI principal)."""
    with st.sidebar:
        st.markdown("### 🤖 Asistente IA")
        st.caption("Bioestadística aplicada (salud). Respuestas didácticas y académicas.")

        # API Key UI
        api_key = _get_gemini_key()
        if not api_key:
            st.warning("🔒 No se encontró GEMINI_API_KEY en Streamlit Secrets.")
            st.info("Configura tu key en Settings → Secrets (Cloud) o .streamlit/secrets.toml (Local)")
            k = st.text_input("API Key (fallback)", type="password", key="gemini_api_key_input")
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
    """Página completa multi-chat estilo ChatGPT/Gemini."""
    _init_chat_state()

    # --- Modelo ---
    model = _get_or_create_model()
    if model is None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("🤖 Asistente de Investigación BioStat")
        st.warning("🔒 No se encontró GEMINI_API_KEY en Streamlit Secrets (Settings → Secrets).")
        st.info("Agrega en Secrets: GEMINI_API_KEY = \"TU_KEY\"")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # --- Layout principal: historial (izq) + chat (der) ---
    left, right = st.columns([1, 3], gap="large")

    # =========================
    # PANEL IZQUIERDO: CHATS
    # =========================
    with left:
        st.markdown("### 💬 Chats")
        if st.button("➕ Nuevo chat", key="ai_new_chat_btn", use_container_width=True):
            _new_chat(set_active=True)
            st.rerun()

        st.write("")

        # Ordenar por updated_at (más reciente arriba)
        convos = st.session_state["ai_conversations"]
        def _sort_key(item):
            cid, data = item
            return data.get("updated_at", "")
        ordered = sorted(convos.items(), key=_sort_key, reverse=True)

        active_id = st.session_state["ai_active_chat_id"]

        popover_fn = getattr(st, "popover", None)

        for cid, cdata in ordered:
            is_active = (cid == active_id)
            row = st.columns([0.82, 0.18], gap="small")

            # Botón para abrir chat
            with row[0]:
                label = cdata.get("title", "Nuevo chat")
                if st.button(
                    label,
                    key=f"open_chat_{cid}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True
                ):
                    st.session_state["ai_active_chat_id"] = cid
                    st.rerun()

            # Menú "⋯" (popover si existe, si no expander)
            with row[1]:
                if popover_fn:
                    with st.popover("⋯", use_container_width=True):
                        st.caption(f"ID: {cid}")
                        new_name = st.text_input("Renombrar", value=cdata.get("title", "Nuevo chat"), key=f"rename_{cid}")
                        if st.button("Guardar nombre", key=f"save_name_{cid}", use_container_width=True):
                            st.session_state["ai_conversations"][cid]["title"] = new_name.strip() or "Nuevo chat"
                            st.session_state["ai_conversations"][cid]["updated_at"] = _now_str()
                            st.rerun()

                        st.divider()
                        confirm = st.checkbox("Confirmar eliminación", key=f"confirm_del_{cid}")
                        if st.button("🗑️ Eliminar chat", key=f"del_{cid}", disabled=not confirm, use_container_width=True):
                            _delete_chat(cid)
                            st.rerun()
                else:
                    # Fallback si tu Streamlit no soporta popover
                    with st.expander("⋯", expanded=False):
                        st.caption(f"ID: {cid}")
                        new_name = st.text_input("Renombrar", value=cdata.get("title", "Nuevo chat"), key=f"rename_{cid}")
                        if st.button("Guardar nombre", key=f"save_name_{cid}", use_container_width=True):
                            st.session_state["ai_conversations"][cid]["title"] = new_name.strip() or "Nuevo chat"
                            st.session_state["ai_conversations"][cid]["updated_at"] = _now_str()
                            st.rerun()

                        st.divider()
                        confirm = st.checkbox("Confirmar eliminación", key=f"confirm_del_{cid}")
                        if st.button("🗑️ Eliminar chat", key=f"del_{cid}", disabled=not confirm, use_container_width=True):
                            _delete_chat(cid)
                            st.rerun()

    # =========================
    # PANEL DERECHO: CHAT ACTIVO
    # =========================
    with right:
        chat_id, chat = _get_active_chat()

        st.markdown('<div class="card">', unsafe_allow_html=True)

        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.header("🤖 Asistente de Investigación BioStat")
            st.caption("Tu experto metodológico personal. Multi-chat con historial.")
        with header_cols[1]:
            if st.button("🧹 Limpiar chat", key="clear_active_chat", use_container_width=True):
                st.session_state["ai_conversations"][chat_id]["messages"] = []
                st.session_state["ai_conversations"][chat_id]["updated_at"] = _now_str()
                st.rerun()

        st.markdown("---")

        # Mostrar mensajes del chat activo
        messages = chat.get("messages", [])
        for m in messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        prompt = st.chat_input("Escribe tu pregunta…")
        if prompt:
            # Guardar mensaje usuario
            messages.append({"role": "user", "content": prompt})
            st.session_state["ai_conversations"][chat_id]["updated_at"] = _now_str()

            # Auto título si es primer mensaje real
            if st.session_state["ai_conversations"][chat_id]["title"] == "Nuevo chat":
                st.session_state["ai_conversations"][chat_id]["title"] = _auto_title_from_prompt(prompt)

            with st.chat_message("user"):
                st.markdown(prompt)

            # Contexto dataset (sin datos sensibles)
            df = st.session_state.get("df_principal")
            contexto_datos = "No hay un dataset cargado actualmente."
            if df is not None:
                contexto_datos = generar_resumen_tecnico(df)

            system_prompt = f"""
Eres un Asistente Senior en Bioestadística para investigación en salud.
No inventes datos ni números. Si falta información, dilo y sugiere qué falta.
Explica de forma entendible para estudiante e investigador (claro y académico).

CONTEXTO (sin datos sensibles):
{contexto_datos}
"""

            full_prompt = f"{system_prompt}\n\nPregunta del usuario: {prompt}"

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Analizando..."):
                        response = model.generate_content(full_prompt)
                        txt = (response.text or "").strip()
                        if not txt:
                            txt = "No se recibió texto del modelo. Intenta nuevamente."
                        st.markdown(txt)

                messages.append({"role": "assistant", "content": txt})
                st.session_state["ai_conversations"][chat_id]["messages"] = messages
                st.session_state["ai_conversations"][chat_id]["updated_at"] = _now_str()

            except Exception as e:
                err = f"Error de conexión con Gemini: {e}"
                messages.append({"role": "assistant", "content": err})
                st.session_state["ai_conversations"][chat_id]["messages"] = messages
                st.session_state["ai_conversations"][chat_id]["updated_at"] = _now_str()
                st.error(err)

        st.markdown('</div>', unsafe_allow_html=True)
