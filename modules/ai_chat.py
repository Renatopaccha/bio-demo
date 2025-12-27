import os
import streamlit as st
import google.generativeai as genai
from datetime import datetime
import uuid

# Import robusto con fallback para diferentes estructuras de proyecto
try:
    from modules.ai_logic import generar_resumen_tecnico, configurar_gemini, build_result_prompt, df_to_compact_markdown
except ImportError:
    try:
        from ai_logic import generar_resumen_tecnico, configurar_gemini, build_result_prompt, df_to_compact_markdown
    except ImportError as e:
        # Si falla todo, mostrar error claro
        import streamlit as st
        st.error(f"❌ No se pudo importar ai_logic.py: {e}")
        st.stop()

# Import de persistencia (BD)
try:
    from modules.ai_store import (
        get_user_id,
        init_db,
        list_chats,
        create_chat,
        rename_chat,
        delete_chat as db_delete_chat,
        load_messages,
        append_message,
        update_chat_activity,
        clear_chat_messages,
    )
except ImportError:
    try:
        from ai_store import (
            get_user_id,
            init_db,
            list_chats,
            create_chat,
            rename_chat,
            delete_chat as db_delete_chat,
            load_messages,
            append_message,
            update_chat_activity,
            clear_chat_messages,
        )
    except ImportError as e:
        st.error(f"❌ No se pudo importar ai_store.py: {e}")
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
# SISTEMA MULTI-CHAT CON PERSISTENCIA (helpers)
# ==========================================

def _ensure_active_chat_exists(user_id: str):
    """Asegura que existe al menos un chat y uno está activo."""
    chats = list_chats(user_id)

    if not chats:
        # Crear primer chat
        chat_id = create_chat(user_id, "Nuevo chat")
        st.session_state["ai_active_chat_id"] = chat_id
        return

    # Si no hay chat activo o el activo no existe, seleccionar el más reciente
    active_id = st.session_state.get("ai_active_chat_id")
    chat_ids = [c["id"] for c in chats]

    if not active_id or active_id not in chat_ids:
        st.session_state["ai_active_chat_id"] = chats[0]["id"]


def _handle_delete_chat(user_id: str, chat_id: str):
    """Elimina un chat y ajusta el chat activo."""
    db_delete_chat(chat_id)

    # Si era el activo, seleccionar otro
    if st.session_state.get("ai_active_chat_id") == chat_id:
        chats = list_chats(user_id)
        if chats:
            st.session_state["ai_active_chat_id"] = chats[0]["id"]
        else:
            # Crear nuevo chat si no queda ninguno
            new_chat_id = create_chat(user_id, "Nuevo chat")
            st.session_state["ai_active_chat_id"] = new_chat_id


def _auto_title_from_prompt(prompt: str) -> str:
    """Genera título automático del chat basado en primeras palabras del prompt."""
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



def set_focus_artifact(df_resultado, titulo, notas=""):
    """
    Establece una tabla específica como 'foco' o contexto activo para el chat.
    """
    if df_resultado is None:
        return

    tabla_md = df_to_compact_markdown(df_resultado)
    
    st.session_state["ai_focus_artifact"] = {
        "titulo": titulo,
        "notas": notas,
        "tabla_md": tabla_md,
        "ts": datetime.now().isoformat()
    }

def render_ai_actions_for_result(df_resultado, titulo, notas="", key=None):
    """
    Muestra botones de acción IA para una tabla de resultados:
    1. Interpretar (borrador académico)
    2. Continuar en chat (contexto anclado)
    """
    if df_resultado is None:
        return

    key = key or titulo
    
    # Contenedor de acciones
    col1, col2 = st.columns([1, 1], gap="small")
    
    with col1:
        # Reutilizamos la lógica de interpretación existente pero adaptada
        render_interpretar_tabla(df_resultado, titulo, notas, button_label="📌 Interpretar con IA", key_suffix=key)
        
    with col2:
        if st.button("💬 Continuar en chat", key=f"chat_focus_{key}", use_container_width=True):
            set_focus_artifact(df_resultado, titulo, notas)
            # Redirección manual (dependiendo de cómo manejes la navegación en tu app)
            # Asumimos que hay un 'menu_option' en session_state que controla la vista principal
            st.session_state["menu_option"] = "Asistente IA"
            st.toast(f"✅ Tabla '{titulo}' anclada al chat", icon="📌")
            st.rerun()


def render_asistente_completo():
    """Página completa multi-chat estilo ChatGPT/Gemini con persistencia en BD."""
    # Inicializar BD
    init_db()

    # Obtener ID de usuario (cookie o session_state)
    user_id = get_user_id()

    # Asegurar que existe al menos un chat
    _ensure_active_chat_exists(user_id)

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
            new_chat_id = create_chat(user_id, "Nuevo chat")
            st.session_state["ai_active_chat_id"] = new_chat_id
            st.rerun()

        st.write("")

        # Cargar chats del usuario desde BD (ya vienen ordenados por updated_at desc)
        chats = list_chats(user_id)
        active_id = st.session_state.get("ai_active_chat_id")

        popover_fn = getattr(st, "popover", None)

        for chat_data in chats:
            cid = chat_data["id"]
            is_active = (cid == active_id)
            row = st.columns([0.82, 0.18], gap="small")

            # Botón para abrir chat
            with row[0]:
                label = chat_data.get("title", "Nuevo chat")
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
                        new_name = st.text_input("Renombrar", value=chat_data.get("title", "Nuevo chat"), key=f"rename_{cid}")
                        if st.button("Guardar nombre", key=f"save_name_{cid}", use_container_width=True):
                            rename_chat(cid, new_name.strip() or "Nuevo chat")
                            st.rerun()

                        st.divider()
                        confirm = st.checkbox("Confirmar eliminación", key=f"confirm_del_{cid}")
                        if st.button("🗑️ Eliminar chat", key=f"del_{cid}", disabled=not confirm, use_container_width=True):
                            _handle_delete_chat(user_id, cid)
                            st.rerun()
                else:
                    # Fallback si tu Streamlit no soporta popover
                    with st.expander("⋯", expanded=False):
                        st.caption(f"ID: {cid}")
                        new_name = st.text_input("Renombrar", value=chat_data.get("title", "Nuevo chat"), key=f"rename_{cid}")
                        if st.button("Guardar nombre", key=f"save_name_{cid}", use_container_width=True):
                            rename_chat(cid, new_name.strip() or "Nuevo chat")
                            st.rerun()

                        st.divider()
                        confirm = st.checkbox("Confirmar eliminación", key=f"confirm_del_{cid}")
                        if st.button("🗑️ Eliminar chat", key=f"del_{cid}", disabled=not confirm, use_container_width=True):
                            _handle_delete_chat(user_id, cid)
                            st.rerun()

    # =========================
    # PANEL DERECHO: CHAT ACTIVO
    # =========================
    with right:
        chat_id = st.session_state["ai_active_chat_id"]

        st.markdown('<div class="card">', unsafe_allow_html=True)

        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.header("🤖 Asistente de Investigación BioStat")
            st.caption("Tu experto metodológico personal. Multi-chat con historial persistente.")
        with header_cols[1]:
            if st.button("🧹 Limpiar chat", key="clear_active_chat", use_container_width=True):
                clear_chat_messages(chat_id)
                st.rerun()

        st.markdown("---")

        # --- MOSTRAR ARTEFACTO EN FOCO (SI EXISTE) ---
        focus_artifact = st.session_state.get("ai_focus_artifact")
        if focus_artifact:
            with st.container():
                st.info(f"📌 **Tabla activa:** {focus_artifact['titulo']}")
                
                with st.expander("Ver tabla anclada (contexto)", expanded=False):
                    st.markdown(focus_artifact.get("notas", ""))
                    st.markdown(focus_artifact.get("tabla_md", ""))
                
                if st.button("Quitar tabla activa", key="remove_focus_artifact", use_container_width=True):
                    del st.session_state["ai_focus_artifact"]
                    st.rerun()
                st.divider()

        # Cargar mensajes del chat activo desde BD
        messages = load_messages(chat_id)
        for m in messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        prompt = st.chat_input("Escribe tu pregunta…")
        if prompt:
            # Guardar mensaje usuario en BD
            append_message(chat_id, "user", prompt)

            # Auto título si es primer mensaje real
            current_chats = list_chats(user_id)
            current_chat = next((c for c in current_chats if c["id"] == chat_id), None)
            if current_chat and current_chat["title"] == "Nuevo chat":
                auto_title = _auto_title_from_prompt(prompt)
                rename_chat(chat_id, auto_title)

            with st.chat_message("user"):
                st.markdown(prompt)

            # Contexto dataset (sin datos sensibles)
            df = st.session_state.get("df_principal")
            contexto_datos = "No hay un dataset cargado actualmente."
            if df is not None:
                contexto_datos = generar_resumen_tecnico(df)

            # Incluir contexto del artefacto si existe
            contexto_artefacto = ""
            if focus_artifact:
                contexto_artefacto = f"""
TEN EN CUENTA ESTA TABLA DE RESULTADOS QUE EL USUARIO ESTÁ VIENDO AHORA MISMO:
TÍTULO: {focus_artifact['titulo']}
NOTAS: {focus_artifact.get('notas', '')}
TABLA (Markdown):
{focus_artifact.get('tabla_md', '')}
"""

            system_prompt = f"""
Eres un Asistente Senior en Bioestadística para investigación en salud.
No inventes datos ni números. Si falta información, dilo y sugiere qué falta.
Explica de forma entendible para estudiante e investigador (claro y académico).

CONTEXTO GLOBAL DEL DATASET (sin datos sensibles):
{contexto_datos}

{contexto_artefacto}
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

                # Guardar respuesta del asistente en BD
                append_message(chat_id, "assistant", txt)

            except Exception as e:
                err = f"Error de conexión con Gemini: {e}"
                append_message(chat_id, "assistant", err)
                st.error(err)

            # Recargar para mostrar nuevos mensajes
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# COPILOTO CONECTADO A TABLAS
# ==========================================

def render_interpretar_tabla(df_resultado, titulo, notas="", button_label="📌 Interpretar con IA", key_suffix=None):
    """
    Renderiza botón "Interpretar con IA" para tablas de resultados estadísticos.

    Args:
        df_resultado: DataFrame con la tabla de resultados
        titulo: Título descriptivo de la tabla
        notas: Notas adicionales sobre el análisis (opcional)
        button_label: Texto del botón
        key_suffix: Sufijo único para keys
    """
    # Generar key única para esta tabla (basada en título)
    base_key = key_suffix or titulo
    tabla_key = f"ai_interpret_{hash(base_key)}"

    # Verificar si hay API key configurada
    api_key = _get_gemini_key()
    if not api_key:
        st.info("💡 Configura tu API Key de Gemini para usar IA")
        return

    # Botón para interpretar
    if st.button(button_label, key=f"btn_{tabla_key}", use_container_width=True):
        # Obtener dataset principal si existe
        df_principal = st.session_state.get("df_principal")

        # Construir prompt
        prompt = build_result_prompt(df_resultado, titulo, notas, df_principal)

        # Obtener modelo
        model = _get_or_create_model()
        if model is None:
            st.error("❌ No se pudo inicializar el modelo de IA. Verifica tu API Key.")
            return

        # Generar interpretación
        with st.spinner("🤖 Generando interpretación académica..."):
            try:
                response = model.generate_content(prompt)
                interpretacion = (response.text or "").strip()

                if not interpretacion:
                    interpretacion = "⚠️ No se recibió respuesta del modelo. Intenta nuevamente."

                # Guardar en session_state
                st.session_state[tabla_key] = interpretacion

            except Exception as e:
                st.error(f"❌ Error al conectar con Gemini: {e}")
                return

    # Mostrar interpretación si existe
    if tabla_key in st.session_state:
        interpretacion = st.session_state[tabla_key]

        # Usar dialog si existe (Streamlit >= 1.31), si no usar expander
        dialog_fn = getattr(st, "dialog", None)

        if dialog_fn:
            # Modal dialog (más moderno)
            @dialog_fn(f"🎓 Interpretación: {titulo}")
            def mostrar_interpretacion():
                st.markdown(interpretacion)

                # Botón para copiar (muestra en code block para copiar manualmente)
                st.divider()
                st.caption("📋 Copia el texto desde aquí:")
                st.code(interpretacion, language="markdown")

                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("✅ Cerrar", use_container_width=True):
                        st.rerun()
                with col2:
                    if st.button("🗑️ Borrar interpretación", use_container_width=True):
                        del st.session_state[tabla_key]
                        st.rerun()

            if st.button("📖 Ver interpretación", key=f"ver_{tabla_key}", use_container_width=True):
                mostrar_interpretacion()
        else:
            # Expander (fallback para versiones antiguas)
            with st.expander("📖 Ver Interpretación Académica", expanded=True):
                st.markdown(interpretacion)

                st.divider()
                st.caption("📋 Copia el texto desde aquí:")
                st.code(interpretacion, language="markdown")

                if st.button("🗑️ Borrar interpretación", key=f"borrar_{tabla_key}"):
                    del st.session_state[tabla_key]
                    st.rerun()
