"""
Persistencia de chats del Asistente IA (multiusuario con cookies).
Soporta PostgreSQL (vía DATABASE_URL en secrets) o SQLite local.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine, Column, String, Integer, Text, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import con fallback robusto
try:
    from streamlit_extras.cookie_manager import CookieManager
except ImportError:
    try:
        from extra_streamlit_components import CookieManager
    except ImportError:
        # Si no existe ninguna librería de cookies, usar fallback con session_state
        CookieManager = None

Base = declarative_base()


# ==========================================
# MODELOS DE BD
# ==========================================

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True)
    created_at = Column(String(50))


class Chat(Base):
    __tablename__ = "chats"
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)
    title = Column(String(255), default="Nuevo chat")
    created_at = Column(String(50))
    updated_at = Column(String(50))


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(String(36), nullable=False)
    role = Column(String(20), nullable=False)  # "user" o "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(String(50))


# ==========================================
# CONEXIÓN A BD (ROBUSTA CON FALLBACK)
# ==========================================

def _sqlite_url():
    """Retorna URL de SQLite local."""
    Path("data").mkdir(exist_ok=True)
    return "sqlite:///data/chats.db"


def _get_database_url():
    """Obtiene DATABASE_URL de secrets o env vars."""
    try:
        url = st.secrets.get("DATABASE_URL")
        if url:
            return url
    except Exception:
        pass

    return os.getenv("DATABASE_URL")


def _build_engine():
    """
    Construye engine de SQLAlchemy con configuración robusta.
    - Postgres: con pool_pre_ping, pool_recycle, sslmode
    - SQLite: con check_same_thread=False
    """
    db_url = _get_database_url()

    # Si no hay DATABASE_URL, usar SQLite
    if not db_url:
        return create_engine(
            _sqlite_url(),
            connect_args={"check_same_thread": False},
            echo=False
        )

    # Normalizar postgres:// a postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    # Connect args para Postgres
    connect_args = {}
    if "sslmode=" not in db_url and db_url.startswith("postgresql://"):
        connect_args["sslmode"] = "require"

    return create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args=connect_args,
        echo=False
    )


def _get_cached_engine():
    """Retorna el engine cacheado en session_state."""
    if "_ai_engine" not in st.session_state:
        st.session_state["_ai_engine"] = _build_engine()
    return st.session_state["_ai_engine"]


def init_db():
    """
    Inicializa tablas en BD.
    Si falla Postgres, hace fallback automático a SQLite local.
    """
    engine = _get_cached_engine()

    try:
        Base.metadata.create_all(engine)
        # Detectar tipo de BD
        db_mode = "postgres" if "postgresql://" in str(engine.url) else "sqlite"
        st.session_state["_ai_db_mode"] = db_mode

    except OperationalError as e:
        # Fallback seguro a SQLite si Postgres falla
        st.warning(
            "⚠️ No se pudo conectar a PostgreSQL. Usando SQLite local como fallback.\n"
            "Verifica DATABASE_URL en Secrets y que el servidor Postgres esté disponible."
        )

        # Crear nuevo engine con SQLite
        engine = create_engine(
            _sqlite_url(),
            connect_args={"check_same_thread": False},
            echo=False
        )
        st.session_state["_ai_engine"] = engine
        Base.metadata.create_all(engine)
        st.session_state["_ai_db_mode"] = "sqlite"


def _get_session():
    """Retorna sesión de SQLAlchemy usando el engine cacheado."""
    engine = _get_cached_engine()
    Session = sessionmaker(bind=engine)
    return Session()


# ==========================================
# GESTIÓN DE USUARIO (COOKIE)
# ==========================================

def get_user_id() -> str:
    """
    Obtiene o crea un user_id único usando cookies.
    Fallback a session_state si no hay librería de cookies.
    """
    # Si CookieManager no está disponible, usar session_state como fallback
    if CookieManager is None:
        if "biometric_uid" not in st.session_state:
            uid = uuid.uuid4().hex
            st.session_state["biometric_uid"] = uid
            # Crear usuario en BD
            _create_user_if_not_exists(uid)
        return st.session_state["biometric_uid"]

    # Usar CookieManager
    if "cookie_manager" not in st.session_state:
        st.session_state["cookie_manager"] = CookieManager()

    cookies = st.session_state["cookie_manager"]
    uid = cookies.get("biometric_uid")

    if not uid:
        # Crear nueva cookie
        uid = uuid.uuid4().hex
        cookies.set("biometric_uid", uid, max_age=60 * 60 * 24 * 365)  # 1 año

    # Asegurar que el usuario existe en BD
    _create_user_if_not_exists(uid)
    return uid


def _create_user_if_not_exists(user_id: str):
    """Crea usuario en BD si no existe."""
    session = _get_session()
    try:
        existing = session.query(User).filter_by(id=user_id).first()
        if not existing:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user = User(id=user_id, created_at=now)
            session.add(user)
            session.commit()
    finally:
        session.close()


# ==========================================
# OPERACIONES DE CHATS
# ==========================================

def list_chats(user_id: str) -> list[dict]:
    """
    Retorna lista de chats del usuario ordenados por updated_at descendente.
    Cada chat: {"id": str, "title": str, "created_at": str, "updated_at": str}
    """
    session = _get_session()
    try:
        chats = (
            session.query(Chat)
            .filter_by(user_id=user_id)
            .order_by(Chat.updated_at.desc())
            .all()
        )
        return [
            {
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
            }
            for c in chats
        ]
    finally:
        session.close()


def create_chat(user_id: str, title: str = "Nuevo chat") -> str:
    """Crea un nuevo chat y retorna su ID."""
    session = _get_session()
    try:
        chat_id = uuid.uuid4().hex[:10]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat = Chat(
            id=chat_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )
        session.add(chat)
        session.commit()
        return chat_id
    finally:
        session.close()


def rename_chat(chat_id: str, new_title: str):
    """Renombra un chat existente."""
    session = _get_session()
    try:
        chat = session.query(Chat).filter_by(id=chat_id).first()
        if chat:
            chat.title = new_title
            chat.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session.commit()
    finally:
        session.close()


def delete_chat(chat_id: str):
    """Elimina un chat y todos sus mensajes."""
    session = _get_session()
    try:
        # Borrar mensajes
        session.query(Message).filter_by(chat_id=chat_id).delete()
        # Borrar chat
        session.query(Chat).filter_by(id=chat_id).delete()
        session.commit()
    finally:
        session.close()


def update_chat_activity(chat_id: str):
    """Actualiza el timestamp updated_at del chat."""
    session = _get_session()
    try:
        chat = session.query(Chat).filter_by(id=chat_id).first()
        if chat:
            chat.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session.commit()
    finally:
        session.close()


# ==========================================
# OPERACIONES DE MENSAJES
# ==========================================

def load_messages(chat_id: str) -> list[dict]:
    """
    Retorna mensajes de un chat ordenados por created_at ascendente.
    Cada mensaje: {"role": str, "content": str}
    """
    session = _get_session()
    try:
        messages = (
            session.query(Message)
            .filter_by(chat_id=chat_id)
            .order_by(Message.id.asc())
            .all()
        )
        return [{"role": m.role, "content": m.content} for m in messages]
    finally:
        session.close()


def append_message(chat_id: str, role: str, content: str):
    """Agrega un mensaje a un chat y actualiza su actividad."""
    session = _get_session()
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = Message(chat_id=chat_id, role=role, content=content, created_at=now)
        session.add(msg)
        session.commit()
        # Actualizar actividad del chat
        update_chat_activity(chat_id)
    finally:
        session.close()


def clear_chat_messages(chat_id: str):
    """Borra todos los mensajes de un chat (pero mantiene el chat)."""
    session = _get_session()
    try:
        session.query(Message).filter_by(chat_id=chat_id).delete()
        session.commit()
        update_chat_activity(chat_id)
    finally:
        session.close()
