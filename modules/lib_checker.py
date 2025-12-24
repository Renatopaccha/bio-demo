import sys
import importlib
import logging
import time
from functools import wraps
import streamlit as st

logger = logging.getLogger(__name__)

class LibraryChecker:
    """
    Clase robusta para verificar dependencias críticas en tiempo de ejecución.
    Implementa caché, reintentos y logging silencioso.
    """
    
    def __init__(self):
        """Inicializa el verificador y el caché."""
        self.required_libraries = [
            'sklearn',      # scikit-learn
            'statsmodels',  # statsmodels
            'pandas',       # pandas
            'numpy',        # numpy
            'scipy',        # scipy
            'plotly'        # plotly
        ]
        
        # Inicializar caché en memoria de la instancia
        self._local_cache = {}
        
        # Inicializar caché en session_state si no existe
        if 'lib_checker_cache' not in st.session_state:
            st.session_state.lib_checker_cache = {}

    def check_library(self, lib_name, max_retries=3, retry_delay=0.5):
        """
        Verifica si una librería está instalada, con reintentos para robustez.
        
        Args:
            lib_name (str): Nombre del módulo a importar
            max_retries (int): Número máximo de intentos
            retry_delay (float): Segundos de espera entre intentos
            
        Returns:
            bool: True si está disponible, False si falla
        """
        # 1. Verificar caché local (más rápido)
        if lib_name in self._local_cache:
            return self._local_cache[lib_name]
            
        # 2. Verificar caché persistente de Streamlit
        if lib_name in st.session_state.lib_checker_cache:
            self._local_cache[lib_name] = st.session_state.lib_checker_cache[lib_name]
            return st.session_state.lib_checker_cache[lib_name]

        # 3. Intento de importación con reintentos
        for attempt in range(max_retries):
            try:
                importlib.import_module(lib_name)
                # Éxito
                self._update_cache(lib_name, True)
                return True
                
            except ImportError:
                # Fallo, reintentar si no es el último intento
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    # Fallo definitivo
                    logger.warning(f"Librería no encontrada tras {max_retries} intentos: {lib_name}")
                    self._update_cache(lib_name, False)
                    return False
                    
            except Exception as e:
                # Otro tipo de error
                logger.error(f"Error inesperado verificando {lib_name}: {str(e)}")
                self._update_cache(lib_name, False)
                return False

    def _update_cache(self, lib_name, status):
        """Actualiza ambos cachés con el resultado."""
        self._local_cache[lib_name] = status
        st.session_state.lib_checker_cache[lib_name] = status

    def get_status(self):
        """
        Verifica el estado de TODAS las librerías críticas.
        
        Returns:
            dict: Diccionario {nombre_lib: bool}
        """
        status = {}
        for lib in self.required_libraries:
            status[lib] = self.check_library(lib)
        return status

    def is_available(self, lib_name):
        """
        Verifica disponibilidad de una librería específica (wrapper simple).
        
        Args:
            lib_name (str): Nombre de la librería
            
        Returns:
            bool: True/False
        """
        return self.check_library(lib_name)

    def require_library(self, lib_name, function_name=""):
        """
        Decorador para requerir una librería antes de ejecutar una función.
        Si falla, loguea error y retorna None, evitando crash.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.check_library(lib_name):
                    return func(*args, **kwargs)
                else:
                    func_id = function_name if function_name else func.__name__
                    msg = f"Dependencia faltante ('{lib_name}') impide ejecutar: {func_id}"
                    logger.error(msg)
                    # Opcional: st.error(msg) pero el requisito dice no usar mensajes UI
                    return None
            return wrapper
        return decorator

    def get_missing_libraries(self):
        """
        Identifica cuáles librerías críticas faltan.
        
        Returns:
            list: Lista de nombres de librerías faltantes.
        """
        status = self.get_status()
        missing = [lib for lib, available in status.items() if not available]
        return missing

    def validate_all(self):
        """
        Confirmación booleana global de que TODO está listo.
        
        Returns:
            bool: True si todas las librerías están, False si falta alguna.
        """
        missing = self.get_missing_libraries()
        return len(missing) == 0

    def clear_cache(self):
        """Limpia los cachés para forzar reverificación."""
        self._local_cache = {}
        if 'lib_checker_cache' in st.session_state:
            del st.session_state.lib_checker_cache
            st.session_state.lib_checker_cache = {}

# Instancia global para importar
checker = LibraryChecker()

# Funciones auxiliares expuestas a nivel de módulo
def validate_dependencies():
    """Valida todas las dependencias críticas."""
    return checker.validate_all()

def get_library_status():
    """Obtiene diccionario de estado de librerías."""
    return checker.get_status()
