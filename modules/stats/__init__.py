# modules/stats/__init__.py

# 1. IMPORTACIONES DIRECTAS
# (Asumiendo que tus archivos se llaman como dijiste en tu estructura: inferencia.py, multivariado.py, etc.)

from .descriptiva import render_descriptiva
from .inferencia import render_inferencia      # Antes hipotesis
from .modelos import render_regression         # Antes modelos
from .multivariado import render_multivariado  # Antes avanzado
from .supervivencia import render_survival     # Antes supervivencia
from .psicometria import render_psicometria
from .tabla1 import render_tabla1
from .asociaciones import render_asociaciones
from .concordancia import render_concordancia
from .diagnostico import render_diagnostico
from .calculadora import render_calculadora

# 2. DEFINICIÓN DE LA INTERFAZ PÚBLICA
__all__ = [
    "render_descriptiva",
    "render_inferencia",
    "render_regression",
    "render_multivariado",
    "render_survival",
    "render_psicometria",
    "render_tabla1",
    "render_asociaciones",
    "render_concordancia",
    "render_diagnostico",
    "render_calculadora"
]
