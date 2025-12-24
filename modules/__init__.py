"""
BioStat Easy - Bioestadística para Investigación Médica

Módulos disponibles:
- types: Definiciones de tipos y alias para análisis estadístico.
- stats: Suite completa de funciones estadísticas (Descriptiva, Pruebas, Modelos, Supervivencia, Psicometría).
- utils: Utilidades generales (carga de CSS, reportes, animaciones).
"""

from . import types
from . import stats
from . import utils

__version__ = "1.0.0"
__all__ = ["types", "stats", "utils"]
