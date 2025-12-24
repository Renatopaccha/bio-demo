import sys
import os
import importlib

print("🔍 INICIANDO DIAGNÓSTICO DE BIOSTAT EASY (ACTUALIZADO v2)...\n")

def check_module(module_path, functions_to_check):
    print(f"📂 Revisando módulo: {module_path}...")
    try:
        # Importación dinámica
        mod = importlib.import_module(module_path)
        print(f"   ✅ Archivo encontrado.")
        
        all_good = True
        for func in functions_to_check:
            if hasattr(mod, func):
                print(f"      OK: Función '{func}' encontrada.")
            else:
                print(f"      ❌ ALERTA: La función '{func}' NO está en este archivo.")
                all_good = False
        
        if all_good:
            print("   ✨ Módulo en perfecto estado.\n")
            
    except ImportError as e:
        print(f"   ❌ ERROR CRÍTICO: No existe el archivo o carpeta. Detalle: {e}\n")
    except Exception as e:
        print(f"   ❌ ERROR DESCONOCIDO: {e}\n")

# --- LISTA DE CHEQUEO CORREGIDA (Nombres Modernos) ---

# 1. Limpieza
check_module("modules.cleaning", ["render_limpieza"])

# 2. Gráficos
check_module("modules.plots", ["render_graficos"])

# 3. Reportes
check_module("modules.reports", ["render_reportes", "render_proyectos"])

# 4. Estadísticas (La parte más importante)
# Nota: Buscamos dentro de 'modules.stats' que usa el __init__.py para exponer las funciones
stats_funcs = [
    "render_descriptiva",
    "render_inferencia",      # CORREGIDO: Antes era render_pruebas_hipotesis
    "render_regression",      # CORREGIDO: Antes era render_modelos
    "render_multivariado",    # CORREGIDO: Antes era render_avanzado
    "render_survival",        # CORREGIDO: Antes era render_survival_analysis
    "render_psicometria",
    "render_tabla1",
    "render_asociaciones",
    "render_concordancia",
    "render_diagnostico",
    "render_calculadora"
]
check_module("modules.stats", stats_funcs)

# 5. Utilidades
check_module("modules.utils", ["cargar_estilo_medico"])

print("Diagnóstico finalizado. Si todo está en verde, ejecuta: streamlit run app.py")
