import sys
import subprocess
import pkg_resources

def install(package):
    print(f"🔧 Instalando {package} en: {sys.executable}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required = {'plotly', 'statsmodels', 'scipy', 'pandas', 'streamlit-option-menu'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"⚠️ Faltan: {missing}")
    for lib in missing:
        install(lib)
    print("✅ Todo instalado correctamente.")
else:
    print("✅ Todas las librerías ya estaban instaladas en este entorno.")
