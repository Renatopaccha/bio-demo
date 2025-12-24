"""
Módulo de componentes HTML para la Landing Page de BioStat Easy
Incluye Hero Section, Features, Comparison y Onboarding
"""

import streamlit as st


def hero_section():
    """Sección Hero principal de la landing page"""
    return """\u003cdiv class="hero-container">
\u003ch1 class="hero-title">Biometric\u003c/h1>
\u003cp class="hero-subtitle">
Precisión Clínica. Simplicidad Digital.
\u003cbr>
Diseñado para investigadores que valoran su tiempo.
\u003c/p>
\u003c/div>"""


def features_section():
    """Grid de características principales"""
    return """\u003cdiv class="feature-grid">
\u003cdiv class="feature-card">
\u003cdiv class="feature-icon">📊\u003c/div>
\u003ch3 class="feature-title">Estadística Avanzada\u003c/h3>
\u003cp class="feature-description">
Desde descriptivos hasta regresión multivariada. Todo lo que necesitas en un solo lugar.
\u003c/p>
\u003c/div>

\u003cdiv class="feature-card">
\u003cdiv class="feature-icon">🧹\u003c/div>
\u003ch3 class="feature-title">Limpieza Inteligente\u003c/h3>
\u003cp class="feature-description">
Procesa datos de KoboToolbox, detecta valores atípicos y prepara tus datos en minutos.
\u003c/p>
\u003c/div>

\u003cdiv class="feature-card">
\u003cdiv class="feature-icon">📈\u003c/div>
\u003ch3 class="feature-title">Visualización Premium\u003c/h3>
\u003cp class="feature-description">
Gráficos interactivos de calidad publicable. Exporta en alta resolución.
\u003c/p>
\u003c/div>

\u003cdiv class="feature-card">
\u003cdiv class="feature-icon">🤖\u003c/div>
\u003ch3 class="feature-title">IA Integrada\u003c/h3>
\u003cp class="feature-description">
Asistente inteligente que interpreta resultados y responde dudas estadísticas.
\u003c/p>
\u003c/div>

\u003cdiv class="feature-card">
\u003cdiv class="feature-icon">⚡\u003c/div>
\u003ch3 class="feature-title">Rapidez Extrema\u003c/h3>
\u003cp class="feature-description">
Realiza análisis que tomarían horas en SPSS, en cuestión de minutos.
\u003c/p>
\u003c/div>

\u003cdiv class="feature-card">
\u003cdiv class="feature-icon">📄\u003c/div>
\u003ch3 class="feature-title">APA 7 Automático\u003c/h3>
\u003cp class="feature-description">
Resultados formateados en APA 7. Copia y pega directo a tu paper.
\u003c/p>
\u003c/div>
\u003c/div>"""


def comparison_section():
    """Sección de comparación de flujos de trabajo"""
    return """\u003cdiv class="comparison-box">
\u003ch2 class="comparison-title">¿Por qué Biometric?\u003c/h2>
\u003cp class="comparison-subtitle">
Hemos simplificado el flujo de trabajo estadístico para que te concentres en lo importante: tus hallazgos.
\u003c/p>

\u003cdiv style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; max-width: 800px; margin-top: 2rem;">
\u003cdiv style="background: white; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #E74C3C;">
\u003ch4 style="color: #E74C3C; margin-bottom: 1rem;">❌ Método Tradicional\u003c/h4>
\u003cul style="color: #5D6D7E; font-size: 0.9rem; line-height: 1.8;">
\u003cli>Excel → SPSS → GraphPad\u003c/li>
\u003cli>Curva de aprendizaje empinada\u003c/li>
\u003cli>Licencias costosas\u003c/li>
\u003cli>Formatos incompatibles\u003c/li>
\u003cli>Errores de copy-paste\u003c/li>
\u003c/ul>
\u003c/div>

\u003cdiv style="background: white; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #27AE60;">
\u003ch4 style="color: #27AE60; margin-bottom: 1rem;">✅ Con Biometric\u003c/h4>
\u003cul style="color: #5D6D7E; font-size: 0.9rem; line-height: 1.8;">
\u003cli>Todo en una interfaz web\u003c/li>
\u003cli>Intuitivo desde el minuto 1\u003c/li>
\u003cli>100% gratuito y open-source\u003c/li>
\u003cli>Exporta en cualquier formato\u003c/li>
\u003cli>Reproducible y auditable\u003c/li>
\u003c/ul>
\u003c/div>
\u003c/div>
\u003c/div>"""


def onboarding_section():
    """Sección de pasos de inicio"""
    return """\u003cdiv style="text-align: center; margin-bottom: 3rem;">
\u003ch2 style="font-size: 2rem; font-weight: 700; color: #2C3E50; margin-bottom: 1rem;">
Comienza en 3 Pasos
\u003c/h2>
\u003cp style="font-size: 1.1rem; color: #5D6D7E; margin-bottom: 2rem;">
Tu primer análisis en menos de 5 minutos
\u003c/p>

\u003cdiv class="steps-container">
\u003cdiv class="step-item">
\u003cdiv class="step-number">1\u003c/div>
\u003ch4 class="step-title">Carga tus datos\u003c/h4>
\u003cp class="step-description">
Arrastra tu Excel o CSV. Compatible con KoboToolbox.
\u003c/p>
\u003c/div>

\u003cdiv class="step-item">
\u003cdiv class="step-number">2\u003c/div>
\u003ch4 class="step-title">Explora y limpia\u003c/h4>
\u003cp class="step-description">
Detecta valores atípicos, transforma variables, filtra datos.
\u003c/p>
\u003c/div>

\u003cdiv class="step-item">
\u003cdiv class="step-number">3\u003c/div>
\u003ch4 class="step-title">Analiza y exporta\u003c/h4>
\u003cp class="step-description">
Elige tu prueba estadística y descarga resultados en APA.
\u003c/p>
\u003c/div>
\u003c/div>
\u003c/div>"""


# ============================================================
# COMPONENTES ANTERIORES (Métricas, Encabezados, Badges)
# Mantenemos las funciones originales para compatibilidad
# ============================================================

def tarjeta_metrica(titulo: str, valor: str, icono: str = "fa-chart-line", color: str = "#2E86C1"):
    """
    Renderiza una tarjeta métrica personalizada con diseño médico/clean.
    
    Args:
        titulo: Título de la métrica (ej: "Total Pacientes")
        valor: Valor a mostrar (ej: "1,250" o "45.6%")
        icono: Clase de ícono FontAwesome (ej: "fa-users", "fa-heartbeat")
        color: Color de acento en formato hex (por defecto azul médico)
    """
    html = f"""
    \u003clink rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    \u003cstyle>
        .metrica-card {{
            background: #FFFFFF;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border-left: 4px solid {color};
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .metrica-card:hover {{
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
            transform: translateY(-2px);
        }}
        
        .metrica-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }}
        
        .metrica-icon {{
            width: 48px;
            height: 48px;
            border-radius: 10px;
            background: linear-gradient(135deg, {color}15, {color}25);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: {color};
        }}
        
        .metrica-titulo {{
            font-family: 'Inter', 'Roboto', -apple-system, sans-serif;
            font-size: 14px;
            font-weight: 500;
            color: #5D6D7E;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 0;
        }}
        
        .metrica-valor {{
            font-family: 'Inter', 'Roboto', -apple-system, sans-serif;
            font-size: 32px;
            font-weight: 700;
            color: #2C3E50;
            margin: 0;
            line-height: 1.2;
        }}
    \u003c/style>
    
    \u003cdiv class="metrica-card">
        \u003cdiv class="metrica-header">
            \u003cdiv class="metrica-icon">
                \u003ci class="fas {icono}">\u003c/i>
            \u003c/div>
            \u003cp class="metrica-titulo">{titulo}\u003c/p>
        \u003c/div>
        \u003ch2 class="metrica-valor">{valor}\u003c/h2>
    \u003c/div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def encabezado_seccion(titulo: str, subtitulo: str = ""):
    """
    Renderiza un encabezado de sección moderno y elegante.
    
    Args:
        titulo: Título principal de la sección
        subtitulo: Subtítulo o descripción opcional
    """
    subtitulo_html = f'\u003cp class="encabezado-subtitulo">{subtitulo}\u003c/p>' if subtitulo else ''
    
    html = f"""
    \u003cstyle>
        .encabezado-container {{
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 2px solid #E8F4F8;
        }}
        
        .encabezado-titulo {{
            font-family: 'Inter', 'Roboto', -apple-system, sans-serif;
            font-size: 28px;
            font-weight: 700;
            color: #2E86C1;
            margin: 0 0 8px 0;
            line-height: 1.3;
            letter-spacing: -0.5px;
        }}
        
        .encabezado-subtitulo {{
            font-family: 'Inter', 'Roboto', -apple-system, sans-serif;
            font-size: 15px;
            font-weight: 400;
            color: #5D6D7E;
            margin: 0;
            line-height: 1.5;
        }}
    \u003c/style>
    
    \u003cdiv class="encabezado-container">
        \u003ch1 class="encabezado-titulo">{titulo}\u003c/h1>
        {subtitulo_html}
    \u003c/div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def badge_resultado(texto: str, estado: str = "neutro"):
    """
    Renderiza una etiqueta tipo píldora para mostrar resultados.
    
    Args:
        texto: Texto a mostrar en el badge (ej: "p < 0.05", "Significativo")
        estado: Tipo de badge - "exito", "alerta", o "neutro"
    """
    # Configuración de colores según el estado
    colores = {
        "exito": {
            "bg": "#D5F4E6",
            "text": "#0B5345",
            "border": "#A9DFBF"
        },
        "alerta": {
            "bg": "#FADBD8",
            "text": "#78281F",
            "border": "#F1948A"
        },
        "neutro": {
            "bg": "#E8F4F8",
            "text": "#1B4F72",
            "border": "#AED6F1"
        }
    }
    
    # Usar colores neutros si el estado no es reconocido
    color_config = colores.get(estado, colores["neutro"])
    
    html = f"""
    \u003cstyle>
        .badge-resultado {{
            display: inline-block;
            font-family: 'Inter', 'Roboto', -apple-system, sans-serif;
            font-size: 13px;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 20px;
            background-color: {color_config['bg']};
            color: {color_config['text']};
            border: 1px solid {color_config['border']};
            letter-spacing: 0.3px;
            white-space: nowrap;
            transition: all 0.2s ease;
        }}
        
        .badge-resultado:hover {{
            transform: scale(1.05);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }}
    \u003c/style>
    
    \u003cspan class="badge-resultado">{texto}\u003c/span>
    """
    
    st.markdown(html, unsafe_allow_html=True)
