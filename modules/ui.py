"""
Módulo de componentes HTML para Biometric Landing Page V2
Premium Health-Tech SaaS Design con Glassmorphism
"""

import streamlit as st


def hero_section(app_name="Biometric"):
    """Hero section moderna con glassmorphism y mini-KPIs"""
    return f"""<div class="bm-hero"><div class="bm-hero-content"><div class="bm-hero-pretitle">🔬 Health-Tech Analytics</div><h1>{app_name}</h1><p class="bm-hero-subtitle">Plataforma bioestadística de nueva generación. <br>Transforma datos complejos en insights publicables.<br><strong>Sin código. Sin complicaciones.</strong></p></div><div class="bm-hero-card"><h3>✨ Todo lo que necesitas, integrado</h3><p>Desde limpieza de datos hasta regresión multivariada. Visualizaciones interactivas, interpretación por IA y exportación APA 7.</p><div class="bm-kpis"><div class="bm-kpi-item"><span class="bm-kpi-value">16+</span><span class="bm-kpi-label">Módulos</span></div><div class="bm-kpi-item"><span class="bm-kpi-value">&lt; 5min</span><span class="bm-kpi-label">Setup</span></div><div class="bm-kpi-item"><span class="bm-kpi-value">100%</span><span class="bm-kpi-label">Gratis</span></div></div></div></div>"""


def features_section():
    """Grid de 6 características con hover effects"""
    return """<div class="bm-features"><h2>Diseñado para Investigadores</h2><p class="bm-features-subtitle">Herramientas profesionales sin la complejidad de software tradicional</p><div class="bm-features-grid"><div class="bm-feature-card"><span class="bm-feature-icon">🧹</span><h3>Limpieza Inteligente</h3><p>Procesa KoboToolbox, detecta outliers automáticamente, transforma variables con un clic.</p></div><div class="bm-feature-card"><span class="bm-feature-icon">📊</span><h3>Estadística Completa</h3><p>Descriptivos, hipótesis, regresión, supervivencia, ROC. 20+ pruebas estadísticas implementadas.</p></div><div class="bm-feature-card"><span class="bm-feature-icon">✨</span><h3>IA Integrada</h3><p>Asistente Gemini que interpreta resultados y responde dudas metodológicas en lenguaje natural.</p></div><div class="bm-feature-card"><span class="bm-feature-icon">📈</span><h3>Gráficos Premium</h3><p>Plotly interactivo, exportación HD (PNG/SVG), paletas optimizadas para publicación.</p></div><div class="bm-feature-card"><span class="bm-feature-icon">📄</span><h3>APA 7 Automático</h3><p>Tablas formateadas listas para copiar/pegar. Genera reportes completos en Excel.</p></div><div class="bm-feature-card"><span class="bm-feature-icon">⚡</span><h3>Cloud Ready</h3><p>Funciona en navegador. No instalaciones. Compatible con Windows, Mac, Linux, tablets.</p></div></div></div>"""


def comparison_section():
    """Comparación lado a lado: tradicional vs Biometric"""
    return """<div class="bm-comparison"><h2>¿Por qué elegir Biometric?</h2><p class="bm-comparison-subtitle">Simplificamos radicalmente el proceso de análisis estadístico</p><div class="bm-comparison-grid"><div class="bm-comparison-col bad"><h4>❌ Flujo Tradicional</h4><ul><li>Excel → SPSS → GraphPad → Word</li><li>Licencias costosas (€500-2000/año)</li><li>Curva de aprendizaje empinada</li><li>Incompatibilidad entre formatos</li><li>Copy-paste manual = errores</li><li>No reproducible ni auditable</li></ul></div><div class="bm-comparison-col good"><h4>✅ Con Biometric</h4><ul><li>Todo en un solo lugar, web</li><li>100% gratuito y open-source</li><li>Intuitivo desde el minuto 1</li><li>Exporta en cualquier formato</li><li>Flujo end-to-end sin fricciones</li><li>Historial completo y código a la vista</li></ul></div></div></div>"""


def onboarding_section():
    """3 pasos para comenzar con diseño tipo stepper"""
    return """<div class="bm-onboarding"><h2>Tu Primer Análisis en Minutos</h2><p class="bm-onboarding-subtitle">No necesitas ser experto en software estadístico</p><div class="bm-steps"><div class="bm-step"><div class="bm-step-number">1</div><h4>Carga tus Datos</h4><p>Arrastra tu Excel, CSV o Google Sheets. Compatible con KoboToolbox y REDCap.</p></div><div class="bm-step"><div class="bm-step-number">2</div><h4>Explora y Limpia</h4><p>Visualiza distribuciones, detecta outliers, aplica transformaciones. Todo visual.</p></div><div class="bm-step"><div class="bm-step-number">3</div><h4>Analiza y Exporta</h4><p>Elige tu prueba, revisa resultados, descarga reporte APA 7 automático.</p></div></div></div>"""


# ============================================================
# COMPONENTES LEGACY (conservados para compatibilidad)
# ============================================================

def tarjeta_metrica(titulo: str, valor: str, icono: str = "fa-chart-line", color: str = "#2E86C1"):
    """Tarjeta métrica personalizada (legacy)"""
    html = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
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
    </style>
    
    <div class="metrica-card">
        <div class="metrica-header">
            <div class="metrica-icon">
                <i class="fas {icono}"></i>
            </div>
            <p class="metrica-titulo">{titulo}</p>
        </div>
        <h2 class="metrica-valor">{valor}</h2>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def encabezado_seccion(titulo: str, subtitulo: str = ""):
    """Encabezado de sección (legacy)"""
    subtitulo_html = f'<p class="encabezado-subtitulo">{subtitulo}</p>' if subtitulo else ''
    
    html = f"""
    <style>
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
    </style>
    
    <div class="encabezado-container">
        <h1 class="encabezado-titulo">{titulo}</h1>
        {subtitulo_html}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def badge_resultado(texto: str, estado: str = "neutro"):
    """Badge tipo píldora para resultados (legacy)"""
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
    
    color_config = colores.get(estado, colores["neutro"])
    
    html = f"""
    <style>
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
    </style>
    
    <span class="badge-resultado">{texto}</span>
    """
    
    st.markdown(html, unsafe_allow_html=True)
