import { 
  ArrowRight, BarChart2, Brain, FileSpreadsheet, TrendingUp, Zap, 
  CheckCircle, Sparkles, Shield, Users, Microscope, Code, Rocket,
  Activity, Target, Database, LineChart, Award, Clock
} from 'lucide-react';

export function Home() {
  return (
    <div className="space-y-16 pb-12">
      
      {/* ============================================
          HERO SECTION - Glassmorphism Premium
          ============================================ */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-[#0B3A82] via-[#2c5f7c] to-[#4a9ebb] p-1">
        {/* Fondo con patrón decorativo */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAxMCAwIEwgMCAwIDAgMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMC41IiBvcGFjaXR5PSIwLjEiLz48L3BhdHRlcm4+PC9kZWZzPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9InVybCgjZ3JpZCkiLz48L3N2Zz4=')] opacity-30"></div>
        
        <div className="relative bg-white/5 backdrop-blur-xl rounded-[22px] p-12">
          <div className="max-w-4xl mx-auto text-center">
            {/* Badge superior */}
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-md border border-white/20 rounded-full mb-6 text-white/90 text-sm">
              <Sparkles className="w-4 h-4" />
              <span>Bioestadística para Investigadores en Salud</span>
            </div>
            
            {/* Título principal */}
            <h1 className="text-5xl md:text-6xl text-white mb-6 tracking-tight">
              Estadística que <span className="text-[#7dd3fc]">impulsa</span> tu investigación
            </h1>
            
            {/* Subtítulo */}
            <p className="text-xl text-white/80 mb-8 max-w-2xl mx-auto leading-relaxed">
              Plataforma integral de análisis bioestadístico diseñada para investigadores, 
              tesistas y profesionales de la salud. Del dato crudo a la publicación científica.
            </p>
            
            {/* Mini KPIs en Hero */}
            <div className="flex flex-wrap justify-center gap-8 mb-10 text-white">
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 bg-white/10 rounded-xl flex items-center justify-center">
                  <CheckCircle className="w-5 h-5 text-emerald-300" />
                </div>
                <div className="text-left">
                  <div className="text-2xl">17+</div>
                  <div className="text-xs text-white/70">Módulos</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 bg-white/10 rounded-xl flex items-center justify-center">
                  <Activity className="w-5 h-5 text-purple-300" />
                </div>
                <div className="text-left">
                  <div className="text-2xl">50+</div>
                  <div className="text-xs text-white/70">Pruebas Estadísticas</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 bg-white/10 rounded-xl flex items-center justify-center">
                  <Brain className="w-5 h-5 text-blue-300" />
                </div>
                <div className="text-left">
                  <div className="text-2xl">IA</div>
                  <div className="text-xs text-white/70">Asistente 24/7</div>
                </div>
              </div>
            </div>
            
            {/* CTAs principales */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="group px-8 py-4 bg-white text-[#0B3A82] rounded-xl hover:shadow-2xl hover:shadow-white/20 transition-all flex items-center justify-center gap-2">
                <Rocket className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                <span>Comenzar Ahora</span>
              </button>
              <button className="px-8 py-4 bg-white/10 border-2 border-white/30 text-white rounded-xl hover:bg-white/20 transition-all backdrop-blur-md">
                📊 Ver Ejemplo de Análisis
              </button>
            </div>
          </div>
        </div>
      </div>

      
      {/* ============================================
          FEATURES GRID - Tarjetas Premium
          ============================================ */}
      <div>
        <div className="text-center mb-10">
          <h2 className="text-3xl text-[#0B3A82] mb-3">Herramientas Completas de Bioestadística</h2>
          <p className="text-gray-600">Todo lo que necesitas para tu tesis o investigación en un solo lugar</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          
          {/* Feature 1 */}
          <div className="group bg-white rounded-2xl p-6 border border-gray-200 hover:border-[#4a9ebb] hover:shadow-xl transition-all cursor-pointer">
            <div className="w-14 h-14 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <BarChart2 className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-[#1e293b] mb-2">Estadística Descriptiva</h3>
            <p className="text-sm text-gray-600 leading-relaxed mb-3">
              Medidas de tendencia central, dispersión, gráficos de distribución y análisis exploratorio completo
            </p>
            <div className="flex items-center gap-2 text-sm text-[#4a9ebb] group-hover:gap-3 transition-all">
              <span>Explorar</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </div>

          {/* Feature 2 */}
          <div className="group bg-white rounded-2xl p-6 border border-gray-200 hover:border-[#4a9ebb] hover:shadow-xl transition-all cursor-pointer">
            <div className="w-14 h-14 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Target className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-[#1e293b] mb-2">Pruebas de Inferencia</h3>
            <p className="text-sm text-gray-600 leading-relaxed mb-3">
              T-test, ANOVA, Chi-cuadrado, Mann-Whitney, Kruskal-Wallis y más pruebas paramétricas y no paramétricas
            </p>
            <div className="flex items-center gap-2 text-sm text-[#4a9ebb] group-hover:gap-3 transition-all">
              <span>Analizar</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </div>

          {/* Feature 3 */}
          <div className="group bg-white rounded-2xl p-6 border border-gray-200 hover:border-[#4a9ebb] hover:shadow-xl transition-all cursor-pointer">
            <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <TrendingUp className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-[#1e293b] mb-2">Modelado Avanzado</h3>
            <p className="text-sm text-gray-600 leading-relaxed mb-3">
              Regresión lineal múltiple, regresión logística, análisis multivariado y modelos predictivos
            </p>
            <div className="flex items-center gap-2 text-sm text-[#4a9ebb] group-hover:gap-3 transition-all">
              <span>Modelar</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </div>

          {/* Feature 4 */}
          <div className="group bg-white rounded-2xl p-6 border border-gray-200 hover:border-[#4a9ebb] hover:shadow-xl transition-all cursor-pointer">
            <div className="w-14 h-14 bg-gradient-to-br from-orange-500 to-orange-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <LineChart className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-[#1e293b] mb-2">Análisis de Supervivencia</h3>
            <p className="text-sm text-gray-600 leading-relaxed mb-3">
              Curvas Kaplan-Meier, Log-Rank test, Cox proportional hazards para estudios de seguimiento
            </p>
            <div className="flex items-center gap-2 text-sm text-[#4a9ebb] group-hover:gap-3 transition-all">
              <span>Calcular</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </div>

          {/* Feature 5 */}
          <div className="group bg-white rounded-2xl p-6 border border-gray-200 hover:border-[#4a9ebb] hover:shadow-xl transition-all cursor-pointer">
            <div className="w-14 h-14 bg-gradient-to-br from-pink-500 to-pink-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <FileSpreadsheet className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-[#1e293b] mb-2">Tabla 1 Automática</h3>
            <p className="text-sm text-gray-600 leading-relaxed mb-3">
              Genera tablas de características basales listas para publicación en formato APA/Vancouver
            </p>
            <div className="flex items-center gap-2 text-sm text-[#4a9ebb] group-hover:gap-3 transition-all">
              <span>Generar</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </div>

          {/* Feature 6 */}
          <div className="group bg-white rounded-2xl p-6 border border-gray-200 hover:border-[#4a9ebb] hover:shadow-xl transition-all cursor-pointer">
            <div className="w-14 h-14 bg-gradient-to-br from-cyan-500 to-cyan-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Brain className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-[#1e293b] mb-2">Asistente IA</h3>
            <p className="text-sm text-gray-600 leading-relaxed mb-3">
              Ayuda inteligente para elegir pruebas estadísticas, interpretar resultados y redactar hallazgos
            </p>
            <div className="flex items-center gap-2 text-sm text-[#4a9ebb] group-hover:gap-3 transition-all">
              <span>Consultar</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </div>

        </div>
      </div>


      {/* ============================================
          COMPARISON SECTION - Antes vs Ahora
          ============================================ */}
      <div className="bg-gradient-to-br from-gray-50 to-blue-50/30 rounded-3xl p-10">
        <div className="text-center mb-10">
          <h2 className="text-3xl text-[#0B3A82] mb-3">¿Por qué Biometric?</h2>
          <p className="text-gray-600">Compara el método tradicional con nuestra solución</p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          
          {/* Método Tradicional */}
          <div className="bg-white rounded-2xl p-8 border-2 border-gray-200">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
                <Clock className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h3 className="text-lg text-[#1e293b]">Método Tradicional</h3>
                <p className="text-sm text-gray-500">Complejo y fragmentado</p>
              </div>
            </div>
            
            <ul className="space-y-3">
              <li className="flex items-start gap-3 text-sm text-gray-700">
                <div className="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-red-600 text-xs">✕</span>
                </div>
                <span>Múltiples programas: Excel, SPSS, GraphPad, R/Python separados</span>
              </li>
              <li className="flex items-start gap-3 text-sm text-gray-700">
                <div className="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-red-600 text-xs">✕</span>
                </div>
                <span>Curva de aprendizaje empinada en programación estadística</span>
              </li>
              <li className="flex items-start gap-3 text-sm text-gray-700">
                <div className="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-red-600 text-xs">✕</span>
                </div>
                <span>Semanas de análisis manual y verificación de resultados</span>
              </li>
              <li className="flex items-start gap-3 text-sm text-gray-700">
                <div className="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-red-600 text-xs">✕</span>
                </div>
                <span>Formateo manual de tablas y gráficos para publicación</span>
              </li>
              <li className="flex items-start gap-3 text-sm text-gray-700">
                <div className="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-red-600 text-xs">✕</span>
                </div>
                <span>Sin guía para elegir la prueba estadística correcta</span>
              </li>
            </ul>
          </div>

          {/* Con Biometric */}
          <div className="bg-gradient-to-br from-[#0B3A82] to-[#2c5f7c] rounded-2xl p-8 border-2 border-[#4a9ebb] shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 bg-emerald-400/20 rounded-xl flex items-center justify-center">
                <Zap className="w-6 h-6 text-emerald-300" />
              </div>
              <div>
                <h3 className="text-lg text-white">Con Biometric</h3>
                <p className="text-sm text-white/70">Todo integrado y guiado</p>
              </div>
            </div>
            
            <ul className="space-y-3">
              <li className="flex items-start gap-3 text-sm text-white/90">
                <div className="w-5 h-5 rounded-full bg-emerald-400/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <CheckCircle className="w-3 h-3 text-emerald-300" />
                </div>
                <span>Plataforma todo-en-uno: importa, limpia, analiza y reporta en un lugar</span>
              </li>
              <li className="flex items-start gap-3 text-sm text-white/90">
                <div className="w-5 h-5 rounded-full bg-emerald-400/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <CheckCircle className="w-3 h-3 text-emerald-300" />
                </div>
                <span>Interfaz intuitiva sin necesidad de código o comandos complejos</span>
              </li>
              <li className="flex items-start gap-3 text-sm text-white/90">
                <div className="w-5 h-5 rounded-full bg-emerald-400/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <CheckCircle className="w-3 h-3 text-emerald-300" />
                </div>
                <span>Resultados en minutos con validación automática de supuestos</span>
              </li>
              <li className="flex items-start gap-3 text-sm text-white/90">
                <div className="w-5 h-5 rounded-full bg-emerald-400/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <CheckCircle className="w-3 h-3 text-emerald-300" />
                </div>
                <span>Tablas y gráficos listos para copiar a tu manuscrito</span>
              </li>
              <li className="flex items-start gap-3 text-sm text-white/90">
                <div className="w-5 h-5 rounded-full bg-emerald-400/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <CheckCircle className="w-3 h-3 text-emerald-300" />
                </div>
                <span>Asistente IA que recomienda y explica pruebas apropiadas</span>
              </li>
            </ul>
          </div>
          
        </div>
      </div>


      {/* ============================================
          ONBOARDING - 3 Pasos Tipo Stepper
          ============================================ */}
      <div>
        <div className="text-center mb-10">
          <h2 className="text-3xl text-[#0B3A82] mb-3">Comienza en 3 Pasos</h2>
          <p className="text-gray-600">De datos crudos a resultados publicables en minutos</p>
        </div>
        
        <div className="max-w-4xl mx-auto relative">
          {/* Línea conectora - solo visible en desktop */}
          <div className="hidden md:block absolute top-20 left-0 right-0 h-0.5 bg-gradient-to-r from-[#4a9ebb] via-[#2c5f7c] to-[#4a9ebb] opacity-20"></div>
          
          <div className="grid md:grid-cols-3 gap-8 relative">
            
            {/* Paso 1 */}
            <div className="text-center">
              <div className="relative inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl mb-4 shadow-lg">
                <Database className="w-8 h-8 text-white" />
                <div className="absolute -top-2 -right-2 w-8 h-8 bg-white rounded-full flex items-center justify-center text-sm text-[#0B3A82] shadow-md">
                  1
                </div>
              </div>
              <h3 className="text-lg text-[#1e293b] mb-2">Importa tus Datos</h3>
              <p className="text-sm text-gray-600 leading-relaxed">
                Arrastra tu archivo Excel o CSV. Limpia valores faltantes y detecta tipos de variables automáticamente
              </p>
            </div>

            {/* Paso 2 */}
            <div className="text-center">
              <div className="relative inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl mb-4 shadow-lg">
                <Activity className="w-8 h-8 text-white" />
                <div className="absolute -top-2 -right-2 w-8 h-8 bg-white rounded-full flex items-center justify-center text-sm text-[#0B3A82] shadow-md">
                  2
                </div>
              </div>
              <h3 className="text-lg text-[#1e293b] mb-2">Analiza con IA</h3>
              <p className="text-sm text-gray-600 leading-relaxed">
                Selecciona tus variables, consulta al Asistente IA qué prueba usar, ejecuta análisis con un clic
              </p>
            </div>

            {/* Paso 3 */}
            <div className="text-center">
              <div className="relative inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl mb-4 shadow-lg">
                <Award className="w-8 h-8 text-white" />
                <div className="absolute -top-2 -right-2 w-8 h-8 bg-white rounded-full flex items-center justify-center text-sm text-[#0B3A82] shadow-md">
                  3
                </div>
              </div>
              <h3 className="text-lg text-[#1e293b] mb-2">Exporta Resultados</h3>
              <p className="text-sm text-gray-600 leading-relaxed">
                Descarga tablas formateadas, gráficos de alta resolución y tu reporte completo en PDF/Word
              </p>
            </div>
            
          </div>
        </div>
      </div>


      {/* ============================================
          TRUST BADGES - Para Investigadores
          ============================================ */}
      <div className="grid md:grid-cols-3 gap-6">
        
        <div className="bg-white rounded-2xl p-8 border border-gray-200 text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Microscope className="w-8 h-8 text-[#0B3A82]" />
          </div>
          <h4 className="text-lg text-[#0B3A82] mb-2">Para Investigadores</h4>
          <p className="text-sm text-gray-600 leading-relaxed">
            Enfocado en ciencias de la salud, epidemiología y ensayos clínicos. Métodos validados académicamente
          </p>
        </div>

        <div className="bg-white rounded-2xl p-8 border border-gray-200 text-center">
          <div className="w-16 h-16 bg-emerald-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Code className="w-8 h-8 text-emerald-600" />
          </div>
          <h4 className="text-lg text-[#0B3A82] mb-2">Open Source</h4>
          <p className="text-sm text-gray-600 leading-relaxed">
            Código abierto, transparente y completamente auditable. Comunidad activa y en constante mejora
          </p>
        </div>

        <div className="bg-white rounded-2xl p-8 border border-gray-200 text-center">
          <div className="w-16 h-16 bg-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Rocket className="w-8 h-8 text-purple-600" />
          </div>
          <h4 className="text-lg text-[#0B3A82] mb-2">En Desarrollo Activo</h4>
          <p className="text-sm text-gray-600 leading-relaxed">
            Nuevas funcionalidades y mejoras cada semana basadas en feedback de usuarios reales
          </p>
        </div>

      </div>


      {/* ============================================
          MÓDULOS ADICIONALES - Accordion Visual
          ============================================ */}
      <div className="bg-gradient-to-br from-blue-50 to-purple-50/30 rounded-3xl p-10">
        <div className="text-center mb-8">
          <h2 className="text-3xl text-[#0B3A82] mb-3">17 Módulos Especializados</h2>
          <p className="text-gray-600">Herramientas para cada etapa de tu investigación</p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 max-w-6xl mx-auto">
          
          {[
            { name: 'Limpieza de Datos', icon: Database },
            { name: 'Explorador de Datos', icon: BarChart2 },
            { name: 'Estadística Descriptiva', icon: Activity },
            { name: 'Ajuste de Tasas', icon: TrendingUp },
            { name: 'Tabla 1', icon: FileSpreadsheet },
            { name: 'Pruebas de Inferencia', icon: Target },
            { name: 'Modelado', icon: LineChart },
            { name: 'Multivariado', icon: Zap },
            { name: 'Supervivencia', icon: Clock },
            { name: 'Psicometría', icon: Brain },
            { name: 'Asociaciones', icon: Users },
            { name: 'Concordancia', icon: CheckCircle },
            { name: 'Diagnósticos', icon: Shield },
            { name: 'Gráficos Suite', icon: BarChart2 },
            { name: 'Asistente IA', icon: Sparkles },
            { name: 'Mi Reporte', icon: FileSpreadsheet }
          ].map((module, idx) => (
            <div key={idx} className="bg-white rounded-xl p-4 border border-gray-200 hover:border-[#4a9ebb] hover:shadow-md transition-all cursor-pointer group">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[#4a9ebb]/10 rounded-lg flex items-center justify-center group-hover:bg-[#4a9ebb]/20 transition-colors flex-shrink-0">
                  <module.icon className="w-5 h-5 text-[#2c5f7c]" />
                </div>
                <span className="text-sm text-[#1e293b]">{module.name}</span>
              </div>
            </div>
          ))}
          
        </div>
      </div>


      {/* ============================================
          FINAL CTA - Quote Motivacional
          ============================================ */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-[#0B3A82] via-[#2c5f7c] to-[#4a9ebb] p-12 text-center">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAxMCAwIEwgMCAwIDAgMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMC41IiBvcGFjaXR5PSIwLjEiLz48L3BhdHRlcm4+PC9kZWZzPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9InVybCgjZ3JpZCkiLz48L3N2Zz4=')] opacity-20"></div>
        
        <div className="relative max-w-3xl mx-auto">
          <div className="text-6xl mb-6">💡</div>
          <blockquote className="text-2xl md:text-3xl text-white mb-4 leading-relaxed">
            "La estadística no tiene que ser intimidante.
            <br />
            Con las herramientas correctas, puede ser <span className="text-[#7dd3fc]">poderosa y accesible</span>."
          </blockquote>
          <p className="text-white/80 text-lg mb-8">— Equipo Biometric</p>
          
          <button className="px-10 py-4 bg-white text-[#0B3A82] rounded-xl hover:shadow-2xl hover:shadow-white/20 transition-all inline-flex items-center gap-3 group">
            <span className="text-lg">Empieza tu Primera Tesis</span>
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </div>

    </div>
  );
}
