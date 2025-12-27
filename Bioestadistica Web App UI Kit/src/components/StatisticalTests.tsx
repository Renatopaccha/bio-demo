import { Calculator, CheckCircle2, Play } from 'lucide-react';

const tests = [
  {
    category: 'Pruebas Paramétricas',
    items: [
      { name: 'T-Test (Student)', description: 'Comparación de medias entre dos grupos', color: 'bg-blue-500' },
      { name: 'ANOVA', description: 'Análisis de varianza múltiple', color: 'bg-blue-500' },
      { name: 'Regresión Lineal', description: 'Relación entre variables continuas', color: 'bg-blue-500' },
      { name: 'Correlación de Pearson', description: 'Asociación lineal entre variables', color: 'bg-blue-500' },
    ],
  },
  {
    category: 'Pruebas No Paramétricas',
    items: [
      { name: 'Mann-Whitney U', description: 'Alternativa al T-Test', color: 'bg-emerald-500' },
      { name: 'Kruskal-Wallis', description: 'Alternativa a ANOVA', color: 'bg-emerald-500' },
      { name: 'Chi-Cuadrado', description: 'Asociación entre variables categóricas', color: 'bg-emerald-500' },
      { name: 'Wilcoxon', description: 'Prueba de rangos con signo', color: 'bg-emerald-500' },
    ],
  },
  {
    category: 'Análisis de Supervivencia',
    items: [
      { name: 'Kaplan-Meier', description: 'Curvas de supervivencia', color: 'bg-purple-500' },
      { name: 'Log-Rank Test', description: 'Comparación de curvas', color: 'bg-purple-500' },
      { name: 'Cox Regression', description: 'Riesgos proporcionales', color: 'bg-purple-500' },
    ],
  },
  {
    category: 'Pruebas de Diagnóstico',
    items: [
      { name: 'Sensibilidad/Especificidad', description: 'Precisión diagnóstica', color: 'bg-cyan-500' },
      { name: 'Curva ROC', description: 'Capacidad discriminativa', color: 'bg-cyan-500' },
      { name: 'Valores Predictivos', description: 'VPP y VPN', color: 'bg-cyan-500' },
    ],
  },
];

export function StatisticalTests() {
  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl text-gray-900">Pruebas Estadísticas</h1>
        <p className="text-gray-600">Selecciona y ejecuta análisis estadísticos para tu investigación</p>
      </div>

      {/* Quick Test Panel */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
            <Calculator className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-xl">Asistente de Pruebas</h3>
            <p className="text-blue-100 text-sm">Déjanos ayudarte a elegir la prueba correcta</p>
          </div>
        </div>
        <button className="px-6 py-3 bg-white text-blue-600 rounded-lg hover:bg-blue-50 transition-colors">
          Iniciar Asistente
        </button>
      </div>

      {/* Tests Grid */}
      <div className="space-y-6">
        {tests.map((category) => (
          <div key={category.category} className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-lg text-gray-900 mb-4">{category.category}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {category.items.map((test) => (
                <div 
                  key={test.name} 
                  className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-all cursor-pointer group"
                >
                  <div className="flex items-start gap-3">
                    <div className={`w-10 h-10 ${test.color} rounded-lg flex items-center justify-center flex-shrink-0`}>
                      <CheckCircle2 className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-gray-900 mb-1">{test.name}</h4>
                      <p className="text-sm text-gray-600 mb-3">{test.description}</p>
                      <button className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Play className="w-4 h-4" />
                        Ejecutar Prueba
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Sample Size Calculator */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h3 className="text-lg text-gray-900 mb-4">Calculadora de Tamaño Muestral</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-gray-700 mb-2">Nivel de Confianza (%)</label>
            <input 
              type="number" 
              defaultValue="95" 
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-700 mb-2">Margen de Error (%)</label>
            <input 
              type="number" 
              defaultValue="5" 
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-700 mb-2">Poder Estadístico</label>
            <input 
              type="number" 
              defaultValue="0.80" 
              step="0.01"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>
        <div className="mt-4 flex justify-between items-center">
          <div className="text-sm text-gray-600">
            Tamaño muestral recomendado: <span className="text-xl text-gray-900">n = 384</span>
          </div>
          <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            Calcular
          </button>
        </div>
      </div>
    </div>
  );
}
