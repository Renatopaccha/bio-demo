import { FlaskConical, CheckCircle, XCircle, TrendingUp } from 'lucide-react';

export function InferenceTests() {
  const testResults = [
    { test: 't de Student', statistic: '2.845', pValue: '0.012', result: 'Rechazar H₀', conclusion: 'Existe diferencia significativa entre grupos' },
    { test: 'Mann-Whitney U', statistic: '1234.5', pValue: '0.008', result: 'Rechazar H₀', conclusion: 'Diferencia significativa en medianas' },
    { test: 'Chi-cuadrado', statistic: '12.45', pValue: '0.002', result: 'Rechazar H₀', conclusion: 'Asociación significativa entre variables' },
    { test: 'ANOVA', statistic: 'F=5.67', pValue: '0.018', result: 'Rechazar H₀', conclusion: 'Diferencias entre 3 o más grupos' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Pruebas de Hipótesis</h1>
        <p className="text-gray-500">Tests estadísticos paramétricos y no paramétricos</p>
      </div>

      {/* Test Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white cursor-pointer hover:shadow-xl transition-all">
          <FlaskConical className="w-10 h-10 mb-4" />
          <h3 className="text-xl mb-2">Pruebas Paramétricas</h3>
          <p className="text-blue-100 text-sm mb-4">Asumen distribución normal de los datos</p>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4" />
              <span>t de Student</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4" />
              <span>ANOVA</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4" />
              <span>Correlación de Pearson</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 cursor-pointer hover:shadow-md transition-all">
          <FlaskConical className="w-10 h-10 text-[#2c5f7c] mb-4" />
          <h3 className="text-xl text-[#1e293b] mb-2">Pruebas No Paramétricas</h3>
          <p className="text-gray-500 text-sm mb-4">No requieren normalidad en los datos</p>
          <div className="space-y-2 text-sm text-gray-700">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-[#2c5f7c]" />
              <span>Mann-Whitney U</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-[#2c5f7c]" />
              <span>Kruskal-Wallis</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-[#2c5f7c]" />
              <span>Spearman</span>
            </div>
          </div>
        </div>
      </div>

      {/* Test Configuration */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Configurar Prueba</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm text-gray-700 mb-2">Tipo de Prueba</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]">
              <option>t de Student (2 grupos)</option>
              <option>ANOVA (3+ grupos)</option>
              <option>Mann-Whitney U</option>
              <option>Chi-cuadrado</option>
              <option>Correlación</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-700 mb-2">Variable Dependiente</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]">
              <option>Presión Arterial</option>
              <option>IMC</option>
              <option>Edad</option>
              <option>Colesterol</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-700 mb-2">Variable Independiente/Grupo</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]">
              <option>Grupo Tratamiento</option>
              <option>Sexo</option>
              <option>Diabetes</option>
              <option>Hipertensión</option>
            </select>
          </div>
        </div>

        <div className="mt-6 flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-sm text-gray-700 mb-2">Nivel de Significancia (α)</label>
            <input 
              type="number" 
              defaultValue="0.05" 
              step="0.01"
              className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]"
            />
          </div>
          <div className="flex-1">
            <label className="block text-sm text-gray-700 mb-2">Tipo de Test</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]">
              <option>Dos colas</option>
              <option>Una cola (mayor)</option>
              <option>Una cola (menor)</option>
            </select>
          </div>
        </div>

        <button className="mt-6 px-6 py-3 bg-gradient-to-r from-[#2c5f7c] to-[#4a9ebb] text-white rounded-xl hover:shadow-lg transition-all">
          Ejecutar Prueba
        </button>
      </div>

      {/* Results */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Resultados de Pruebas</h3>
        
        <div className="space-y-4">
          {testResults.map((test, index) => (
            <div key={index} className="p-5 bg-gray-50 rounded-xl border border-gray-100">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h4 className="text-[#1e293b] mb-1">{test.test}</h4>
                  <p className="text-sm text-gray-500">Estadístico: {test.statistic}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500 mb-1">p-valor</p>
                  <p className={`text-xl ${parseFloat(test.pValue) < 0.05 ? 'text-[#2c5f7c]' : 'text-gray-700'}`}>
                    {test.pValue}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-3 p-3 bg-white rounded-lg">
                {parseFloat(test.pValue) < 0.05 ? (
                  <CheckCircle className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                ) : (
                  <XCircle className="w-5 h-5 text-gray-400 flex-shrink-0" />
                )}
                <div>
                  <p className="text-sm text-[#1e293b]">{test.result}</p>
                  <p className="text-xs text-gray-600 mt-1">{test.conclusion}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interpretation Guide */}
      <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6">
        <h4 className="text-[#1e293b] mb-3 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-[#2c5f7c]" />
          Guía de Interpretación
        </h4>
        <ul className="space-y-2 text-sm text-gray-700">
          <li className="flex items-start gap-2">
            <span className="text-[#2c5f7c] flex-shrink-0">•</span>
            <span><strong>p &lt; 0.05:</strong> Resultado estadísticamente significativo, rechazar hipótesis nula</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-[#2c5f7c] flex-shrink-0">•</span>
            <span><strong>p ≥ 0.05:</strong> No hay evidencia suficiente para rechazar la hipótesis nula</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-[#2c5f7c] flex-shrink-0">•</span>
            <span><strong>IC 95%:</strong> Rango donde se encuentra el verdadero valor poblacional con 95% de confianza</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
