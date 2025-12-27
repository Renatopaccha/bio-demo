import { TrendingUp, Activity, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

const regressionData = [
  { x: 22, y: 115, predicted: 118 },
  { x: 28, y: 128, predicted: 126 },
  { x: 32, y: 138, predicted: 135 },
  { x: 36, y: 145, predicted: 143 },
  { x: 40, y: 152, predicted: 152 },
  { x: 45, y: 165, predicted: 162 },
];

export function Modeling() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Modelos de Regresión</h1>
        <p className="text-gray-500">Regresión lineal, logística y modelos avanzados</p>
      </div>

      {/* Model Type Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white cursor-pointer hover:shadow-xl transition-all">
          <TrendingUp className="w-10 h-10 mb-4" />
          <h3 className="text-lg mb-2">Regresión Lineal</h3>
          <p className="text-blue-100 text-sm">Variable dependiente continua</p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 cursor-pointer hover:shadow-md transition-all">
          <Activity className="w-10 h-10 text-[#2c5f7c] mb-4" />
          <h3 className="text-lg text-[#1e293b] mb-2">Regresión Logística</h3>
          <p className="text-gray-500 text-sm">Variable dependiente binaria</p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 cursor-pointer hover:shadow-md transition-all">
          <TrendingUp className="w-10 h-10 text-[#2c5f7c] mb-4" />
          <h3 className="text-lg text-[#1e293b] mb-2">Cox (Supervivencia)</h3>
          <p className="text-gray-500 text-sm">Análisis de tiempo al evento</p>
        </div>
      </div>

      {/* Model Configuration */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Configuración del Modelo</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-700 mb-2">Variable Dependiente (Y)</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]">
              <option>Presión Arterial Sistólica</option>
              <option>IMC</option>
              <option>Colesterol Total</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-700 mb-2">Variables Independientes (X)</label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <label className="flex items-center gap-2 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl cursor-pointer hover:bg-gray-100">
                <input type="checkbox" className="w-4 h-4 text-[#2c5f7c]" />
                <span className="text-sm text-gray-700">Edad</span>
              </label>
              <label className="flex items-center gap-2 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl cursor-pointer hover:bg-gray-100">
                <input type="checkbox" className="w-4 h-4 text-[#2c5f7c]" />
                <span className="text-sm text-gray-700">Sexo</span>
              </label>
              <label className="flex items-center gap-2 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl cursor-pointer hover:bg-gray-100">
                <input type="checkbox" className="w-4 h-4 text-[#2c5f7c]" defaultChecked />
                <span className="text-sm text-gray-700">IMC</span>
              </label>
              <label className="flex items-center gap-2 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl cursor-pointer hover:bg-gray-100">
                <input type="checkbox" className="w-4 h-4 text-[#2c5f7c]" />
                <span className="text-sm text-gray-700">Diabetes</span>
              </label>
              <label className="flex items-center gap-2 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl cursor-pointer hover:bg-gray-100">
                <input type="checkbox" className="w-4 h-4 text-[#2c5f7c]" />
                <span className="text-sm text-gray-700">Tabaquismo</span>
              </label>
              <label className="flex items-center gap-2 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl cursor-pointer hover:bg-gray-100">
                <input type="checkbox" className="w-4 h-4 text-[#2c5f7c]" />
                <span className="text-sm text-gray-700">Actividad Física</span>
              </label>
            </div>
          </div>

          <button className="px-6 py-3 bg-gradient-to-r from-[#2c5f7c] to-[#4a9ebb] text-white rounded-xl hover:shadow-lg transition-all">
            Ajustar Modelo
          </button>
        </div>
      </div>

      {/* Model Results */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-xl p-6 text-white text-center">
          <p className="text-sm text-blue-100 mb-1">R²</p>
          <p className="text-3xl mb-1">0.68</p>
          <p className="text-xs text-blue-100">Bondad de ajuste</p>
        </div>
        <div className="bg-white rounded-xl p-6 border border-gray-200 text-center">
          <p className="text-sm text-gray-500 mb-1">R² Ajustado</p>
          <p className="text-3xl text-[#1e293b] mb-1">0.65</p>
          <p className="text-xs text-gray-500">Corregido por variables</p>
        </div>
        <div className="bg-white rounded-xl p-6 border border-gray-200 text-center">
          <p className="text-sm text-gray-500 mb-1">RMSE</p>
          <p className="text-3xl text-[#1e293b] mb-1">12.4</p>
          <p className="text-xs text-gray-500">Error cuadrático medio</p>
        </div>
      </div>

      {/* Coefficients Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Coeficientes del Modelo</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="text-left px-4 py-3 text-sm text-gray-700">Variable</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Coeficiente (β)</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Error Est.</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">t-valor</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">p-valor</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">IC 95%</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-4 py-3 text-sm text-[#1e293b]">Intercepto</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">82.45</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">8.23</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">10.02</td>
                <td className="px-4 py-3 text-sm text-[#2c5f7c] text-center">&lt;0.001</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">[66.3, 98.6]</td>
              </tr>
              <tr className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-4 py-3 text-sm text-[#1e293b]">IMC</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">1.85</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">0.34</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">5.44</td>
                <td className="px-4 py-3 text-sm text-[#2c5f7c] text-center">&lt;0.001</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">[1.18, 2.52]</td>
              </tr>
              <tr className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-4 py-3 text-sm text-[#1e293b]">Edad</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">0.42</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">0.15</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">2.80</td>
                <td className="px-4 py-3 text-sm text-[#2c5f7c] text-center">0.006</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">[0.12, 0.72]</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Regression Plot */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Gráfico de Regresión</h3>
        
        <ResponsiveContainer width="100%" height={350}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="x" stroke="#94a3b8" label={{ value: 'IMC (kg/m²)', position: 'insideBottom', offset: -5 }} />
            <YAxis stroke="#94a3b8" label={{ value: 'Presión Arterial (mmHg)', angle: 270, position: 'insideLeft' }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="Observado" data={regressionData} fill="#2c5f7c" />
            <Line type="monotone" dataKey="predicted" stroke="#4a9ebb" strokeWidth={3} dot={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Assumptions */}
      <div className="bg-orange-50 border border-orange-200 rounded-2xl p-6">
        <h4 className="text-[#1e293b] mb-3 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-orange-600" />
          Verificación de Supuestos
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <span className="text-emerald-600">✓</span>
            <span className="text-gray-700">Linealidad: Cumple</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-emerald-600">✓</span>
            <span className="text-gray-700">Normalidad de residuos: Cumple</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-emerald-600">✓</span>
            <span className="text-gray-700">Homocedasticidad: Cumple</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-orange-600">⚠</span>
            <span className="text-gray-700">Multicolinealidad: VIF máximo = 2.1</span>
          </div>
        </div>
      </div>
    </div>
  );
}
