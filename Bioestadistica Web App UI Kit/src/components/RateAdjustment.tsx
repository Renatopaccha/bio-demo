import { Scale, TrendingUp, Users, Calculator } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const adjustedData = [
  { age: '0-19', crude: 12.5, adjusted: 11.8, standard: 10.2 },
  { age: '20-39', crude: 28.3, adjusted: 26.5, standard: 25.1 },
  { age: '40-59', crude: 45.7, adjusted: 42.3, standard: 41.8 },
  { age: '60-79', crude: 62.4, adjusted: 58.9, standard: 60.2 },
  { age: '80+', crude: 85.2, adjusted: 79.4, standard: 82.1 },
];

export function RateAdjustment() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Ajuste de Tasas</h1>
        <p className="text-gray-500">Estandarización directa e indirecta de tasas epidemiológicas</p>
      </div>

      {/* Method Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white cursor-pointer hover:shadow-xl transition-all">
          <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center mb-4">
            <Scale className="w-6 h-6" />
          </div>
          <h3 className="text-lg mb-2">Estandarización Directa</h3>
          <p className="text-blue-100 text-sm mb-4">Aplicar población estándar conocida</p>
          <button className="px-4 py-2 bg-white text-[#2c5f7c] rounded-lg hover:shadow-lg transition-all text-sm">
            Seleccionar Método
          </button>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 cursor-pointer hover:shadow-md transition-all">
          <div className="w-12 h-12 bg-[#4a9ebb]/10 rounded-xl flex items-center justify-center mb-4">
            <Calculator className="w-6 h-6 text-[#2c5f7c]" />
          </div>
          <h3 className="text-lg text-[#1e293b] mb-2">Estandarización Indirecta</h3>
          <p className="text-gray-500 text-sm mb-4">Calcular Razón de Mortalidad Estandarizada (SMR)</p>
          <button className="px-4 py-2 border border-[#2c5f7c] text-[#2c5f7c] rounded-lg hover:bg-[#2c5f7c] hover:text-white transition-all text-sm">
            Seleccionar Método
          </button>
        </div>
      </div>

      {/* Configuration */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Configuración del Ajuste</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm text-gray-700 mb-2">Variable de Estratificación</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb] focus:border-transparent">
              <option>Edad</option>
              <option>Sexo</option>
              <option>Región Geográfica</option>
              <option>Nivel Socioeconómico</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-700 mb-2">Población de Referencia</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb] focus:border-transparent">
              <option>OMS 2000 (Mundial)</option>
              <option>EE.UU. 2010</option>
              <option>Europa 2013</option>
              <option>Latinoamérica 2015</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-700 mb-2">Tipo de Tasa</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb] focus:border-transparent">
              <option>Mortalidad</option>
              <option>Incidencia</option>
              <option>Prevalencia</option>
              <option>Morbilidad</option>
            </select>
          </div>
        </div>
      </div>

      {/* Results Comparison */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Comparación de Tasas por Grupo de Edad</h3>
        
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={adjustedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
            <XAxis dataKey="age" stroke="#94a3b8" style={{ fontSize: '12px' }} />
            <YAxis stroke="#94a3b8" style={{ fontSize: '12px' }} label={{ value: 'Tasa por 100,000', angle: 270, position: 'insideLeft' }} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'white', 
                border: '1px solid #e2e8f0',
                borderRadius: '12px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
              }} 
            />
            <Legend />
            <Bar dataKey="crude" fill="#94a3b8" radius={[8, 8, 0, 0]} name="Tasa Cruda" />
            <Bar dataKey="adjusted" fill="#4a9ebb" radius={[8, 8, 0, 0]} name="Tasa Ajustada" />
            <Bar dataKey="standard" fill="#2c5f7c" radius={[8, 8, 0, 0]} name="Población Estándar" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-gray-100 rounded-xl flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-gray-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Tasa Cruda</p>
              <p className="text-2xl text-[#1e293b]">46.8</p>
            </div>
          </div>
          <p className="text-xs text-gray-500">por 100,000 habitantes</p>
        </div>

        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm text-blue-100">Tasa Ajustada</p>
              <p className="text-2xl">43.8</p>
            </div>
          </div>
          <p className="text-xs text-blue-100">por 100,000 habitantes</p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-emerald-100 rounded-xl flex items-center justify-center">
              <Users className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">IC 95%</p>
              <p className="text-xl text-[#1e293b]">[41.2, 46.4]</p>
            </div>
          </div>
          <p className="text-xs text-gray-500">Intervalo de confianza</p>
        </div>
      </div>

      {/* Interpretation */}
      <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6">
        <h4 className="text-[#1e293b] mb-3 flex items-center gap-2">
          <Scale className="w-5 h-5 text-[#2c5f7c]" />
          Interpretación
        </h4>
        <p className="text-sm text-gray-700 leading-relaxed">
          La tasa ajustada por edad (43.8 por 100,000) es menor que la tasa cruda (46.8 por 100,000), 
          lo que sugiere que la población de estudio tiene una estructura etaria más joven que la población estándar. 
          La diferencia del 6.4% es estadísticamente significativa (IC 95%: 41.2-46.4).
        </p>
      </div>
    </div>
  );
}
