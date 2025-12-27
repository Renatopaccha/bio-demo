import { Database, Filter, Eye, Download } from 'lucide-react';
import { useState } from 'react';

export function DataExplorer() {
  const [selectedVariable, setSelectedVariable] = useState('edad');

  const sampleData = [
    { id: 1, edad: 45, sexo: 'M', imc: 27.5, presion: 135, diabetes: 'No' },
    { id: 2, edad: 52, sexo: 'F', imc: 29.3, presion: 142, diabetes: 'Sí' },
    { id: 3, edad: 38, sexo: 'M', imc: 24.8, presion: 128, diabetes: 'No' },
    { id: 4, edad: 61, sexo: 'F', imc: 31.2, presion: 156, diabetes: 'Sí' },
    { id: 5, edad: 47, sexo: 'M', imc: 26.9, presion: 138, diabetes: 'No' },
  ];

  const variables = [
    { name: 'edad', type: 'Numérica', missing: 0, unique: 248 },
    { name: 'sexo', type: 'Categórica', missing: 0, unique: 2 },
    { name: 'imc', type: 'Numérica', missing: 12, unique: 456 },
    { name: 'presion', type: 'Numérica', missing: 8, unique: 89 },
    { name: 'diabetes', type: 'Categórica', missing: 3, unique: 2 },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Modo Explorador</h1>
        <p className="text-gray-500">Visualiza y explora tus datos interactivamente</p>
      </div>

      {/* Dataset Info */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-xl p-4 text-white text-center">
          <Database className="w-8 h-8 mx-auto mb-2" />
          <p className="text-2xl">1,248</p>
          <p className="text-sm text-blue-100">Filas</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <Eye className="w-8 h-8 text-[#2c5f7c] mx-auto mb-2" />
          <p className="text-2xl text-[#1e293b]">24</p>
          <p className="text-sm text-gray-500">Variables</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <Filter className="w-8 h-8 text-[#2c5f7c] mx-auto mb-2" />
          <p className="text-2xl text-[#1e293b]">98.5%</p>
          <p className="text-sm text-gray-500">Completitud</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <Download className="w-8 h-8 text-[#2c5f7c] mx-auto mb-2" />
          <p className="text-2xl text-[#1e293b]">2.4 MB</p>
          <p className="text-sm text-gray-500">Tamaño</p>
        </div>
      </div>

      {/* Variables Summary */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Resumen de Variables</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="text-left px-4 py-3 text-sm text-gray-700">Variable</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Tipo</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Valores Únicos</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Faltantes</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Acción</th>
              </tr>
            </thead>
            <tbody>
              {variables.map((variable, index) => (
                <tr 
                  key={index} 
                  className={`border-b border-gray-100 hover:bg-gray-50 cursor-pointer ${
                    selectedVariable === variable.name ? 'bg-blue-50' : ''
                  }`}
                  onClick={() => setSelectedVariable(variable.name)}
                >
                  <td className="px-4 py-3 text-sm text-[#1e293b]">{variable.name}</td>
                  <td className="px-4 py-3 text-center">
                    <span className={`px-3 py-1 rounded-lg text-xs ${
                      variable.type === 'Numérica' 
                        ? 'bg-blue-100 text-blue-700' 
                        : 'bg-purple-100 text-purple-700'
                    }`}>
                      {variable.type}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700 text-center">{variable.unique}</td>
                  <td className="px-4 py-3 text-center">
                    <span className={`text-sm ${
                      variable.missing === 0 ? 'text-emerald-600' : 'text-orange-600'
                    }`}>
                      {variable.missing}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <button className="px-3 py-1 text-xs bg-[#2c5f7c] text-white rounded-lg hover:bg-[#234a61] transition-colors">
                      Ver Detalle
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Data Preview */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg text-[#1e293b]">Vista Previa de Datos</h3>
          <button className="px-4 py-2 text-sm text-[#2c5f7c] border border-[#2c5f7c] rounded-lg hover:bg-[#2c5f7c] hover:text-white transition-colors">
            Ver Todas las Filas
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-gray-200 bg-gray-50">
                <th className="px-4 py-3 text-left text-gray-700">ID</th>
                <th className="px-4 py-3 text-left text-gray-700">Edad</th>
                <th className="px-4 py-3 text-left text-gray-700">Sexo</th>
                <th className="px-4 py-3 text-left text-gray-700">IMC</th>
                <th className="px-4 py-3 text-left text-gray-700">Presión</th>
                <th className="px-4 py-3 text-left text-gray-700">Diabetes</th>
              </tr>
            </thead>
            <tbody>
              {sampleData.map((row) => (
                <tr key={row.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="px-4 py-3 text-gray-600">{row.id}</td>
                  <td className="px-4 py-3 text-[#1e293b]">{row.edad}</td>
                  <td className="px-4 py-3 text-[#1e293b]">{row.sexo}</td>
                  <td className="px-4 py-3 text-[#1e293b]">{row.imc}</td>
                  <td className="px-4 py-3 text-[#1e293b]">{row.presion}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs ${
                      row.diabetes === 'Sí' 
                        ? 'bg-red-100 text-red-700' 
                        : 'bg-emerald-100 text-emerald-700'
                    }`}>
                      {row.diabetes}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Filtros Avanzados</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-gray-700 mb-2">Variable</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]">
              <option>edad</option>
              <option>sexo</option>
              <option>imc</option>
              <option>presion</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-gray-700 mb-2">Condición</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]">
              <option>Mayor que</option>
              <option>Menor que</option>
              <option>Igual a</option>
              <option>Entre</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-gray-700 mb-2">Valor</label>
            <input 
              type="text" 
              placeholder="Ingresa valor..." 
              className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]"
            />
          </div>
        </div>

        <div className="mt-4 flex gap-3">
          <button className="px-5 py-2.5 bg-gradient-to-r from-[#2c5f7c] to-[#4a9ebb] text-white rounded-xl hover:shadow-lg transition-all">
            Aplicar Filtro
          </button>
          <button className="px-5 py-2.5 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 transition-all">
            Limpiar Filtros
          </button>
        </div>
      </div>
    </div>
  );
}
