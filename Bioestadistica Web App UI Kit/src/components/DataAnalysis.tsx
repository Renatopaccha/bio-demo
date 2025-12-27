import { Upload, Table, FileSpreadsheet, Download, Filter } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from 'recharts';

const scatterData = [
  { x: 25, y: 120, z: 65 },
  { x: 32, y: 135, z: 72 },
  { x: 45, y: 145, z: 68 },
  { x: 28, y: 118, z: 70 },
  { x: 51, y: 152, z: 75 },
  { x: 38, y: 128, z: 66 },
  { x: 42, y: 138, z: 71 },
  { x: 55, y: 162, z: 78 },
  { x: 30, y: 122, z: 64 },
  { x: 48, y: 148, z: 73 },
];

export function DataAnalysis() {
  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl text-gray-900">Análisis de Datos</h1>
        <p className="text-gray-600">Importa, visualiza y analiza tus datos de investigación</p>
      </div>

      {/* Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <button className="p-4 bg-white rounded-lg border-2 border-dashed border-blue-300 hover:border-blue-500 hover:bg-blue-50 transition-all group">
          <Upload className="w-8 h-8 text-blue-500 mb-2 group-hover:scale-110 transition-transform" />
          <p className="text-gray-900">Importar Datos</p>
          <p className="text-xs text-gray-600">CSV, Excel, SPSS</p>
        </button>
        
        <button className="p-4 bg-white rounded-lg border border-gray-200 hover:shadow-md transition-all">
          <Table className="w-8 h-8 text-emerald-500 mb-2" />
          <p className="text-gray-900">Ver Tabla</p>
          <p className="text-xs text-gray-600">Datos actuales</p>
        </button>
        
        <button className="p-4 bg-white rounded-lg border border-gray-200 hover:shadow-md transition-all">
          <Filter className="w-8 h-8 text-purple-500 mb-2" />
          <p className="text-gray-900">Filtrar Datos</p>
          <p className="text-xs text-gray-600">Criterios avanzados</p>
        </button>
        
        <button className="p-4 bg-white rounded-lg border border-gray-200 hover:shadow-md transition-all">
          <Download className="w-8 h-8 text-cyan-500 mb-2" />
          <p className="text-gray-900">Exportar</p>
          <p className="text-xs text-gray-600">Resultados</p>
        </button>
      </div>

      {/* Visualization Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <div className="mb-4">
            <h3 className="text-lg text-gray-900">Análisis de Correlación</h3>
            <p className="text-sm text-gray-600">IMC vs Presión Arterial vs Edad</p>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="x" name="IMC" stroke="#6b7280" />
              <YAxis dataKey="y" name="Presión" stroke="#6b7280" />
              <ZAxis dataKey="z" range={[60, 400]} name="Edad" />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px'
                }}
              />
              <Scatter name="Pacientes" data={scatterData} fill="#8b5cf6" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <div className="mb-4">
            <h3 className="text-lg text-gray-900">Estadísticas Descriptivas</h3>
            <p className="text-sm text-gray-600">Resumen de variables principales</p>
          </div>
          
          <div className="space-y-4">
            <div className="p-4 bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-700">Media (IMC)</span>
                <span className="text-gray-900">26.8 kg/m²</span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-700">Desviación Estándar</span>
                <span className="text-gray-900">±4.2</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-700">IC 95%</span>
                <span className="text-gray-900">24.6 - 29.0</span>
              </div>
            </div>

            <div className="p-4 bg-gradient-to-r from-emerald-50 to-emerald-100 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-700">Media (Presión Sistólica)</span>
                <span className="text-gray-900">138.5 mmHg</span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-700">Desviación Estándar</span>
                <span className="text-gray-900">±12.6</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-700">IC 95%</span>
                <span className="text-gray-900">132.1 - 144.9</span>
              </div>
            </div>

            <div className="p-4 bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-700">Tamaño de Muestra</span>
                <span className="text-gray-900">n = 1,248</span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-700">Poder Estadístico</span>
                <span className="text-gray-900">0.85</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-700">Nivel de Significancia</span>
                <span className="text-gray-900">α = 0.05</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Data Table Preview */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-lg text-gray-900">Vista Previa de Datos</h3>
            <p className="text-sm text-gray-600">Últimos registros importados</p>
          </div>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            Ver Todo
          </button>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="px-4 py-3 text-left text-sm text-gray-700">ID Paciente</th>
                <th className="px-4 py-3 text-left text-sm text-gray-700">Edad</th>
                <th className="px-4 py-3 text-left text-sm text-gray-700">Sexo</th>
                <th className="px-4 py-3 text-left text-sm text-gray-700">IMC</th>
                <th className="px-4 py-3 text-left text-sm text-gray-700">Presión (mmHg)</th>
                <th className="px-4 py-3 text-left text-sm text-gray-700">Glucosa (mg/dL)</th>
                <th className="px-4 py-3 text-left text-sm text-gray-700">Estado</th>
              </tr>
            </thead>
            <tbody>
              {[
                { id: 'P-1245', edad: 45, sexo: 'M', imc: 28.5, presion: '140/90', glucosa: 102, estado: 'Normal' },
                { id: 'P-1246', edad: 52, sexo: 'F', imc: 31.2, presion: '155/95', glucosa: 128, estado: 'Alerta' },
                { id: 'P-1247', edad: 38, sexo: 'M', imc: 24.8, presion: '125/80', glucosa: 95, estado: 'Normal' },
                { id: 'P-1248', edad: 61, sexo: 'F', imc: 29.4, presion: '148/92', glucosa: 115, estado: 'Normal' },
              ].map((row) => (
                <tr key={row.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900">{row.id}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{row.edad}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{row.sexo}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{row.imc}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{row.presion}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{row.glucosa}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      row.estado === 'Normal' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
                    }`}>
                      {row.estado}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
