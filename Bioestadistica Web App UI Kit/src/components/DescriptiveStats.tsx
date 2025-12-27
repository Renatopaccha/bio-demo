import { BarChart3, PieChart, TrendingUp } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart as RPieChart, Pie, Cell } from 'recharts';
import { useEffect, useState } from 'react';
import { getDescriptiveStats, generatePlot } from '../services/api';

const COLORS = ['#2c5f7c', '#4a9ebb'];

export function DescriptiveStats() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [plotImage, setPlotImage] = useState<string | null>(null);

  useEffect(() => {
    const sessionId = localStorage.getItem('session_id');
    if (sessionId) {
      setLoading(true);
      // Fetch stats for 'age' (assuming it exists in the uploaded file for demo)
      // In a real app, we would select the variable.
      getDescriptiveStats(sessionId, 'age')
        .then(data => {
          setStats(data);
          setLoading(false);
        })
        .catch(err => {
          console.error("Failed to fetch stats", err);
          setLoading(false);
        });

      generatePlot(sessionId, 'hist', 'age')
        .then(data => setPlotImage(data.image_base64))
        .catch(err => console.error("Plot error", err));
    }
  }, []);

  if (loading) return <div>Cargando estadísticas...</div>;
  if (!stats) return <div>No hay datos. Sube un archivo en Limpieza de Datos.</div>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Estadística Descriptiva</h1>
        <p className="text-gray-500">Resumen completo de medidas de tendencia central y dispersión</p>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white">
          <BarChart3 className="w-8 h-8 mb-3" />
          <p className="text-sm text-blue-100 mb-1">Media (Age)</p>
          <p className="text-3xl">{stats.mean?.toFixed(1)}</p>
          <p className="text-sm text-blue-100 mt-1">± {stats.std?.toFixed(1)} SD</p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200">
          <TrendingUp className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">Mediana</p>
          <p className="text-3xl text-[#1e293b]">{stats.median?.toFixed(1)}</p>
          <p className="text-sm text-gray-500 mt-1">IQR: {stats.iqr}</p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200">
          <BarChart3 className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">C.V.</p>
          <p className="text-3xl text-[#1e293b]">{stats.cv?.toFixed(1)}%</p>
          <p className="text-sm text-gray-500 mt-1">Coef. Variación</p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200">
          <PieChart className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">Normalidad (SW)</p>
          <p className="text-3xl text-[#1e293b]">{stats.shapiro_p?.toFixed(3)}</p>
          <p className="text-sm text-gray-500 mt-1">{stats.shapiro_p > 0.05 ? 'Normal' : 'No Normal'}</p>
        </div>
      </div>

      {/* Detailed Statistics Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Estadísticos por Variable</h3>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="text-left px-4 py-3 text-sm text-gray-700">Variable</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">n</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Media</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">DE</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Mediana</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">RIC</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Mín-Máx</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-4 py-3 text-sm text-[#1e293b]">Edad (años)</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">1,248</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">52.3</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">12.4</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">51.0</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">[43, 61]</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">22-89</td>
              </tr>
              <tr className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-4 py-3 text-sm text-[#1e293b]">IMC (kg/m²)</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">1,236</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">27.8</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">4.6</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">27.2</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">[24.5, 31.1]</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">18.5-42.8</td>
              </tr>
              <tr className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-4 py-3 text-sm text-[#1e293b]">Presión Arterial (mmHg)</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">1,240</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">138.5</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">18.3</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">136.0</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">[125, 150]</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-center">90-195</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Visualizations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-lg text-[#1e293b] mb-6">Distribución de Edad (Python)</h3>
          {plotImage ? (
            <img
              src={`data:image/png;base64,${plotImage}`}
              alt="Distribución de Edad"
              className="w-full h-auto rounded-lg"
            />
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-400">
              Cargando gráfico...
            </div>
          )}
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-lg text-[#1e293b] mb-6">Distribución por Sexo</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RPieChart>
              <Pie
                data={genderData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {genderData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </RPieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Normality Tests */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Pruebas de Normalidad</h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-4 bg-gray-50 rounded-xl">
            <p className="text-sm text-gray-500 mb-2">Shapiro-Wilk</p>
            <p className="text-2xl text-[#1e293b] mb-1">0.987</p>
            <p className="text-xs text-emerald-600">p = 0.156 (Normal)</p>
          </div>

          <div className="p-4 bg-gray-50 rounded-xl">
            <p className="text-sm text-gray-500 mb-2">Kolmogorov-Smirnov</p>
            <p className="text-2xl text-[#1e293b] mb-1">0.042</p>
            <p className="text-xs text-emerald-600">p = 0.089 (Normal)</p>
          </div>

          <div className="p-4 bg-gray-50 rounded-xl">
            <p className="text-sm text-gray-500 mb-2">Asimetría</p>
            <p className="text-2xl text-[#1e293b] mb-1">0.23</p>
            <p className="text-xs text-gray-600">Curtosis: 0.15</p>
          </div>
        </div>
      </div>
    </div>
  );
}
