import { Activity, Calendar, TrendingDown } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const survivalData = [
  { time: 0, group1: 100, group2: 100 },
  { time: 6, group1: 95, group2: 92 },
  { time: 12, group1: 88, group2: 82 },
  { time: 18, group1: 82, group2: 71 },
  { time: 24, group1: 75, group2: 62 },
  { time: 30, group1: 68, group2: 54 },
  { time: 36, group1: 62, group2: 48 },
];

export function Survival() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Análisis de Supervivencia</h1>
        <p className="text-gray-500">Kaplan-Meier, Log-Rank y Regresión de Cox</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white">
          <Activity className="w-8 h-8 mb-3" />
          <p className="text-sm text-blue-100 mb-1">Supervivencia a 3 años</p>
          <p className="text-3xl">62%</p>
        </div>
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <Calendar className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">Mediana de Supervivencia</p>
          <p className="text-3xl text-[#1e293b]">28.5</p>
          <p className="text-sm text-gray-500">meses</p>
        </div>
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <TrendingDown className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">Log-Rank p-valor</p>
          <p className="text-3xl text-[#2c5f7c]">&lt;0.001</p>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Curva de Kaplan-Meier</h3>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={survivalData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="time" stroke="#94a3b8" label={{ value: 'Tiempo (meses)', position: 'insideBottom', offset: -5 }} />
            <YAxis stroke="#94a3b8" domain={[0, 100]} label={{ value: 'Probabilidad de Supervivencia (%)', angle: 270, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="stepAfter" dataKey="group1" stroke="#2c5f7c" strokeWidth={3} name="Grupo Tratamiento" />
            <Line type="stepAfter" dataKey="group2" stroke="#4a9ebb" strokeWidth={3} name="Grupo Control" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
