import { Target, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const rocData = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.05, tpr: 0.62 },
  { fpr: 0.10, tpr: 0.78 },
  { fpr: 0.20, tpr: 0.88 },
  { fpr: 0.35, tpr: 0.94 },
  { fpr: 0.50, tpr: 0.97 },
  { fpr: 1.0, tpr: 1.0 },
];

export function Diagnostics() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Pruebas Diagnósticas (ROC)</h1>
        <p className="text-gray-500">Sensibilidad, Especificidad y Curva ROC</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-xl p-4 text-white text-center">
          <p className="text-sm text-blue-100 mb-1">AUC</p>
          <p className="text-3xl">0.89</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <p className="text-sm text-gray-500 mb-1">Sensibilidad</p>
          <p className="text-2xl text-[#1e293b]">85.3%</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <p className="text-sm text-gray-500 mb-1">Especificidad</p>
          <p className="text-2xl text-[#1e293b]">79.2%</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <p className="text-sm text-gray-500 mb-1">Punto de Corte</p>
          <p className="text-2xl text-[#1e293b]">125</p>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Curva ROC</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={rocData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="fpr" stroke="#94a3b8" label={{ value: '1 - Especificidad', position: 'insideBottom', offset: -5 }} />
            <YAxis dataKey="tpr" stroke="#94a3b8" label={{ value: 'Sensibilidad', angle: 270, position: 'insideLeft' }} />
            <Tooltip />
            <Line type="monotone" dataKey="tpr" stroke="#2c5f7c" strokeWidth={3} dot={{ fill: '#4a9ebb', r: 5 }} />
            <Line type="monotone" data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" stroke="#94a3b8" strokeDasharray="5 5" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <p className="text-sm text-gray-700">
          <strong>Interpretación:</strong> El área bajo la curva (AUC = 0.89) indica una excelente capacidad discriminatoria del test. 
          Con un punto de corte de 125, se logra un balance óptimo entre sensibilidad (85.3%) y especificidad (79.2%).
        </p>
      </div>
    </div>
  );
}
