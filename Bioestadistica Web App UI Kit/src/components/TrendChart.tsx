import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { mes: 'Ene', muestras: 245, analisis: 198 },
  { mes: 'Feb', muestras: 312, analisis: 267 },
  { mes: 'Mar', muestras: 289, analisis: 241 },
  { mes: 'Abr', muestras: 378, analisis: 325 },
  { mes: 'May', muestras: 425, analisis: 389 },
  { mes: 'Jun', muestras: 468, analisis: 412 },
];

export function TrendChart() {
  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
      <div className="mb-4">
        <h3 className="text-lg text-gray-900">Tendencia de Investigación</h3>
        <p className="text-sm text-gray-600">Muestras y análisis por mes</p>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="mes" stroke="#6b7280" />
          <YAxis stroke="#6b7280" />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'white', 
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
            }} 
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="muestras" 
            stroke="#3b82f6" 
            strokeWidth={3}
            name="Muestras"
            dot={{ fill: '#3b82f6', r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="analisis" 
            stroke="#10b981" 
            strokeWidth={3}
            name="Análisis Completados"
            dot={{ fill: '#10b981', r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
