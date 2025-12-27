import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { categoria: 'Cardiología', casos: 145 },
  { categoria: 'Oncología', casos: 298 },
  { categoria: 'Neurología', casos: 187 },
  { categoria: 'Endocrinología', casos: 156 },
  { categoria: 'Inmunología', casos: 223 },
];

export function DistributionChart() {
  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
      <div className="mb-4">
        <h3 className="text-lg text-gray-900">Distribución por Especialidad</h3>
        <p className="text-sm text-gray-600">Casos activos en cada área</p>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="categoria" stroke="#6b7280" angle={-15} textAnchor="end" height={80} />
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
          <Bar 
            dataKey="casos" 
            fill="url(#colorGradient)" 
            radius={[8, 8, 0, 0]}
            name="Casos"
          />
          <defs>
            <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#8b5cf6" stopOpacity={1} />
              <stop offset="100%" stopColor="#6366f1" stopOpacity={0.8} />
            </linearGradient>
          </defs>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
