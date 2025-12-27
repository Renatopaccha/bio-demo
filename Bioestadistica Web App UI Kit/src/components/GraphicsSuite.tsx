import { Palette, BarChart, PieChart, LineChart as LineChartIcon, ScatterChart as ScatterIcon } from 'lucide-react';

export function GraphicsSuite() {
  const chartTypes = [
    { name: 'Histograma', icon: BarChart, description: 'Distribución de frecuencias' },
    { name: 'Boxplot', icon: BarChart, description: 'Medidas de dispersión' },
    { name: 'Scatter Plot', icon: ScatterIcon, description: 'Relación entre variables' },
    { name: 'Gráfico de Líneas', icon: LineChartIcon, description: 'Tendencias temporales' },
    { name: 'Gráfico de Barras', icon: BarChart, description: 'Comparación categórica' },
    { name: 'Gráfico de Torta', icon: PieChart, description: 'Proporciones' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Suite Gráfica</h1>
        <p className="text-gray-500">Visualizaciones avanzadas para tus datos</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {chartTypes.map((chart, index) => {
          const Icon = chart.icon;
          return (
            <div 
              key={index}
              className="bg-white rounded-2xl p-6 border border-gray-200 hover:shadow-md transition-all cursor-pointer group"
            >
              <div className="w-12 h-12 bg-[#4a9ebb]/10 rounded-xl flex items-center justify-center mb-4 group-hover:bg-[#4a9ebb]/20 transition-colors">
                <Icon className="w-6 h-6 text-[#2c5f7c]" />
              </div>
              <h3 className="text-[#1e293b] mb-2">{chart.name}</h3>
              <p className="text-sm text-gray-500">{chart.description}</p>
            </div>
          );
        })}
      </div>

      <div className="bg-gradient-to-r from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-8 text-white">
        <div className="flex items-center gap-4 mb-4">
          <Palette className="w-10 h-10" />
          <div>
            <h3 className="text-xl mb-1">Exportación Profesional</h3>
            <p className="text-blue-100 text-sm">Gráficos de alta resolución listos para publicación</p>
          </div>
        </div>
        <div className="flex gap-3">
          <button className="px-5 py-2 bg-white text-[#2c5f7c] rounded-lg hover:shadow-lg transition-all">
            PNG (300 DPI)
          </button>
          <button className="px-5 py-2 bg-white/10 border border-white/30 rounded-lg hover:bg-white/20 transition-all">
            SVG (Vector)
          </button>
          <button className="px-5 py-2 bg-white/10 border border-white/30 rounded-lg hover:bg-white/20 transition-all">
            PDF
          </button>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-4">Personalización Avanzada</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm text-gray-700 mb-2">Paleta de Colores</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]">
              <option>Clínica (Azul-Verde)</option>
              <option>Viridis</option>
              <option>Plasma</option>
              <option>Personalizada</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-700 mb-2">Estilo de Gráfico</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb]">
              <option>Moderno</option>
              <option>Clásico</option>
              <option>Minimalista</option>
              <option>Científico</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
