import { Network, GitBranch, Box } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from 'recharts';

const pcaData = [
  { pc1: 2.5, pc2: 1.8, pc3: 0.5, group: 'A' },
  { pc1: -1.2, pc2: 2.1, pc3: -0.8, group: 'B' },
  { pc1: 3.1, pc2: -0.5, pc3: 1.2, group: 'A' },
  { pc1: -2.3, pc2: -1.8, pc3: 0.3, group: 'C' },
  { pc1: 1.5, pc2: 0.8, pc3: -1.1, group: 'B' },
];

export function Multivariate() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Análisis Multivariado</h1>
        <p className="text-gray-500">PCA, Análisis Factorial, Cluster y más</p>
      </div>

      {/* Method Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white cursor-pointer hover:shadow-xl transition-all">
          <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center mb-4">
            <Network className="w-6 h-6" />
          </div>
          <h3 className="text-lg mb-2">PCA</h3>
          <p className="text-blue-100 text-sm">Análisis de Componentes Principales</p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 cursor-pointer hover:shadow-md transition-all">
          <div className="w-12 h-12 bg-[#4a9ebb]/10 rounded-xl flex items-center justify-center mb-4">
            <Box className="w-6 h-6 text-[#2c5f7c]" />
          </div>
          <h3 className="text-lg text-[#1e293b] mb-2">Cluster</h3>
          <p className="text-gray-500 text-sm">Análisis de Conglomerados</p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 cursor-pointer hover:shadow-md transition-all">
          <div className="w-12 h-12 bg-[#4a9ebb]/10 rounded-xl flex items-center justify-center mb-4">
            <GitBranch className="w-6 h-6 text-[#2c5f7c]" />
          </div>
          <h3 className="text-lg text-[#1e293b] mb-2">Factorial</h3>
          <p className="text-gray-500 text-sm">Análisis Factorial Exploratorio</p>
        </div>
      </div>

      {/* PCA Results */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Biplot: PC1 vs PC2</h3>
        
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="pc1" name="PC1" stroke="#94a3b8" label={{ value: 'Componente Principal 1 (45.2%)', position: 'insideBottom', offset: -5 }} />
            <YAxis dataKey="pc2" name="PC2" stroke="#94a3b8" label={{ value: 'Componente Principal 2 (28.7%)', angle: 270, position: 'insideLeft' }} />
            <ZAxis range={[100, 400]} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="Grupo A" data={pcaData.filter(d => d.group === 'A')} fill="#2c5f7c" />
            <Scatter name="Grupo B" data={pcaData.filter(d => d.group === 'B')} fill="#4a9ebb" />
            <Scatter name="Grupo C" data={pcaData.filter(d => d.group === 'C')} fill="#7ec8de" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Variance Explained */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <p className="text-sm text-gray-500 mb-1">PC1</p>
          <p className="text-2xl text-[#1e293b]">45.2%</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <p className="text-sm text-gray-500 mb-1">PC2</p>
          <p className="text-2xl text-[#1e293b]">28.7%</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-200 text-center">
          <p className="text-sm text-gray-500 mb-1">PC3</p>
          <p className="text-2xl text-[#1e293b]">15.3%</p>
        </div>
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-xl p-4 text-white text-center">
          <p className="text-sm text-blue-100 mb-1">Total</p>
          <p className="text-2xl">89.2%</p>
        </div>
      </div>
    </div>
  );
}
