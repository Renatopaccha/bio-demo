import { Brain, CheckSquare, TrendingUp } from 'lucide-react';

export function Psychometrics() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Psicometría</h1>
        <p className="text-gray-500">Análisis de confiabilidad, validez y estructura factorial</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white">
          <Brain className="w-8 h-8 mb-3" />
          <p className="text-sm text-blue-100 mb-1">Alpha de Cronbach</p>
          <p className="text-3xl">0.89</p>
        </div>
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <CheckSquare className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">Ítems</p>
          <p className="text-3xl text-[#1e293b]">24</p>
        </div>
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <TrendingUp className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">KMO</p>
          <p className="text-3xl text-[#1e293b]">0.92</p>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Análisis de Ítems</h3>
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((item) => (
            <div key={item} className="p-4 bg-gray-50 rounded-xl flex items-center justify-between">
              <span className="text-[#1e293b]">Ítem {item}</span>
              <div className="flex gap-6 text-sm">
                <div><span className="text-gray-500">r:</span> <span className="text-[#2c5f7c]">0.{75 + item}</span></div>
                <div><span className="text-gray-500">α sin ítem:</span> <span className="text-gray-700">0.{88 - item}</span></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
