import { CheckCircle, Users } from 'lucide-react';

export function Agreement() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Análisis de Concordancia</h1>
        <p className="text-gray-500">Kappa, CCI y Bland-Altman</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white">
          <CheckCircle className="w-8 h-8 mb-3" />
          <p className="text-sm text-blue-100 mb-1">Kappa de Cohen</p>
          <p className="text-3xl">0.82</p>
          <p className="text-sm text-blue-100 mt-2">Acuerdo sustancial</p>
        </div>
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <Users className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">CCI</p>
          <p className="text-3xl text-[#1e293b]">0.91</p>
          <p className="text-sm text-gray-500 mt-2">IC 95%: [0.87, 0.94]</p>
        </div>
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <CheckCircle className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">% Acuerdo</p>
          <p className="text-3xl text-[#1e293b]">89.5%</p>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Tabla de Concordancia</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="p-4"></th>
                <th className="p-4 text-center text-gray-700">Observador 2: Positivo</th>
                <th className="p-4 text-center text-gray-700">Observador 2: Negativo</th>
                <th className="p-4 text-center text-gray-700">Total</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-100">
                <td className="p-4 text-gray-700">Observador 1: Positivo</td>
                <td className="p-4 text-center bg-emerald-50 text-emerald-700">156</td>
                <td className="p-4 text-center bg-red-50 text-red-700">12</td>
                <td className="p-4 text-center text-gray-600">168</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="p-4 text-gray-700">Observador 1: Negativo</td>
                <td className="p-4 text-center bg-red-50 text-red-700">18</td>
                <td className="p-4 text-center bg-emerald-50 text-emerald-700">234</td>
                <td className="p-4 text-center text-gray-600">252</td>
              </tr>
              <tr>
                <td className="p-4 text-gray-700">Total</td>
                <td className="p-4 text-center text-gray-600">174</td>
                <td className="p-4 text-center text-gray-600">246</td>
                <td className="p-4 text-center text-[#2c5f7c]">420</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
