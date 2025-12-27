import { Link, TrendingUp } from 'lucide-react';

export function Associations() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Medidas de Asociación</h1>
        <p className="text-gray-500">Odds Ratio, Riesgo Relativo y más</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-2xl p-6 text-white">
          <Link className="w-8 h-8 mb-3" />
          <p className="text-sm text-blue-100 mb-1">Odds Ratio</p>
          <p className="text-3xl mb-1">2.45</p>
          <p className="text-sm text-blue-100">IC 95%: [1.82, 3.29]</p>
        </div>
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <TrendingUp className="w-8 h-8 text-[#2c5f7c] mb-3" />
          <p className="text-sm text-gray-500 mb-1">Riesgo Relativo</p>
          <p className="text-3xl text-[#1e293b] mb-1">1.85</p>
          <p className="text-sm text-gray-500">IC 95%: [1.52, 2.25]</p>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Tabla 2x2</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="p-4"></th>
                <th className="p-4 text-center text-gray-700">Expuesto</th>
                <th className="p-4 text-center text-gray-700">No Expuesto</th>
                <th className="p-4 text-center text-gray-700">Total</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-100">
                <td className="p-4 text-gray-700">Casos</td>
                <td className="p-4 text-center text-[#1e293b]">145</td>
                <td className="p-4 text-center text-[#1e293b]">78</td>
                <td className="p-4 text-center text-gray-600">223</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="p-4 text-gray-700">Controles</td>
                <td className="p-4 text-center text-[#1e293b]">112</td>
                <td className="p-4 text-center text-[#1e293b]">298</td>
                <td className="p-4 text-center text-gray-600">410</td>
              </tr>
              <tr>
                <td className="p-4 text-gray-700">Total</td>
                <td className="p-4 text-center text-gray-600">257</td>
                <td className="p-4 text-center text-gray-600">376</td>
                <td className="p-4 text-center text-[#2c5f7c]">633</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
