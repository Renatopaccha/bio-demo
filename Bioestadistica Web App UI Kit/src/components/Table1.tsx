import { FileText, Download, Copy, Settings } from 'lucide-react';

export function Table1() {
  const variables = [
    { name: 'Edad (años)', overall: '52.3 ± 12.4', group1: '48.5 ± 11.2', group2: '56.1 ± 13.1', pValue: '0.012', test: 't-test' },
    { name: 'Sexo Masculino, n (%)', overall: '645 (51.7)', group1: '312 (49.8)', group2: '333 (53.5)', pValue: '0.156', test: 'χ²' },
    { name: 'IMC (kg/m²)', overall: '27.8 ± 4.6', group1: '26.2 ± 4.1', group2: '29.4 ± 4.9', pValue: '<0.001', test: 't-test' },
    { name: 'Hipertensión, n (%)', overall: '456 (36.5)', group1: '198 (31.6)', group2: '258 (41.5)', pValue: '0.001', test: 'χ²' },
    { name: 'Diabetes, n (%)', overall: '234 (18.8)', group1: '98 (15.6)', group2: '136 (21.9)', pValue: '0.005', test: 'χ²' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Tabla 1 (Paper Académico)</h1>
        <p className="text-gray-500">Genera tablas de características basales para publicaciones científicas</p>
      </div>

      {/* Configuration */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-6">Configuración de la Tabla</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm text-gray-700 mb-2">Variable de Agrupación</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb] focus:border-transparent">
              <option>Grupo de Tratamiento</option>
              <option>Sexo</option>
              <option>Estado de Enfermedad</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-700 mb-2">Estadísticas para Continuas</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb] focus:border-transparent">
              <option>Media ± DE</option>
              <option>Mediana [RIC]</option>
              <option>Ambos</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-700 mb-2">Formato de Exportación</label>
            <select className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#4a9ebb] focus:border-transparent">
              <option>Word (.docx)</option>
              <option>Excel (.xlsx)</option>
              <option>LaTeX</option>
              <option>HTML</option>
            </select>
          </div>
        </div>
      </div>

      {/* Generated Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg text-[#1e293b]">Vista Previa de Tabla 1</h3>
          
          <div className="flex gap-2">
            <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors" title="Copiar">
              <Copy className="w-5 h-5" />
            </button>
            <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors" title="Configurar">
              <Settings className="w-5 h-5" />
            </button>
            <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors" title="Descargar">
              <Download className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="text-left px-4 py-3 text-sm text-gray-700">Característica</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Total<br/>(n=1,248)</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Grupo Control<br/>(n=627)</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">Grupo Tratamiento<br/>(n=621)</th>
                <th className="text-center px-4 py-3 text-sm text-gray-700">p-valor</th>
              </tr>
            </thead>
            <tbody>
              {variables.map((variable, index) => (
                <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-[#1e293b]">{variable.name}</td>
                  <td className="px-4 py-3 text-sm text-gray-700 text-center">{variable.overall}</td>
                  <td className="px-4 py-3 text-sm text-gray-700 text-center">{variable.group1}</td>
                  <td className="px-4 py-3 text-sm text-gray-700 text-center">{variable.group2}</td>
                  <td className="px-4 py-3 text-center">
                    <span className={`text-sm ${
                      parseFloat(variable.pValue.replace('<', '')) < 0.05 
                        ? 'text-[#2c5f7c]' 
                        : 'text-gray-600'
                    }`}>
                      {variable.pValue}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 p-4 bg-gray-50 rounded-xl">
          <p className="text-xs text-gray-600">
            <strong>Nota:</strong> Los valores se presentan como media ± desviación estándar para variables continuas y n (%) para variables categóricas. 
            Las comparaciones se realizaron mediante prueba t para variables continuas y χ² para categóricas.
          </p>
        </div>
      </div>

      {/* Export Options */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <button className="p-4 bg-white rounded-xl border border-gray-200 hover:shadow-md transition-all text-center">
          <FileText className="w-8 h-8 text-[#2c5f7c] mx-auto mb-2" />
          <p className="text-sm text-[#1e293b]">Word</p>
        </button>
        <button className="p-4 bg-white rounded-xl border border-gray-200 hover:shadow-md transition-all text-center">
          <FileText className="w-8 h-8 text-[#2c5f7c] mx-auto mb-2" />
          <p className="text-sm text-[#1e293b]">Excel</p>
        </button>
        <button className="p-4 bg-white rounded-xl border border-gray-200 hover:shadow-md transition-all text-center">
          <FileText className="w-8 h-8 text-[#2c5f7c] mx-auto mb-2" />
          <p className="text-sm text-[#1e293b]">LaTeX</p>
        </button>
        <button className="p-4 bg-white rounded-xl border border-gray-200 hover:shadow-md transition-all text-center">
          <FileText className="w-8 h-8 text-[#2c5f7c] mx-auto mb-2" />
          <p className="text-sm text-[#1e293b]">HTML</p>
        </button>
      </div>
    </div>
  );
}
