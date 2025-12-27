import { Upload, FileSpreadsheet, X, CheckCircle, AlertTriangle, Trash2, Download } from 'lucide-react';
import { useState } from 'react';
import { uploadFile } from '../services/api';

export function DataCleaning() {
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setUploadedFile('datos_clinicos.xlsx');
  };

  const removeFile = () => {
    setUploadedFile(null);
  };

  const dataQualityIssues = [
    { type: 'Valores faltantes', count: 45, severity: 'medium', column: 'Edad, Peso' },
    { type: 'Valores duplicados', count: 12, severity: 'high', column: 'ID_Paciente' },
    { type: 'Valores atípicos', count: 8, severity: 'low', column: 'Presión_Arterial' },
    { type: 'Formato inconsistente', count: 23, severity: 'medium', column: 'Fecha_Nacimiento' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl text-[#1e293b] mb-2">Limpieza de Datos</h1>
        <p className="text-gray-500">Carga y prepara tus datos para el análisis estadístico</p>
      </div>

      {/* File Upload */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
        <h3 className="text-lg text-[#1e293b] mb-6">Cargar Dataset</h3>

        {!uploadedFile ? (
          <>
            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".xlsx,.csv"
              onChange={async (e) => {
                if (e.target.files && e.target.files[0]) {
                  const file = e.target.files[0];
                  try {
                    // Import dynamically or assume it's imported at top (I will add import via another edit if needed, or assume I can add it here)
                    // Better to add import at top first. 
                    // I'll assume I can edit the whole file or I'll add the import in a previous step?
                    // No, I can replace the whole functional component content effectively or use multiple chunks.
                    // I will strictly follow the tool usage.
                    // Let's use the 'api' service. I need to import it.
                    // For now I will write the logic assuming 'uploadFile' is available in scope, 
                    // but I must add the import statement at the top of the file as well.
                    // I will DO THAT in this same tool call using a separate chunk? 
                    // No, "Do NOT use this tool if you are only editing a single contiguous block".
                    // But I need to edit imports AND the component body. So I must use `multi_replace_file_content`.
                  } catch (err) {
                    console.error(err);
                    alert("Error uploading file");
                  }
                }
              }}
            />
            {/* ... */}
          </>
        ) : null}
        {/* ... */}
      </div>

      {uploadedFile && (
        <>
          {/* Data Quality Issues */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <h3 className="text-lg text-[#1e293b] mb-6">Problemas de Calidad Detectados</h3>

            <div className="space-y-3">
              {dataQualityIssues.map((issue, index) => (
                <div key={index} className="p-4 bg-gray-50 rounded-xl border border-gray-100 flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${issue.severity === 'high' ? 'bg-red-100' :
                      issue.severity === 'medium' ? 'bg-orange-100' :
                        'bg-yellow-100'
                      }`}>
                      <AlertTriangle className={`w-5 h-5 ${issue.severity === 'high' ? 'text-red-600' :
                        issue.severity === 'medium' ? 'text-orange-600' :
                          'text-yellow-600'
                        }`} />
                    </div>

                    <div>
                      <p className="text-[#1e293b]">{issue.type}</p>
                      <p className="text-sm text-gray-500">{issue.count} casos en: {issue.column}</p>
                    </div>
                  </div>

                  <button className="px-4 py-2 bg-[#2c5f7c] text-white rounded-lg hover:bg-[#234a61] transition-colors text-sm">
                    Corregir
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Cleaning Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <button className="p-6 bg-white rounded-2xl border border-gray-200 hover:shadow-md transition-all text-left group">
              <div className="w-12 h-12 bg-[#4a9ebb]/10 rounded-xl flex items-center justify-center mb-4 group-hover:bg-[#4a9ebb]/20 transition-colors">
                <Trash2 className="w-6 h-6 text-[#2c5f7c]" />
              </div>
              <h4 className="text-[#1e293b] mb-2">Eliminar Duplicados</h4>
              <p className="text-sm text-gray-500">Remover filas duplicadas automáticamente</p>
            </button>

            <button className="p-6 bg-white rounded-2xl border border-gray-200 hover:shadow-md transition-all text-left group">
              <div className="w-12 h-12 bg-[#4a9ebb]/10 rounded-xl flex items-center justify-center mb-4 group-hover:bg-[#4a9ebb]/20 transition-colors">
                <CheckCircle className="w-6 h-6 text-[#2c5f7c]" />
              </div>
              <h4 className="text-[#1e293b] mb-2">Imputar Valores</h4>
              <p className="text-sm text-gray-500">Rellenar valores faltantes con media/mediana</p>
            </button>

            <button className="p-6 bg-white rounded-2xl border border-gray-200 hover:shadow-md transition-all text-left group">
              <div className="w-12 h-12 bg-[#4a9ebb]/10 rounded-xl flex items-center justify-center mb-4 group-hover:bg-[#4a9ebb]/20 transition-colors">
                <Download className="w-6 h-6 text-[#2c5f7c]" />
              </div>
              <h4 className="text-[#1e293b] mb-2">Exportar Limpio</h4>
              <p className="text-sm text-gray-500">Descargar dataset procesado</p>
            </button>
          </div>

          {/* Action Button */}
          <div className="flex justify-end">
            <button className="px-6 py-3 bg-gradient-to-r from-[#2c5f7c] to-[#4a9ebb] text-white rounded-xl hover:shadow-lg transition-all">
              Continuar al Análisis
            </button>
          </div>
        </>
      )}
    </div>
  );
}
