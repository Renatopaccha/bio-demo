import { FileText, Download, Share2, Eye, Calendar } from 'lucide-react';

const reports = [
  {
    id: 1,
    title: 'Reporte Mensual - Hipertensión',
    date: '15 Dic 2025',
    type: 'Análisis Descriptivo',
    status: 'Completado',
    pages: 24,
    color: 'bg-blue-500',
  },
  {
    id: 2,
    title: 'Estudio Comparativo Oncología',
    date: '10 Dic 2025',
    type: 'Inferencial',
    status: 'Completado',
    pages: 38,
    color: 'bg-purple-500',
  },
  {
    id: 3,
    title: 'Análisis Longitudinal Diabetes',
    date: '5 Dic 2025',
    type: 'Supervivencia',
    status: 'En Revisión',
    pages: 45,
    color: 'bg-emerald-500',
  },
  {
    id: 4,
    title: 'Evaluación Tratamiento Cardiovascular',
    date: '1 Dic 2025',
    type: 'Ensayo Clínico',
    status: 'Completado',
    pages: 52,
    color: 'bg-cyan-500',
  },
];

export function Reports() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl text-[#1e293b] mb-2">Mi Reporte</h1>
          <p className="text-gray-500">Genera y gestiona reportes de tus investigaciones</p>
        </div>
        <button className="px-6 py-3 bg-gradient-to-r from-[#2c5f7c] to-[#4a9ebb] text-white rounded-xl hover:shadow-lg transition-all flex items-center gap-2">
          <FileText className="w-5 h-5" />
          Nuevo Reporte
        </button>
      </div>

      {/* Template Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-gray-200 hover:shadow-md transition-all cursor-pointer group">
          <div className="w-12 h-12 bg-[#4a9ebb]/10 rounded-xl flex items-center justify-center mb-4 group-hover:bg-[#4a9ebb]/20 transition-colors">
            <FileText className="w-6 h-6 text-[#2c5f7c]" />
          </div>
          <h3 className="text-[#1e293b] mb-2">Reporte Descriptivo</h3>
          <p className="text-sm text-gray-500 mb-4">Estadísticas descriptivas y visualizaciones básicas</p>
          <button className="text-sm text-[#2c5f7c] hover:text-[#234a61]">Usar plantilla →</button>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 hover:shadow-md transition-all cursor-pointer group">
          <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mb-4 group-hover:bg-purple-200 transition-colors">
            <FileText className="w-6 h-6 text-purple-600" />
          </div>
          <h3 className="text-[#1e293b] mb-2">Reporte Inferencial</h3>
          <p className="text-sm text-gray-500 mb-4">Pruebas de hipótesis y análisis comparativos</p>
          <button className="text-sm text-purple-600 hover:text-purple-700">Usar plantilla →</button>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-gray-200 hover:shadow-md transition-all cursor-pointer group">
          <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center mb-4 group-hover:bg-emerald-200 transition-colors">
            <FileText className="w-6 h-6 text-emerald-600" />
          </div>
          <h3 className="text-[#1e293b] mb-2">Reporte Completo</h3>
          <p className="text-sm text-gray-500 mb-4">Análisis integral para tesis y publicaciones</p>
          <button className="text-sm text-emerald-600 hover:text-emerald-700">Usar plantilla →</button>
        </div>
      </div>

      {/* Reports List */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-4">Reportes Generados</h3>
        <div className="space-y-3">
          {reports.map((report) => (
            <div 
              key={report.id} 
              className="flex items-center justify-between p-4 border border-gray-200 rounded-xl hover:shadow-md transition-all"
            >
              <div className="flex items-center gap-4 flex-1">
                <div className={`w-12 h-12 ${report.color} rounded-xl flex items-center justify-center`}>
                  <FileText className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1">
                  <h4 className="text-[#1e293b] mb-1">{report.title}</h4>
                  <div className="flex items-center gap-4 text-sm text-gray-500">
                    <div className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      <span>{report.date}</span>
                    </div>
                    <span>•</span>
                    <span>{report.type}</span>
                    <span>•</span>
                    <span>{report.pages} páginas</span>
                  </div>
                </div>
                <span className={`px-3 py-1 rounded-lg text-xs ${
                  report.status === 'Completado' 
                    ? 'bg-emerald-100 text-emerald-700' 
                    : 'bg-amber-100 text-amber-700'
                }`}>
                  {report.status}
                </span>
              </div>
              <div className="flex items-center gap-2 ml-4">
                <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors" title="Ver">
                  <Eye className="w-5 h-5" />
                </button>
                <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors" title="Descargar">
                  <Download className="w-5 h-5" />
                </button>
                <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors" title="Compartir">
                  <Share2 className="w-5 h-5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Export Options */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h3 className="text-lg text-[#1e293b] mb-4">Opciones de Exportación</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button className="p-4 border-2 border-gray-200 rounded-xl hover:border-[#4a9ebb] hover:bg-[#4a9ebb]/5 transition-all text-center">
            <div className="text-2xl mb-2">📄</div>
            <p className="text-sm text-[#1e293b]">PDF</p>
          </button>
          <button className="p-4 border-2 border-gray-200 rounded-xl hover:border-emerald-500 hover:bg-emerald-50 transition-all text-center">
            <div className="text-2xl mb-2">📊</div>
            <p className="text-sm text-[#1e293b]">Excel</p>
          </button>
          <button className="p-4 border-2 border-gray-200 rounded-xl hover:border-purple-500 hover:bg-purple-50 transition-all text-center">
            <div className="text-2xl mb-2">📝</div>
            <p className="text-sm text-[#1e293b]">Word</p>
          </button>
          <button className="p-4 border-2 border-gray-200 rounded-xl hover:border-cyan-500 hover:bg-cyan-50 transition-all text-center">
            <div className="text-2xl mb-2">📈</div>
            <p className="text-sm text-[#1e293b]">SPSS</p>
          </button>
        </div>
      </div>
    </div>
  );
}