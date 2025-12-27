import { ChevronRight, Bell, Settings } from 'lucide-react';

interface HeaderProps {
  activeSection: string;
}

const breadcrumbLabels: Record<string, string> = {
  home: 'Inicio',
  reports: 'Mi Reporte',
  ai: 'Asistente IA',
  cleaning: 'Limpieza de Datos',
  explorer: 'Modo Explorador',
  descriptive: 'Estadística Descriptiva',
  rates: 'Ajuste de Tasas',
  table1: 'Tabla 1 (Paper)',
  inference: 'Pruebas de Hipótesis',
  modeling: 'Modelos de Regresión',
  multivariate: 'Análisis Multivariado',
  survival: 'Análisis de Supervivencia',
  psychometrics: 'Psicometría',
  associations: 'Asociaciones',
  agreement: 'Concordancia',
  diagnostics: 'Diagnóstico (ROC)',
  graphics: 'Suite Gráfica',
};

export function Header({ activeSection }: HeaderProps) {
  return (
    <header className="h-20 bg-white border-b border-gray-100 flex items-center justify-between px-8">
      {/* Breadcrumbs */}
      <div className="flex items-center gap-2 text-sm">
        <span className="text-gray-400">Biometric</span>
        <ChevronRight className="w-4 h-4 text-gray-300" />
        <span className="text-[#1e293b]">{breadcrumbLabels[activeSection]}</span>
      </div>

      {/* Right Actions */}
      <div className="flex items-center gap-4">
        <button className="relative p-2 text-gray-500 hover:bg-gray-50 rounded-lg transition-colors">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-[#4a9ebb] rounded-full"></span>
        </button>
        
        <button className="p-2 text-gray-500 hover:bg-gray-50 rounded-lg transition-colors">
          <Settings className="w-5 h-5" />
        </button>

        <div className="w-px h-6 bg-gray-200"></div>

        {/* User Profile */}
        <div className="flex items-center gap-3 cursor-pointer hover:bg-gray-50 px-3 py-2 rounded-lg transition-colors">
          <div className="w-9 h-9 bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-full flex items-center justify-center">
            <span className="text-white text-sm">DR</span>
          </div>
          <div className="hidden md:block">
            <p className="text-sm text-[#1e293b]">Dr. Researcher</p>
            <p className="text-xs text-gray-500">researcher@university.edu</p>
          </div>
        </div>
      </div>
    </header>
  );
}
