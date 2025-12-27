import { Home, FileText, Sparkles, Brush, Database, BarChart2, Scale, FileSpreadsheet, FlaskConical, TrendingUp, Network, Activity, Brain, Link, CheckCircle, Target, Palette, ChevronLeft, ChevronRight } from 'lucide-react';

interface SidebarProps {
  activeSection: string;
  setActiveSection: (section: string) => void;
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
}

export function Sidebar({ activeSection, setActiveSection, collapsed, setCollapsed }: SidebarProps) {
  const menuItems = [
    { id: 'home', label: 'Inicio', icon: Home },
    { id: 'reports', label: 'Mi Reporte', icon: FileText },
    { id: 'ai', label: 'Asistente IA', icon: Sparkles },
    { id: 'cleaning', label: 'Limpieza de Datos', icon: Brush },
    { id: 'explorer', label: 'Modo Explorador', icon: Database },
    { id: 'descriptive', label: 'Estadística Descriptiva', icon: BarChart2 },
    { id: 'rates', label: 'Ajuste de Tasas', icon: Scale },
    { id: 'table1', label: 'Tabla 1 (Paper)', icon: FileSpreadsheet },
    { id: 'inference', label: 'Pruebas de Hipótesis', icon: FlaskConical },
    { id: 'modeling', label: 'Modelos de Regresión', icon: TrendingUp },
    { id: 'multivariate', label: 'Análisis Multivariado', icon: Network },
    { id: 'survival', label: 'Análisis de Supervivencia', icon: Activity },
    { id: 'psychometrics', label: 'Psicometría', icon: Brain },
    { id: 'associations', label: 'Asociaciones', icon: Link },
    { id: 'agreement', label: 'Concordancia', icon: CheckCircle },
    { id: 'diagnostics', label: 'Diagnóstico (ROC)', icon: Target },
    { id: 'graphics', label: 'Suite Gráfica', icon: Palette },
  ];

  return (
    <aside 
      className={`fixed left-0 top-0 h-screen bg-white border-r border-gray-200 transition-all duration-300 z-40 ${
        collapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Logo */}
      <div className="h-20 flex items-center px-6 border-b border-gray-100">
        {!collapsed ? (
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-xl flex items-center justify-center">
              <span className="text-white text-xl">Σ</span>
            </div>
            <div>
              <h1 className="text-lg text-[#1e293b]">Biometric</h1>
              <p className="text-xs text-gray-500">Tesis Simplificada</p>
            </div>
          </div>
        ) : (
          <div className="w-10 h-10 bg-gradient-to-br from-[#2c5f7c] to-[#4a9ebb] rounded-xl flex items-center justify-center mx-auto">
            <span className="text-white text-xl">Σ</span>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="px-3 py-4 space-y-1 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 140px)' }}>
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeSection === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => setActiveSection(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all relative group ${
                isActive
                  ? 'bg-gradient-to-r from-[#2c5f7c] to-[#4a9ebb] text-white shadow-lg shadow-[#2c5f7c]/20'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
              title={collapsed ? item.label : undefined}
            >
              <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-white' : 'text-gray-500'}`} />
              {!collapsed && <span className="truncate text-sm">{item.label}</span>}
              
              {/* Tooltip for collapsed state */}
              {collapsed && (
                <div className="absolute left-full ml-2 px-3 py-2 bg-gray-900 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap transition-opacity z-50">
                  {item.label}
                </div>
              )}
            </button>
          );
        })}
      </nav>

      {/* Collapse Toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute bottom-6 right-0 transform translate-x-1/2 w-8 h-8 bg-white border border-gray-200 rounded-full flex items-center justify-center hover:shadow-md transition-all"
      >
        {collapsed ? (
          <ChevronRight className="w-4 h-4 text-gray-600" />
        ) : (
          <ChevronLeft className="w-4 h-4 text-gray-600" />
        )}
      </button>
    </aside>
  );
}
