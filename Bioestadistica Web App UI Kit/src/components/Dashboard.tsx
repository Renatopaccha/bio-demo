import { StatCard } from './StatCard';
import { TrendChart } from './TrendChart';
import { DistributionChart } from './DistributionChart';
import { RecentProjects } from './RecentProjects';
import { Users, Activity, FileBarChart, TrendingUp } from 'lucide-react';

export function Dashboard() {
  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl text-gray-900">Dashboard de Bioestadística</h1>
        <p className="text-gray-600">Bienvenido a tu espacio de análisis e investigación en salud</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Estudios Activos"
          value="12"
          change="+2 este mes"
          trend="up"
          icon={FileBarChart}
          color="blue"
        />
        <StatCard
          title="Pacientes Totales"
          value="1,248"
          change="+156 nuevos"
          trend="up"
          icon={Users}
          color="emerald"
        />
        <StatCard
          title="Pruebas Realizadas"
          value="3,847"
          change="+423 esta semana"
          trend="up"
          icon={Activity}
          color="purple"
        />
        <StatCard
          title="Tasa de Confianza"
          value="95.4%"
          change="+1.2% mejora"
          trend="up"
          icon={TrendingUp}
          color="cyan"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <TrendChart />
        <DistributionChart />
      </div>

      {/* Recent Projects */}
      <RecentProjects />
    </div>
  );
}
