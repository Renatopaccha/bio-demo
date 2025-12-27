import { Clock, Users, TrendingUp } from 'lucide-react';

const projects = [
  {
    id: 1,
    title: 'Estudio de Hipertensión en Adultos Mayores',
    status: 'En Progreso',
    participants: 156,
    completion: 67,
    lastUpdate: 'Hace 2 horas',
    statusColor: 'bg-blue-500',
  },
  {
    id: 2,
    title: 'Efectividad de Tratamiento Oncológico',
    status: 'Análisis',
    participants: 89,
    completion: 85,
    lastUpdate: 'Hace 5 horas',
    statusColor: 'bg-purple-500',
  },
  {
    id: 3,
    title: 'Factores de Riesgo Cardiovascular',
    status: 'Recolección',
    participants: 234,
    completion: 42,
    lastUpdate: 'Hace 1 día',
    statusColor: 'bg-emerald-500',
  },
  {
    id: 4,
    title: 'Diabetes Tipo 2 y Nutrición',
    status: 'En Progreso',
    participants: 178,
    completion: 58,
    lastUpdate: 'Hace 3 horas',
    statusColor: 'bg-cyan-500',
  },
];

export function RecentProjects() {
  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
      <div className="mb-6">
        <h3 className="text-lg text-gray-900">Proyectos Recientes</h3>
        <p className="text-sm text-gray-600">Tus estudios activos e investigaciones en curso</p>
      </div>
      
      <div className="space-y-4">
        {projects.map((project) => (
          <div key={project.id} className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <h4 className="text-gray-900 mb-1">{project.title}</h4>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <div className="flex items-center gap-1">
                    <Users className="w-4 h-4" />
                    <span>{project.participants} participantes</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    <span>{project.lastUpdate}</span>
                  </div>
                </div>
              </div>
              <span className={`px-3 py-1 rounded-full text-xs text-white ${project.statusColor}`}>
                {project.status}
              </span>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Progreso</span>
                <span className="text-gray-900">{project.completion}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${project.statusColor}`}
                  style={{ width: `${project.completion}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
