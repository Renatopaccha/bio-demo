import { LucideIcon, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string;
  change: string;
  trend: 'up' | 'down' | 'neutral';
  icon: LucideIcon;
  color: 'blue' | 'emerald' | 'purple' | 'cyan' | 'rose';
}

const colorClasses = {
  blue: 'bg-blue-500',
  emerald: 'bg-emerald-500',
  purple: 'bg-purple-500',
  cyan: 'bg-cyan-500',
  rose: 'bg-rose-500',
};

export function StatCard({ title, value, change, trend, icon: Icon, color }: StatCardProps) {
  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-gray-600 text-sm mb-2">{title}</p>
          <p className="text-3xl text-gray-900 mb-2">{value}</p>
          <div className="flex items-center gap-1 text-sm">
            {trend === 'up' && <TrendingUp className="w-4 h-4 text-emerald-600" />}
            {trend === 'down' && <TrendingDown className="w-4 h-4 text-rose-600" />}
            {trend === 'neutral' && <Minus className="w-4 h-4 text-gray-400" />}

            <span className={
              trend === 'up' ? 'text-emerald-600' :
                trend === 'down' ? 'text-rose-600' : 'text-gray-500'
            }>
              {change}
            </span>
          </div>
        </div>
        <div className={`w-12 h-12 ${colorClasses[color]} rounded-lg flex items-center justify-center`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  );
}
