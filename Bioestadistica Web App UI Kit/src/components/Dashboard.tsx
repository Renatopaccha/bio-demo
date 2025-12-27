import { StatCard } from './StatCard';
import { TrendChart } from './TrendChart';
import { DistributionChart } from './DistributionChart';
import { RecentProjects } from './RecentProjects';
import { Users, Activity, FileBarChart, TrendingUp, Upload as UploadIcon } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { uploadFile } from '../services/api';

export function Dashboard() {
  const [stats, setStats] = useState<{ rows: number; columns: number; filename: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Restore session if available
    const savedSession = localStorage.getItem('session_id');
    const savedRows = localStorage.getItem('session_rows');
    const savedFilename = localStorage.getItem('session_filename');

    if (savedSession && savedRows) {
      setStats({
        rows: parseInt(savedRows),
        columns: 0, // We might not have saved columns count, default to 0 or ignore
        filename: savedFilename || 'Dataset Actual'
      });
    }
  }, []);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    try {
      const response = await uploadFile(file);

      // Save to localStorage
      localStorage.setItem('session_id', response.session_id);
      localStorage.setItem('session_rows', response.rows.toString());
      localStorage.setItem('session_filename', response.filename);

      setStats({
        rows: response.rows,
        columns: response.columns.length,
        filename: response.filename
      });
    } catch (error) {
      console.error("Upload failed", error);
      alert("Error al subir el archivo");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 space-y-6">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept=".xlsx,.xls,.csv"
      />

      {/* Header */}
      <div className="flex justify-between items-start">
        <div className="space-y-2">
          <h1 className="text-3xl text-gray-900">Dashboard de Bioestadística</h1>
          <p className="text-gray-600">Bienvenido a tu espacio de análisis e investigación en salud</p>
        </div>
        <button
          onClick={handleUploadClick}
          disabled={loading}
          className="flex items-center gap-2 bg-[#2c5f7c] text-white px-4 py-2 rounded-lg hover:bg-[#3a7ca0] transition-colors disabled:opacity-50"
        >
          <UploadIcon className="w-4 h-4" />
          {loading ? 'Subiendo...' : 'Nuevo Proyecto / Upload'}
        </button>
      </div>

      {!stats ? (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-8 text-center">
          <h3 className="text-xl text-blue-900 mb-2">Comienza tu Análisis</h3>
          <p className="text-blue-700 mb-4">Sube un dataset para ver estadísticas en tiempo real.</p>
          <button
            onClick={handleUploadClick}
            className="text-blue-600 font-medium hover:underline"
          >
            Cargar archivo ahora &rarr;
          </button>
        </div>
      ) : (
        <>
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatCard
              title="Estudios Activos"
              value="1"
              change="Activo ahora"
              trend="up"
              icon={FileBarChart}
              color="blue"
            />
            <StatCard
              title="Total de Muestras"
              value={stats.rows.toLocaleString()}
              change={`En ${stats.filename}`}
              trend="neutral"
              icon={Users}
              color="emerald"
            />
            <StatCard
              title="Variables"
              value={stats.columns > 0 ? stats.columns.toString() : "-"}
              change="Columnas detectadas"
              trend="neutral"
              icon={Activity}
              color="purple"
            />
            <StatCard
              title="Estado de Datos"
              value="Listo"
              change="Procesado con éxito"
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
        </>
      )}
    </div>
  );
}
