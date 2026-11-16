import React from 'react';
import { Shield, Cpu, Activity, RefreshCw } from 'lucide-react';
import { HealthStatus } from '../types';

interface Props {
  title: string;
  subtitle: string;
  health: HealthStatus | null;
  onRefreshHealth: () => void;
}

export const Header: React.FC<Props> = ({
  title,
  subtitle,
  health,
  onRefreshHealth,
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-[#0d121f]/80 backdrop-blur-md px-8 flex items-center justify-between sticky top-0 z-30">
      <div>
        <h1 className="text-lg font-bold text-white tracking-tight">{title}</h1>
        <p className="text-xs text-slate-400">{subtitle}</p>
      </div>

      <div className="flex items-center gap-4 text-xs font-mono">
        {/* Device Pill */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-blue-400" />
          <span>Device: {health?.device?.toUpperCase() || 'CPU'}</span>
        </div>

        {/* Model Variant Pill */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300">
          <Shield className="w-3.5 h-3.5 text-indigo-400" />
          <span>Model: {health?.model_variant?.toUpperCase() || 'AASIST-L'}</span>
        </div>

        {/* Status indicator button */}
        <button
          onClick={onRefreshHealth}
          title="Refresh health status"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700/60 text-slate-300 transition-colors"
        >
          <Activity className="w-3.5 h-3.5 text-emerald-400" />
          <span>Status: {health?.status?.toUpperCase() || 'CONNECTING'}</span>
          <RefreshCw className="w-3 h-3 text-slate-400 ml-1 hover:rotate-180 transition-transform" />
        </button>
      </div>
    </header>
  );
};
