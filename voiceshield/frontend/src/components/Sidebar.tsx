import React from 'react';
import {
  LayoutDashboard,
  ShieldAlert,
  Radio,
  History,
  Cpu,
  Settings,
  ShieldCheck,
} from 'lucide-react';

interface Props {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isBackendHealthy: boolean;
}

export const Sidebar: React.FC<Props> = ({
  activeTab,
  setActiveTab,
  isBackendHealthy,
}) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'analyze', label: 'Audio Analysis', icon: ShieldAlert },
    { id: 'stream', label: 'Live Call Monitor', icon: Radio },
    { id: 'history', label: 'Detection History', icon: History },
    { id: 'model', label: 'Model Architecture', icon: Cpu },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-[#0d121f] border-r border-slate-800 flex flex-col justify-between shrink-0 h-screen sticky top-0">
      <div>
        {/* App Brand Header */}
        <div className="h-16 px-6 flex items-center gap-3 border-b border-slate-800">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <ShieldCheck className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="font-bold tracking-tight text-white flex items-center gap-1.5">
              <span>VoiceShield</span>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.2 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                PRO
              </span>
            </div>
            <div className="text-[11px] text-slate-400 font-mono">
              VoiceGuard Engine
            </div>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1 mt-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30 shadow-sm shadow-blue-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border border-transparent'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-blue-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Backend Engine Status Footer */}
      <div className="p-4 border-t border-slate-800 bg-[#0a0e17]/60">
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="text-slate-400">Detection Engine</span>
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'
              }`}
            />
            <span className={isBackendHealthy ? 'text-emerald-400' : 'text-rose-400'}>
              {isBackendHealthy ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
        </div>
        <div className="text-[11px] text-slate-400 font-mono mt-1">
          AASIST-L • SincConv + GAT
        </div>
      </div>
    </aside>
  );
};
