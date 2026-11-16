import React from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  FileAudio,
  TrendingUp,
  ArrowRight,
  Clock,
  Zap,
} from 'lucide-react';
import { AnalysisResult } from '../types';
import { ClassificationBadge } from '../components/ClassificationBadge';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';

interface Props {
  history: AnalysisResult[];
  onNavigateToAnalyze: () => void;
  onNavigateToStream: () => void;
  onSelectResult: (result: AnalysisResult) => void;
}

export const OverviewPage: React.FC<Props> = ({
  history,
  onNavigateToAnalyze,
  onNavigateToStream,
  onSelectResult,
}) => {
  const totalAnalyzed = history.length;
  const humanCount = history.filter((h) => h.classification === 'LIKELY HUMAN').length;
  const aiCount = history.filter((h) => h.classification === 'LIKELY AI-GENERATED').length;
  const uncertainCount = history.filter((h) => h.classification === 'UNCERTAIN').length;

  const avgLatency =
    totalAnalyzed > 0
      ? (history.reduce((acc, h) => acc + h.processing_duration_ms, 0) / totalAnalyzed).toFixed(0)
      : '0';

  const chartData = [
    { name: 'Likely Human', value: humanCount, color: '#10b981' },
    { name: 'Uncertain', value: uncertainCount, color: '#f59e0b' },
    { name: 'Likely AI', value: aiCount, color: '#ef4444' },
  ].filter((d) => d.value > 0);

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Welcome Banner */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
              Active Defense
            </span>
            <span className="text-xs text-slate-400 font-mono">
              ASVspoof 2019 Evaluated
            </span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Enterprise Voice Deepfake Detection &amp; Protection
          </h2>
          <p className="text-sm text-slate-400 max-w-2xl">
            VoiceShield utilizes NAVER Clova AI's AASIST graph attention network to detect synthetic speech artifacts in audio recordings and live call streams.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={onNavigateToAnalyze}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold transition-colors shadow-lg shadow-blue-600/20"
          >
            <span>Analyze File</span>
            <ArrowRight className="w-4 h-4" />
          </button>
          <button
            onClick={onNavigateToStream}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium transition-colors border border-slate-700/60"
          >
            <Zap className="w-4 h-4 text-indigo-400" />
            <span>Call Monitor</span>
          </button>
        </div>
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Analyzed */}
        <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-mono uppercase tracking-wider">Total Analyzed</span>
            <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400">
              <FileAudio className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold font-mono text-white tracking-tight">
            {totalAnalyzed}
          </div>
          <div className="text-xs text-slate-400">Recordings inspected</div>
        </div>

        {/* Likely Human */}
        <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-mono uppercase tracking-wider">Likely Human</span>
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold font-mono text-emerald-400 tracking-tight">
            {humanCount}
          </div>
          <div className="text-xs text-slate-400">
            {totalAnalyzed > 0 ? ((humanCount / totalAnalyzed) * 100).toFixed(0) : 0}% of total volume
          </div>
        </div>

        {/* Likely AI-Generated */}
        <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-mono uppercase tracking-wider">AI-Generated</span>
            <div className="w-8 h-8 rounded-lg bg-rose-500/10 flex items-center justify-center text-rose-400">
              <ShieldAlert className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold font-mono text-rose-400 tracking-tight">
            {aiCount}
          </div>
          <div className="text-xs text-slate-400">
            {totalAnalyzed > 0 ? ((aiCount / totalAnalyzed) * 100).toFixed(0) : 0}% synthetic artifacts detected
          </div>
        </div>

        {/* Uncertain */}
        <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-mono uppercase tracking-wider">Uncertain</span>
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center text-amber-400">
              <AlertTriangle className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold font-mono text-amber-400 tracking-tight">
            {uncertainCount}
          </div>
          <div className="text-xs text-slate-400">Requires manual acoustic review</div>
        </div>
      </div>

      {/* Analytics & Recent Activity Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Distribution Chart */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white tracking-tight">
              Classification Breakdown
            </h3>
            <span className="text-xs text-slate-400 font-mono">Live Session</span>
          </div>

          {totalAnalyzed === 0 ? (
            <div className="h-56 flex flex-col items-center justify-center text-center p-4">
              <FileAudio className="w-8 h-8 text-slate-600 mb-2" />
              <p className="text-xs text-slate-400 font-mono">No audio analyses yet</p>
              <p className="text-xs text-slate-400 mt-1">Upload a recording to populate breakdown metrics</p>
            </div>
          ) : (
            <div className="h-56 w-full flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={75}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#111726',
                      borderColor: '#1e293b',
                      borderRadius: '8px',
                      fontSize: '12px',
                      fontFamily: 'monospace',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800/80 text-center text-xs font-mono">
            <div>
              <div className="text-emerald-400 font-bold">{humanCount}</div>
              <div className="text-slate-400 text-[10px]">Human</div>
            </div>
            <div>
              <div className="text-amber-400 font-bold">{uncertainCount}</div>
              <div className="text-slate-400 text-[10px]">Uncertain</div>
            </div>
            <div>
              <div className="text-rose-400 font-bold">{aiCount}</div>
              <div className="text-slate-400 text-[10px]">Synthetic</div>
            </div>
          </div>
        </div>

        {/* Recent Analysis Table */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white tracking-tight">
                Recent Analysis Activity
              </h3>
              <p className="text-xs text-slate-400">Last processed audio files</p>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono">
              <Clock className="w-3.5 h-3.5" />
              <span>Avg latency: {avgLatency}ms</span>
            </div>
          </div>

          {totalAnalyzed === 0 ? (
            <div className="h-56 flex flex-col items-center justify-center text-center p-4">
              <FileAudio className="w-8 h-8 text-slate-600 mb-2" />
              <p className="text-xs text-slate-400 font-mono">No recent audio analyzed</p>
              <button
                onClick={onNavigateToAnalyze}
                className="mt-3 text-xs text-blue-400 hover:text-blue-300 underline font-mono"
              >
                Upload your first audio file &rarr;
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-mono">
                    <th className="pb-2 font-medium">Recording</th>
                    <th className="pb-2 font-medium">Classification</th>
                    <th className="pb-2 font-medium">Score</th>
                    <th className="pb-2 font-medium">Duration</th>
                    <th className="pb-2 font-medium text-right">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {history.slice(0, 5).map((item) => (
                    <tr
                      key={item.id}
                      onClick={() => onSelectResult(item)}
                      className="hover:bg-slate-800/40 cursor-pointer transition-colors"
                    >
                      <td className="py-2.5 text-slate-200 font-sans font-medium truncate max-w-[180px]">
                        {item.filename}
                      </td>
                      <td className="py-2.5">
                        <ClassificationBadge state={item.classification} size="sm" />
                      </td>
                      <td className="py-2.5 text-slate-300">
                        {(item.model_score * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 text-slate-400">
                        {item.audio_metadata.duration_sec.toFixed(1)}s
                      </td>
                      <td className="py-2.5 text-right text-slate-400">
                        {item.processing_duration_ms.toFixed(0)}ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
