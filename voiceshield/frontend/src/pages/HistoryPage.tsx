import React, { useState } from 'react';
import {
  History,
  Trash2,
  Download,
  Search,
  Filter,
  FileAudio,
  Clock,
  ArrowUpDown,
} from 'lucide-react';
import { AnalysisResult } from '../types';
import { clearHistory } from '../services/api';
import { ClassificationBadge } from '../components/ClassificationBadge';

interface Props {
  history: AnalysisResult[];
  onHistoryCleared: () => void;
  onSelectResult: (result: AnalysisResult) => void;
}

export const HistoryPage: React.FC<Props> = ({
  history,
  onHistoryCleared,
  onSelectResult,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterState, setFilterState] = useState<string>('ALL');

  const filtered = history.filter((item) => {
    const matchesSearch = item.filename.toLowerCase().includes(searchQuery.toLowerCase());
    if (filterState === 'ALL') return matchesSearch;
    return matchesSearch && item.classification === filterState;
  });

  const handleClearHistory = async () => {
    if (window.confirm('Are you sure you want to delete all local analysis history?')) {
      try {
        await clearHistory();
        onHistoryCleared();
      } catch (e) {
        console.error('Failed to clear history:', e);
      }
    }
  };

  const handleExportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(history, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute('href', dataStr);
    dlAnchorElem.setAttribute('download', `voiceshield_history_${new Date().toISOString().slice(0, 10)}.json`);
    dlAnchorElem.click();
  };

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <History className="w-5 h-5 text-blue-400" />
            <span>Detection History</span>
          </h2>
          <p className="text-xs text-slate-400">
            Audit log of all analyzed recordings in the current session ({history.length} records)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleExportJSON}
            disabled={history.length === 0}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 text-xs font-medium transition-colors border border-slate-700"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export JSON</span>
          </button>
          <button
            onClick={handleClearHistory}
            disabled={history.length === 0}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 disabled:opacity-40 text-rose-400 text-xs font-medium transition-colors border border-rose-500/20"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear History</span>
          </button>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="glass-panel rounded-2xl p-4 border border-slate-800 flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Search */}
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-900 border border-slate-700/80 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* State Filter Buttons */}
        <div className="flex items-center gap-2 text-xs font-mono">
          {['ALL', 'LIKELY HUMAN', 'UNCERTAIN', 'LIKELY AI-GENERATED'].map((st) => (
            <button
              key={st}
              onClick={() => setFilterState(st)}
              className={`px-3 py-1.5 rounded-lg border transition-colors ${
                filterState === st
                  ? 'bg-blue-600/20 border-blue-500/40 text-blue-400'
                  : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {/* History Table */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
        {filtered.length === 0 ? (
          <div className="h-64 flex flex-col items-center justify-center text-center p-6 space-y-2">
            <FileAudio className="w-8 h-8 text-slate-600" />
            <p className="text-xs text-slate-400 font-mono">No matching detection records found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/40 text-slate-400 uppercase text-[10px]">
                  <th className="py-3 px-5 font-semibold">ID</th>
                  <th className="py-3 px-5 font-semibold">Recording</th>
                  <th className="py-3 px-5 font-semibold">Classification</th>
                  <th className="py-3 px-5 font-semibold">Score</th>
                  <th className="py-3 px-5 font-semibold">Certainty</th>
                  <th className="py-3 px-5 font-semibold">Duration</th>
                  <th className="py-3 px-5 font-semibold">Latency</th>
                  <th className="py-3 px-5 font-semibold text-right">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {filtered.map((item) => (
                  <tr
                    key={item.id}
                    onClick={() => onSelectResult(item)}
                    className="hover:bg-slate-800/30 cursor-pointer transition-colors"
                  >
                    <td className="py-3.5 px-5 text-blue-400 font-semibold">{item.id}</td>
                    <td className="py-3.5 px-5 font-sans font-medium text-slate-200 truncate max-w-[200px]">
                      {item.filename}
                    </td>
                    <td className="py-3.5 px-5">
                      <ClassificationBadge state={item.classification} size="sm" />
                    </td>
                    <td className="py-3.5 px-5 font-bold">
                      {(item.model_score * 100).toFixed(1)}%
                    </td>
                    <td className="py-3.5 px-5 text-slate-400">
                      {(item.confidence * 100).toFixed(0)}%
                    </td>
                    <td className="py-3.5 px-5 text-slate-400">
                      {item.audio_metadata.duration_sec.toFixed(1)}s
                    </td>
                    <td className="py-3.5 px-5 text-slate-400">
                      {item.processing_duration_ms.toFixed(0)}ms
                    </td>
                    <td className="py-3.5 px-5 text-right text-slate-500 text-[11px]">
                      {new Date(item.created_at).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
