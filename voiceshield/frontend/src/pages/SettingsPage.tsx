import React, { useState } from 'react';
import {
  Settings,
  Sliders,
  Shield,
  Layers,
  Save,
  RotateCcw,
  CheckCircle,
} from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const [threshLow, setThreshLow] = useState(0.30);
  const [threshHigh, setThreshHigh] = useState(0.70);
  const [bufferSec, setBufferSec] = useState(6.0);
  const [hopSec, setHopSec] = useState(2.0);
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSave = () => {
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2500);
  };

  const handleReset = () => {
    setThreshLow(0.30);
    setThreshHigh(0.70);
    setBufferSec(6.0);
    setHopSec(2.0);
  };

  return (
    <div className="p-8 space-y-8 max-w-4xl mx-auto">
      {/* Title */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <Settings className="w-5 h-5 text-blue-400" />
          <span>Application &amp; Threshold Settings</span>
        </h2>
        <p className="text-xs text-slate-400">
          Tune decision boundaries, windowing hyperparameters, and real-time buffer sizes
        </p>
      </div>

      {/* Threshold Configuration Card */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-6">
        <div className="flex items-center gap-2 text-sm font-semibold text-white">
          <Sliders className="w-4 h-4 text-blue-400" />
          <span>Decision Boundary Thresholds</span>
        </div>

        <div className="space-y-6">
          {/* Low Threshold */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-300">Human Upper Threshold (Low Risk)</span>
              <span className="text-emerald-400 font-bold">{threshLow.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.10}
              max={0.45}
              step={0.05}
              value={threshLow}
              onChange={(e) => setThreshLow(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
            <p className="text-[11px] text-slate-500">
              Audio with model deepfake score strictly below this threshold is classified as <strong className="text-emerald-400">LIKELY HUMAN</strong>.
            </p>
          </div>

          {/* High Threshold */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-300">Synthetic Lower Threshold (High Risk)</span>
              <span className="text-rose-400 font-bold">{threshHigh.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.55}
              max={0.90}
              step={0.05}
              value={threshHigh}
              onChange={(e) => setThreshHigh(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
            />
            <p className="text-[11px] text-slate-500">
              Audio with model deepfake score at or above this threshold is classified as <strong className="text-rose-400">LIKELY AI-GENERATED</strong>.
            </p>
          </div>

          {/* Visual Zone Distribution */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="text-[11px] text-slate-400 font-mono">Classification Zone Preview:</div>
            <div className="h-4 w-full flex rounded-full overflow-hidden text-[10px] font-mono text-center leading-4 font-bold">
              <div
                style={{ width: `${threshLow * 100}%` }}
                className="bg-emerald-500/80 text-emerald-950"
              >
                Human
              </div>
              <div
                style={{ width: `${(threshHigh - threshLow) * 100}%` }}
                className="bg-amber-500/80 text-amber-950"
              >
                Uncertain
              </div>
              <div
                style={{ width: `${(1.0 - threshHigh) * 100}%` }}
                className="bg-rose-500/80 text-rose-950"
              >
                Synthetic
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Buffer & Hyperparameter Settings */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-6">
        <div className="flex items-center gap-2 text-sm font-semibold text-white">
          <Layers className="w-4 h-4 text-indigo-400" />
          <span>Streaming &amp; Windowing Hyperparameters</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="text-xs font-mono text-slate-300">
              Analysis Window Duration
            </label>
            <input
              type="text"
              disabled
              value="4.0375s (64,600 samples @ 16kHz)"
              className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs font-mono text-slate-400"
            />
            <p className="text-[11px] text-slate-500">
              Fixed by the SincConv filterbank and Graph Attention Network dimensions.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-mono text-slate-300">
              Window Hop Duration
            </label>
            <input
              type="number"
              min={1.0}
              max={4.0}
              step={0.5}
              value={hopSec}
              onChange={(e) => setHopSec(parseFloat(e.target.value))}
              className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs font-mono text-slate-200"
            />
            <p className="text-[11px] text-slate-500">
              Stride between successive analysis segments in seconds.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-mono text-slate-300">
              Streaming Accumulation Buffer (sec)
            </label>
            <input
              type="number"
              min={4.0}
              max={15.0}
              step={1.0}
              value={bufferSec}
              onChange={(e) => setBufferSec(parseFloat(e.target.value))}
              className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs font-mono text-slate-200"
            />
            <p className="text-[11px] text-slate-500">
              Target buffered duration before triggering streaming inference.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-mono text-slate-300">
              Streaming Context Overlap (sec)
            </label>
            <input
              type="text"
              disabled
              value="2.0s context retention"
              className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs font-mono text-slate-400"
            />
            <p className="text-[11px] text-slate-500">
              Maintains temporal continuity across sliding streaming evaluations.
            </p>
          </div>
        </div>
      </div>

      {/* Buttons */}
      <div className="flex items-center justify-between">
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors border border-slate-700"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Defaults</span>
        </button>

        <div className="flex items-center gap-3">
          {savedSuccess && (
            <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-mono">
              <CheckCircle className="w-4 h-4" />
              <span>Settings applied!</span>
            </span>
          )}
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-colors shadow-lg shadow-blue-600/20"
          >
            <Save className="w-4 h-4" />
            <span>Save Preferences</span>
          </button>
        </div>
      </div>
    </div>
  );
};
