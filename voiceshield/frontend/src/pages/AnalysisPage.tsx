import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileAudio,
  Trash2,
  AlertCircle,
  PlayCircle,
  Cpu,
  Layers,
  Activity,
  CheckCircle2,
  HelpCircle,
  Info,
} from 'lucide-react';
import { AnalysisResult } from '../types';
import { analyzeAudioFile, fetchWaveformPeaks } from '../services/api';
import { ClassificationBadge } from '../components/ClassificationBadge';
import { ScoreGauge } from '../components/ScoreGauge';
import { AudioPlayer } from '../components/AudioPlayer';
import { WaveformVisualizer } from '../components/WaveformVisualizer';

interface Props {
  currentResult: AnalysisResult | null;
  setCurrentResult: (result: AnalysisResult | null) => void;
  onAnalysisComplete: (result: AnalysisResult) => void;
}

export const AnalysisPage: React.FC<Props> = ({
  currentResult,
  setCurrentResult,
  onAnalysisComplete,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [waveformPeaks, setWaveformPeaks] = useState<number[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [enableWindowing, setEnableWindowing] = useState(true);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await processSelectedFile(e.target.files[0]);
    }
  };

  const processSelectedFile = async (file: File) => {
    setErrorMsg(null);
    setSelectedFile(file);
    setCurrentResult(null);

    // Create preview URL
    const url = URL.createObjectURL(file);
    setAudioUrl(url);

    // Extract quick waveform peaks
    try {
      const data = await fetchWaveformPeaks(file, 100);
      setWaveformPeaks(data.peaks);
    } catch (e) {
      console.warn('Could not fetch waveform peaks:', e);
      // Fallback synthetic peaks
      setWaveformPeaks(Array.from({ length: 100 }, () => Math.random() * 0.6 + 0.1));
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setAudioUrl(null);
    setWaveformPeaks([]);
    setCurrentResult(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setErrorMsg(null);

    try {
      const result = await analyzeAudioFile(selectedFile, enableWindowing);
      setCurrentResult(result);
      onAnalysisComplete(result);
    } catch (err: any) {
      setErrorMsg(err.message || 'An error occurred during audio inference.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Upload & Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col: Upload & Audio Workspace (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-white tracking-tight">
                  Audio Recording Ingestion
                </h2>
                <p className="text-xs text-slate-400">
                  Upload audio to evaluate through the VoiceGuard AASIST detection engine
                </p>
              </div>

              {selectedFile && (
                <button
                  onClick={handleClear}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 transition-colors border border-rose-500/20"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Remove</span>
                </button>
              )}
            </div>

            {/* Drag & Drop Zone */}
            {!selectedFile ? (
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleFileDrop}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-700/80 hover:border-blue-500/60 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-colors bg-slate-900/40 hover:bg-slate-900/80 group"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/wav,audio/mp3,audio/mpeg,audio/flac,audio/ogg,.wav,.mp3,.flac,.ogg"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="w-12 h-12 rounded-xl bg-blue-500/10 group-hover:bg-blue-500/20 text-blue-400 flex items-center justify-center transition-colors mb-3">
                  <UploadCloud className="w-6 h-6" />
                </div>
                <div className="text-sm font-semibold text-slate-200">
                  Drop audio file here or <span className="text-blue-400 underline">browse</span>
                </div>
                <div className="text-xs text-slate-500 mt-1 font-mono">
                  Supports WAV, MP3, FLAC, OGG (Max 50MB, 0.5s - 180s)
                </div>
              </div>
            ) : (
              /* Selected Audio Card */
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <div className="flex items-center gap-3 truncate">
                    <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400 shrink-0">
                      <FileAudio className="w-5 h-5" />
                    </div>
                    <div className="truncate">
                      <div className="text-sm font-medium text-slate-200 truncate font-mono">
                        {selectedFile.name}
                      </div>
                      <div className="text-xs text-slate-400 font-mono">
                        {(selectedFile.size / 1024).toFixed(1)} KB • {selectedFile.type || 'audio file'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Audio Player */}
                <AudioPlayer
                  audioUrl={audioUrl}
                  filename={selectedFile.name}
                  onTimeUpdate={setCurrentTime}
                />

                {/* Waveform Visualizer */}
                <WaveformVisualizer
                  peaks={waveformPeaks}
                  durationSec={currentResult?.audio_metadata.duration_sec || 0}
                  currentTime={currentTime}
                  segments={currentResult?.segments || []}
                  onSeek={(sec) => setCurrentTime(sec)}
                />

                {/* Windowing Option Toggle */}
                <div className="flex items-center justify-between pt-2 px-1 text-xs text-slate-400">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableWindowing}
                      onChange={(e) => setEnableWindowing(e.target.checked)}
                      className="rounded bg-slate-800 border-slate-700 text-blue-500 focus:ring-0"
                    />
                    <span>Multi-window timeline analysis (4.04s sliding segments)</span>
                  </label>
                  <span className="font-mono text-slate-500">AASIST-L</span>
                </div>

                {/* Action Button */}
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className={`w-full py-3 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all ${
                    isAnalyzing
                      ? 'bg-blue-600/50 text-blue-200 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/25'
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      <span>Running AASIST Graph Inference...</span>
                    </>
                  ) : (
                    <>
                      <PlayCircle className="w-4 h-4" />
                      <span>Execute Deepfake Analysis</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Error Message */}
            {errorMsg && (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-3">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold">Analysis Failed</div>
                  <div className="mt-0.5">{errorMsg}</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Col: Detailed Result Card (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {currentResult ? (
            <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-6">
              {/* Header Status */}
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
                    Classification Result
                  </span>
                  <div className="mt-1">
                    <ClassificationBadge state={currentResult.classification} size="lg" />
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-[11px] font-mono text-slate-400">Latency</span>
                  <div className="text-sm font-bold font-mono text-slate-300">
                    {currentResult.processing_duration_ms.toFixed(1)} ms
                  </div>
                </div>
              </div>

              {/* Score Gauge */}
              <ScoreGauge
                score={currentResult.model_score}
                confidence={currentResult.confidence}
              />

              {/* Uncertainty Alert if applicable */}
              {currentResult.is_uncertain && (
                <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs space-y-1">
                  <div className="flex items-center gap-2 font-semibold">
                    <HelpCircle className="w-4 h-4 text-amber-400" />
                    <span>Inconclusive Acoustic Evidence</span>
                  </div>
                  <p className="text-amber-200/80 leading-relaxed">
                    {currentResult.uncertainty_reason ||
                      'The model score lies between the calibrated human and synthetic thresholds (0.30 - 0.70). Manual listening or secondary verification is recommended.'}
                  </p>
                </div>
              )}

              {/* Audio & Model Specifications */}
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-1">
                  <div className="text-slate-400 text-[10px] uppercase">Duration</div>
                  <div className="text-slate-200 font-semibold">
                    {currentResult.audio_metadata.duration_sec.toFixed(2)}s
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-1">
                  <div className="text-slate-400 text-[10px] uppercase">Sample Rate</div>
                  <div className="text-slate-200 font-semibold">
                    {currentResult.audio_metadata.sample_rate} Hz ({currentResult.audio_metadata.channels}ch)
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-1">
                  <div className="text-slate-400 text-[10px] uppercase">Engine</div>
                  <div className="text-slate-200 font-semibold truncate">
                    {currentResult.model_name}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-1">
                  <div className="text-slate-400 text-[10px] uppercase">Windows Evaluated</div>
                  <div className="text-slate-200 font-semibold">
                    {currentResult.segments.length} segment(s)
                  </div>
                </div>
              </div>

              {/* Spectral Diagnostics Breakdown */}
              {currentResult.spectral_diagnostics && (
                <div className="space-y-3 pt-2 border-t border-slate-800">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-white">Spectral Diagnostics</span>
                    <span className="text-slate-400 font-mono text-[11px]">Explainability</span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800/60">
                      <div className="text-slate-400 text-[10px]">Pitch Stability</div>
                      <div className="text-slate-200 mt-0.5">
                        {currentResult.spectral_diagnostics.pitch_stability.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800/60">
                      <div className="text-slate-400 text-[10px]">Spectral Flatness</div>
                      <div className="text-slate-200 mt-0.5">
                        {currentResult.spectral_diagnostics.spectral_flatness.toFixed(6)}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800/60">
                      <div className="text-slate-400 text-[10px]">Fundamental F0</div>
                      <div className="text-slate-200 mt-0.5">
                        {currentResult.spectral_diagnostics.pitch_mean_hz.toFixed(1)} Hz
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800/60">
                      <div className="text-slate-400 text-[10px]">Harmonic Ratio</div>
                      <div className="text-slate-200 mt-0.5">
                        {currentResult.spectral_diagnostics.harmonic_ratio.toFixed(3)}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Crucial Limitations Notice */}
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 text-[11px] text-slate-400 space-y-1">
                <div className="flex items-center gap-1.5 text-slate-300 font-medium">
                  <Info className="w-3.5 h-3.5 text-blue-400" />
                  <span>Important Operational Notice</span>
                </div>
                <p className="leading-relaxed">
                  {currentResult.limitations_note}
                </p>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="glass-panel rounded-2xl p-8 border border-slate-800 h-full flex flex-col items-center justify-center text-center space-y-3">
              <div className="w-12 h-12 rounded-xl bg-slate-800/60 flex items-center justify-center text-slate-500">
                <Activity className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-semibold text-slate-300">Awaiting Audio Submission</h3>
              <p className="text-xs text-slate-500 max-w-xs">
                Upload an audio file on the left and click Execute Deepfake Analysis to view the model classification and timeline breakdown.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
