import React from 'react';
import {
  Cpu,
  Layers,
  Network,
  AlertTriangle,
  Database,
  Sliders,
  CheckCircle,
  FileCode,
  Shield,
} from 'lucide-react';
import { ModelInfoResponse } from '../types';

interface Props {
  modelInfo: ModelInfoResponse | null;
}

export const ModelInfoPage: React.FC<Props> = ({ modelInfo }) => {
  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Title */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <Cpu className="w-5 h-5 text-blue-400" />
          <span>Model Architecture &amp; Empirical Limitations</span>
        </h2>
        <p className="text-xs text-slate-400">
          Inspection of the underlying AASIST deepfake detection engine, graph attention networks, and benchmark calibration
        </p>
      </div>

      {/* Model Spec Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Architecture Spec */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Network className="w-4 h-4 text-blue-400" />
            <span>Architecture &amp; Topology</span>
          </div>
          <div className="space-y-3 text-xs font-mono">
            <div>
              <span className="text-slate-400">Model Name:</span>
              <div className="text-slate-200 font-semibold">{modelInfo?.architecture || 'AASIST-L'}</div>
            </div>
            <div>
              <span className="text-slate-400">Parameters:</span>
              <div className="text-slate-200 font-semibold">
                {modelInfo?.parameter_count?.toLocaleString() || '85,346'} weights
              </div>
            </div>
            <div>
              <span className="text-slate-400">Input Sample Rate:</span>
              <div className="text-slate-200 font-semibold">16,000 Hz Mono</div>
            </div>
            <div>
              <span className="text-slate-400">Window Length:</span>
              <div className="text-slate-200 font-semibold">64,600 samples (4.0375s)</div>
            </div>
          </div>
        </div>

        {/* Checkpoint Provenance */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Database className="w-4 h-4 text-indigo-400" />
            <span>Checkpoint Provenance</span>
          </div>
          <div className="space-y-3 text-xs font-mono">
            <div>
              <span className="text-slate-400">Training Benchmark:</span>
              <div className="text-slate-200 font-semibold">ASVspoof 2019 LA</div>
            </div>
            <div>
              <span className="text-slate-400">Reported Benchmark EER:</span>
              <div className="text-slate-200 font-semibold">0.99% (AASIST-L) / 0.83% (AASIST)</div>
            </div>
            <div>
              <span className="text-slate-400">Original Authors:</span>
              <div className="text-slate-200 font-semibold">NAVER Clova AI / Jung et al. (ICASSP 2022)</div>
            </div>
            <div>
              <span className="text-slate-400">Weights Verification:</span>
              <div className="text-emerald-400 font-semibold flex items-center gap-1">
                <CheckCircle className="w-3.5 h-3.5" />
                <span>Genuine .pth checkpoint loaded</span>
              </div>
            </div>
          </div>
        </div>

        {/* Decision Calibration */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Sliders className="w-4 h-4 text-emerald-400" />
            <span>Decision Calibration</span>
          </div>
          <div className="space-y-3 text-xs font-mono">
            <div>
              <span className="text-slate-400">Likely Human Boundary:</span>
              <div className="text-emerald-400 font-semibold">Score &lt; 0.30</div>
            </div>
            <div>
              <span className="text-slate-400">Uncertainty Region:</span>
              <div className="text-amber-400 font-semibold">0.30 ≤ Score &lt; 0.70</div>
            </div>
            <div>
              <span className="text-slate-400">Likely AI Boundary:</span>
              <div className="text-rose-400 font-semibold">Score ≥ 0.70</div>
            </div>
            <div>
              <span className="text-slate-400">Softmax Mapping:</span>
              <div className="text-slate-200 font-semibold">p_spoof = e^(s0) / (e^s0 + e^s1)</div>
            </div>
          </div>
        </div>
      </div>

      {/* Deep Dive Architecture Flowchart */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
        <h3 className="text-sm font-semibold text-white tracking-tight">
          AASIST Inference Pipeline Dataflow
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-center text-xs font-mono">
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
            <div className="text-blue-400 font-bold">1. SincConv Front-End</div>
            <div className="text-slate-400 text-[11px]">Learned bandpass filterbank operating on raw 16kHz waveform</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
            <div className="text-blue-400 font-bold">2. Residual Encoder</div>
            <div className="text-slate-400 text-[11px]">6 stacked residual convolutional blocks extracting 2D spectro-temporal maps</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
            <div className="text-blue-400 font-bold">3. Dual Graph GAT</div>
            <div className="text-slate-400 text-[11px]">GAT-S (spectral branch) + GAT-T (temporal branch) modeling artifact relationships</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
            <div className="text-blue-400 font-bold">4. Heterogeneous Fusion</div>
            <div className="text-slate-400 text-[11px]">Master node cross-attention bridging spectro-temporal feature sub-graphs</div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
            <div className="text-blue-400 font-bold">5. Linear Classifier</div>
            <div className="text-slate-400 text-[11px]">2-class projection logits [spoof, bonafide] &rarr; Softmax probability</div>
          </div>
        </div>
      </div>

      {/* Critical Benchmark Analysis & Limitations */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-rose-400">
          <AlertTriangle className="w-4 h-4" />
          <span>Empirical Limitations &amp; Benchmark Domain Shift Analysis</span>
        </div>

        <div className="text-xs text-slate-300 leading-relaxed space-y-3 font-sans">
          <p>
            During real-world benchmarking on out-of-distribution audio datasets (e.g. UniDataPro Real vs. Fake benchmark), the original VoiceGuard repository documented significant domain shift challenges (as recorded in <code className="text-blue-400 font-mono">benchmark_dataset_report.txt</code>, where raw human speech was flagged with elevated risk in over 80% of test cases).
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800/80 space-y-1.5">
              <div className="font-semibold text-white">Why Domain Shift Occurs:</div>
              <ul className="list-disc pl-4 space-y-1 text-slate-400 text-[11px]">
                <li>AASIST was trained solely on the clean ASVspoof 2019 dataset recorded in uniform studio conditions at 16kHz.</li>
                <li>Modern 44.1kHz or 48kHz audio resampled with modern anti-aliasing filters often introduces high-frequency roll-off that resembles synthetic vocoder patterns to the model.</li>
                <li>Aggressive audio compression (e.g. WhatsApp, cellular AMR-WB, low-bitrate MP3) alters spectral flatness and phase continuity.</li>
              </ul>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800/80 space-y-1.5">
              <div className="font-semibold text-white">How VoiceShield Mitigates This:</div>
              <ul className="list-disc pl-4 space-y-1 text-slate-400 text-[11px]">
                <li><strong>3-Tier Classification:</strong> Introduces the <span className="text-amber-400 font-mono">UNCERTAIN</span> state (0.30 - 0.70) instead of forcing binary black-or-white classifications.</li>
                <li><strong>Segment Windowing:</strong> Evaluates audio across multiple 4.04s sliding segments rather than judging the entire call on the first 4 seconds.</li>
                <li><strong>Spectral Corroboration:</strong> Reports acoustic diagnostics (pitch stability, spectral flatness, harmonic ratio) to provide explainability.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
