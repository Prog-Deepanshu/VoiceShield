import React, { useState, useEffect, useRef } from 'react';
import {
  Radio,
  Mic,
  MicOff,
  Play,
  Square,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  Activity,
  Layers,
  Info,
} from 'lucide-react';
import { createStreamSocket } from '../services/api';
import { StreamResultData } from '../types';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

export const LiveStreamPage: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamMode, setStreamMode] = useState<'mic' | 'simulated' | null>(null);
  const [bufferedSeconds, setBufferedSeconds] = useState(0);
  const [targetSeconds, setTargetSeconds] = useState(6.0);
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [lastResult, setLastResult] = useState<StreamResultData | null>(null);
  const [alertActive, setAlertActive] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const simIntervalRef = useRef<any>(null);

  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, []);

  const startStreaming = (mode: 'mic' | 'simulated') => {
    stopStreaming();
    setHistoryData([]);
    setLastResult(null);
    setAlertActive(false);
    setStreamMode(mode);

    // Open WebSocket
    const ws = createStreamSocket(
      (msg) => {
        if (msg.type === 'buffer_status') {
          setBufferedSeconds(msg.buffered_seconds);
          setTargetSeconds(msg.target_seconds);
        } else if (msg.type === 'analysis_result') {
          const data: StreamResultData = msg.data;
          setLastResult(data);
          if (data.alert_triggered) {
            setAlertActive(true);
          }

          setHistoryData((prev) => [
            ...prev.slice(-19), // keep last 20
            {
              time: `#${data.analysis_index}`,
              rawScore: +(data.window_score * 100).toFixed(1),
              smoothedScore: +(data.smoothed_score * 100).toFixed(1),
              status: data.classification,
            },
          ]);
        }
      },
      (err) => {
        console.error('WebSocket stream error:', err);
      },
      () => {
        setIsConnected(false);
        setIsStreaming(false);
      }
    );

    socketRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setIsStreaming(true);

      if (mode === 'mic') {
        startMicCapture(ws);
      } else {
        startSimulationStream(ws);
      }
    };
  };

  const stopStreaming = () => {
    if (simIntervalRef.current) {
      clearInterval(simIntervalRef.current);
      simIntervalRef.current = null;
    }

    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(console.error);
      audioContextRef.current = null;
    }

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'end_stream' }));
      socketRef.current.close();
      socketRef.current = null;
    }

    setIsStreaming(false);
    setIsConnected(false);
    setStreamMode(null);
    setBufferedSeconds(0);
  };

  const startMicCapture = async (ws: WebSocket) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const inputData = e.inputBuffer.getChannelData(0);

        // Convert float32 [-1, 1] to 16-bit PCM little-endian
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        // Convert to base64
        const binary = String.fromCharCode(...new Uint8Array(pcm16.buffer));
        const b64 = btoa(binary);

        ws.send(JSON.stringify({ type: 'audio_chunk', data: b64 }));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
    } catch (err) {
      console.error('Mic access error:', err);
      stopStreaming();
    }
  };

  const startSimulationStream = (ws: WebSocket) => {
    // Generate synthetic chunks (white noise + sine) representing simulated VoIP stream
    simIntervalRef.current = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;

      const chunkSamples = 8000; // 0.5s of 16kHz
      const pcm16 = new Int16Array(chunkSamples);
      for (let i = 0; i < chunkSamples; i++) {
        const val = (Math.random() * 2 - 1) * 0.2; // noise
        pcm16[i] = Math.round(val * 32767);
      }

      const binary = String.fromCharCode(...new Uint8Array(pcm16.buffer));
      const b64 = btoa(binary);

      ws.send(JSON.stringify({ type: 'audio_chunk', data: b64 }));
    }, 500);
  };

  const bufferPercent = Math.min(100, Math.round((bufferedSeconds / targetSeconds) * 100));

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Title & Description */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              Real-Time Protocol
            </span>
            <span className="text-xs text-slate-400 font-mono">
              WebSocket Ring Buffer (6.0s window / 2.0s overlap)
            </span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight mt-1">
            Live Call Deepfake Detection Monitor
          </h2>
          <p className="text-sm text-slate-400 max-w-2xl">
            Streams PCM audio chunks to the VoiceGuard AASIST engine. Uses temporal exponential smoothing to mitigate single-window spikes and prevent false call accusations.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3 shrink-0">
          {!isStreaming ? (
            <>
              <button
                onClick={() => startStreaming('mic')}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold transition-colors shadow-lg shadow-blue-600/20"
              >
                <Mic className="w-4 h-4" />
                <span>Microphone Live</span>
              </button>
              <button
                onClick={() => startStreaming('simulated')}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium transition-colors border border-slate-700/60"
              >
                <Play className="w-4 h-4 text-emerald-400" />
                <span>Simulate Call Stream</span>
              </button>
            </>
          ) : (
            <button
              onClick={stopStreaming}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-sm font-semibold transition-colors shadow-lg shadow-rose-600/20"
            >
              <Square className="w-4 h-4" />
              <span>Stop Stream</span>
            </button>
          )}
        </div>
      </div>

      {/* Real-time Status Strip */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Stream Status */}
        <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="text-xs font-mono uppercase tracking-wider text-slate-400">Stream Connection</div>
          <div className="flex items-center gap-2">
            <span
              className={`w-3 h-3 rounded-full ${
                isStreaming ? 'bg-emerald-400 animate-ping' : 'bg-slate-600'
              }`}
            />
            <span className="text-lg font-bold font-mono text-white">
              {isStreaming ? (streamMode === 'mic' ? 'MIC STREAMING' : 'SIMULATING') : 'IDLE'}
            </span>
          </div>
          <div className="text-xs text-slate-500 font-mono">
            {isConnected ? 'WebSocket: /api/stream/analyze' : 'Disconnected'}
          </div>
        </div>

        {/* Ring Buffer Fill */}
        <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Buffer Fill</span>
            <span>{bufferedSeconds.toFixed(1)}s / {targetSeconds.toFixed(1)}s</span>
          </div>
          <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden border border-slate-700/60 p-0.5">
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-300"
              style={{ width: `${bufferPercent}%` }}
            />
          </div>
          <div className="text-xs text-slate-400 font-mono">
            {bufferedSeconds >= targetSeconds ? 'Window ready for evaluation' : 'Accumulating audio chunk'}
          </div>
        </div>

        {/* Smoothed Score */}
        <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-2">
          <div className="text-xs font-mono uppercase tracking-wider text-slate-400">Smoothed Risk Score</div>
          <div className="text-2xl font-bold font-mono text-white">
            {lastResult ? `${(lastResult.smoothed_score * 100).toFixed(1)}%` : '--'}
          </div>
          <div className="text-xs text-slate-400 font-mono">
            Raw window: {lastResult ? `${(lastResult.window_score * 100).toFixed(1)}%` : '--'}
          </div>
        </div>

        {/* Fraud Alert Status */}
        <div className={`glass-panel rounded-2xl p-5 border transition-colors space-y-2 ${
          alertActive ? 'border-rose-500/50 bg-rose-500/10' : 'border-slate-800'
        }`}>
          <div className="text-xs font-mono uppercase tracking-wider text-slate-400">Security Alert State</div>
          <div className="flex items-center gap-2">
            {alertActive ? (
              <>
                <ShieldAlert className="w-5 h-5 text-rose-400" />
                <span className="text-lg font-bold font-mono text-rose-400">ALERT ACTIVE</span>
              </>
            ) : (
              <>
                <ShieldCheck className="w-5 h-5 text-emerald-400" />
                <span className="text-lg font-bold font-mono text-emerald-400">NORMAL</span>
              </>
            )}
          </div>
          <div className="text-xs text-slate-400 font-mono">
            {alertActive ? '2+ consecutive high-suspicion windows' : 'No fraud threshold breach'}
          </div>
        </div>
      </div>

      {/* Live Graph */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white tracking-tight">
              Real-Time Suspicion Trajectory
            </h3>
            <p className="text-xs text-slate-400">
              Raw window score (cyan) vs. Temporally smoothed score (amber) across consecutive windows
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-cyan-400" />
              <span>Raw Window Score</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-amber-400" />
              <span>Smoothed Trajectory</span>
            </div>
          </div>
        </div>

        {historyData.length === 0 ? (
          <div className="h-64 flex flex-col items-center justify-center text-center p-4">
            <Activity className="w-8 h-8 text-slate-600 mb-2" />
            <p className="text-xs text-slate-400 font-mono">Stream inactive</p>
            <p className="text-xs text-slate-400 mt-1">Start microphone or call simulation above to view live trajectory</p>
          </div>
        ) : (
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={11} fontFamily="monospace" />
                <YAxis domain={[0, 100]} stroke="#64748b" fontSize={11} fontFamily="monospace" unit="%" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111726',
                    borderColor: '#1e293b',
                    borderRadius: '8px',
                    fontSize: '12px',
                    fontFamily: 'monospace',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="rawScore"
                  stroke="#22d3ee"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#22d3ee' }}
                />
                <Line
                  type="monotone"
                  dataKey="smoothedScore"
                  stroke="#f59e0b"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#f59e0b' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Real-time Call Protection Notice */}
      <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-400 space-y-1">
        <div className="flex items-center gap-1.5 text-slate-300 font-medium">
          <Info className="w-4 h-4 text-blue-400" />
          <span>Real-Time Call Protection Guidelines</span>
        </div>
        <p className="leading-relaxed">
          VoiceShield provides audio deepfake detection and decision support. In production call center environments, automated call terminations should never occur on a single window evaluation. Sustained multi-window elevation combined with secondary biometric checks must be employed.
        </p>
      </div>
    </div>
  );
};
