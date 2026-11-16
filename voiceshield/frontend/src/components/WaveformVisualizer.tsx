import React from 'react';
import { SegmentAnalysis } from '../types';

interface Props {
  peaks: number[];
  durationSec: number;
  currentTime?: number;
  segments?: SegmentAnalysis[];
  onSeek?: (sec: number) => void;
}

export const WaveformVisualizer: React.FC<Props> = ({
  peaks,
  durationSec,
  currentTime = 0,
  segments = [],
  onSeek,
}) => {
  if (!peaks || peaks.length === 0) {
    return (
      <div className="h-28 w-full glass-panel rounded-xl flex items-center justify-center text-xs text-slate-500 font-mono">
        Waveform preview unavailable
      </div>
    );
  }

  const playProgress = durationSec > 0 ? (currentTime / durationSec) * 100 : 0;

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onSeek || durationSec <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    onSeek(ratio * durationSec);
  };

  // Helper to find segment at a given normalized position (0.0 to 1.0)
  const getSegmentColorForRatio = (ratio: number) => {
    const timeAtPoint = ratio * durationSec;
    const seg = segments.find(
      (s) => timeAtPoint >= s.start_sec && timeAtPoint <= s.end_sec
    );
    if (!seg) return 'default';
    if (seg.classification === 'LIKELY AI-GENERATED') return 'ai';
    if (seg.classification === 'UNCERTAIN') return 'uncertain';
    return 'human';
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
        <span>Timeline &amp; Segment Analysis</span>
        <span>{durationSec.toFixed(2)}s</span>
      </div>

      <div
        onClick={handleClick}
        className="relative h-28 w-full glass-panel rounded-xl p-3 flex items-center gap-[2px] cursor-pointer overflow-hidden border border-slate-800 hover:border-slate-700 transition-colors"
      >
        {/* Playhead progress line */}
        <div
          className="absolute top-0 bottom-0 w-[2px] bg-blue-400 z-20 pointer-events-none transition-all duration-75 shadow-lg shadow-blue-500"
          style={{ left: `${playProgress}%` }}
        />

        {/* Segment background overlays */}
        {segments.map((seg) => {
          const leftPercent = (seg.start_sec / durationSec) * 100;
          const widthPercent = ((seg.end_sec - seg.start_sec) / durationSec) * 100;

          let bgClass = 'bg-emerald-500/10 border-emerald-500/30';
          if (seg.classification === 'LIKELY AI-GENERATED') {
            bgClass = 'bg-rose-500/15 border-rose-500/40';
          } else if (seg.classification === 'UNCERTAIN') {
            bgClass = 'bg-amber-500/15 border-amber-500/40';
          }

          return (
            <div
              key={seg.window_index}
              title={`Segment ${seg.window_index + 1}: ${seg.start_sec.toFixed(1)}s - ${seg.end_sec.toFixed(1)}s (${seg.classification}, Score: ${(seg.score * 100).toFixed(1)}%)`}
              className={`absolute top-0 bottom-0 border-t-2 z-0 pointer-events-none ${bgClass}`}
              style={{ left: `${leftPercent}%`, width: `${widthPercent}%` }}
            />
          );
        })}

        {/* Waveform amplitude bars */}
        {peaks.map((peak, idx) => {
          const ratio = idx / peaks.length;
          const isPassed = ratio * 100 <= playProgress;
          const segType = getSegmentColorForRatio(ratio);

          let barColor = isPassed ? 'bg-blue-400' : 'bg-slate-600';
          if (segType === 'ai') {
            barColor = isPassed ? 'bg-rose-400' : 'bg-rose-500/70';
          } else if (segType === 'uncertain') {
            barColor = isPassed ? 'bg-amber-400' : 'bg-amber-500/70';
          } else if (segType === 'human') {
            barColor = isPassed ? 'bg-emerald-400' : 'bg-emerald-500/70';
          }

          const heightPercent = Math.max(8, peak * 100);

          return (
            <div
              key={idx}
              className="flex-1 flex items-center justify-center h-full z-10"
            >
              <div
                className={`w-full max-w-[4px] rounded-full transition-all duration-75 ${barColor}`}
                style={{ height: `${heightPercent}%` }}
              />
            </div>
          );
        })}
      </div>

      {segments.length > 0 && (
        <div className="flex items-center gap-4 text-[11px] text-slate-400 font-mono pt-1">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2 rounded bg-emerald-500/40 border border-emerald-500/80" />
            <span>Human Windows</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2 rounded bg-amber-500/40 border border-amber-500/80" />
            <span>Uncertain Windows</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2 rounded bg-rose-500/40 border border-rose-500/80" />
            <span>Suspicious Synthetic Windows</span>
          </div>
        </div>
      )}
    </div>
  );
};
