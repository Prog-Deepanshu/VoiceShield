import React from 'react';

interface Props {
  score: number; // 0.0 to 1.0
  confidence: number;
  thresholdLow?: number;
  thresholdHigh?: number;
}

export const ScoreGauge: React.FC<Props> = ({
  score,
  confidence,
  thresholdLow = 0.3,
  thresholdHigh = 0.7,
}) => {
  const percent = Math.min(100, Math.max(0, score * 100));

  const getColor = () => {
    if (score < thresholdLow) return 'text-emerald-400';
    if (score >= thresholdHigh) return 'text-rose-400';
    return 'text-amber-400';
  };

  const getBarGradient = () => {
    if (score < thresholdLow) return 'from-emerald-500 to-teal-400';
    if (score >= thresholdHigh) return 'from-rose-500 to-red-600';
    return 'from-amber-500 to-orange-400';
  };

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <div>
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
            Deepfake Suspicion Score
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className={`text-3xl font-bold font-mono tracking-tight ${getColor()}`}>
              {(score * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-slate-400 font-mono">
              (raw: {score.toFixed(4)})
            </span>
          </div>
        </div>

        <div className="text-right">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
            Certainty Metric
          </span>
          <div className="text-base font-semibold text-slate-200 mt-1 font-mono">
            {(confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Progress Track */}
      <div className="relative w-full h-3 bg-slate-800/80 rounded-full overflow-hidden border border-slate-700/60 p-0.5">
        {/* Fill bar */}
        <div
          className={`h-full rounded-full transition-all duration-700 bg-gradient-to-r ${getBarGradient()}`}
          style={{ width: `${percent}%` }}
        />
      </div>

      {/* Threshold Zones Legend */}
      <div className="flex items-center justify-between text-[11px] text-slate-400 px-0.5 font-mono">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          <span>Human (&lt;0.30)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-amber-500" />
          <span>Uncertain (0.30-0.70)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-rose-500" />
          <span>Synthetic (≥0.70)</span>
        </div>
      </div>
    </div>
  );
};
