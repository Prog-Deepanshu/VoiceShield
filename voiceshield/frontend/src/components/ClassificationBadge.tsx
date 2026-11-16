import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle } from 'lucide-react';
import { ClassificationState } from '../types';

interface Props {
  state: ClassificationState;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

export const ClassificationBadge: React.FC<Props> = ({
  state,
  size = 'md',
  showIcon = true,
}) => {
  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-sm px-3 py-1 gap-1.5',
    lg: 'text-base px-4 py-2 gap-2 font-semibold',
  };

  if (state === 'LIKELY HUMAN') {
    return (
      <span
        className={`inline-flex items-center rounded-full font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 ${sizeClasses[size]}`}
      >
        {showIcon && <ShieldCheck className={size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} />}
        <span>LIKELY HUMAN</span>
      </span>
    );
  }

  if (state === 'LIKELY AI-GENERATED') {
    return (
      <span
        className={`inline-flex items-center rounded-full font-medium bg-rose-500/10 text-rose-400 border border-rose-500/30 ${sizeClasses[size]}`}
      >
        {showIcon && <ShieldAlert className={size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} />}
        <span>LIKELY AI-GENERATED</span>
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center rounded-full font-medium bg-amber-500/10 text-amber-400 border border-amber-500/30 ${sizeClasses[size]}`}
    >
      {showIcon && <AlertTriangle className={size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} />}
      <span>UNCERTAIN</span>
    </span>
  );
};
