// components/EstimatorDisplay.tsx
'use client';

import { useStore } from '@/lib/store';
import { Clock, DollarSign, Hash, AlertTriangle, Loader2 } from 'lucide-react';

export function EstimatorDisplay() {
  const { estimate, isEstimating, estimateError } = useStore();

  if (estimateError) {
    return (
      <div className="w-full space-y-2 p-4 border border-zinc-800 rounded-md bg-zinc-900/50">
        <p className="text-zinc-400 text-sm font-mono">[EstimatorDisplay]</p>
        <div className="bg-red-900/50 border border-red-400/20 text-red-400 rounded-md p-3 flex items-start gap-3 mt-2">
          <AlertTriangle className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <p className="text-sm">{estimateError}</p>
        </div>
      </div>
    );
  }

  if (isEstimating) {
    return (
      <div className="w-full space-y-2 p-4 border border-zinc-800 rounded-md bg-zinc-900/50">
        <p className="text-zinc-400 text-sm font-mono">[EstimatorDisplay]</p>
        <div className="flex items-center justify-center h-[178px]">
          <div className="flex items-center gap-3 text-cyber-blue">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="text-lg">Calculating...</span>
          </div>
        </div>
      </div>
    );
  }

  if (!estimate) {
    return (
      <div className="w-full space-y-2 p-4 border border-zinc-800 rounded-md bg-zinc-900/50">
        <p className="text-zinc-400 text-sm font-mono">[EstimatorDisplay]</p>
        <div className="flex items-center justify-center h-24 text-zinc-600">
          <span>Cost and time estimation will be displayed here.</span>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full space-y-2 p-4 border border-zinc-800 rounded-md bg-zinc-900/50">
      <p className="text-zinc-400 text-sm font-mono">[EstimatorDisplay]</p>
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 font-mono text-base mt-2">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-xs text-zinc-400 uppercase tracking-widest">Est. Tokens</p>
            <p className="text-cyber-blue text-4xl font-bold tracking-wider mt-1">
              {(estimate.estimated_tokens / 1000000).toFixed(1)}M
            </p>
          </div>
          <div className="border-y sm:border-y-0 sm:border-x border-zinc-800 py-4 sm:py-0">
            <p className="text-xs text-zinc-400 uppercase tracking-widest">Est. Time</p>
            <p className="text-cyber-blue text-4xl font-bold tracking-wider mt-1">
              ~{Math.round(estimate.estimated_time_seconds / 60)}m
            </p>
          </div>
          <div>
            <p className="text-xs text-zinc-400 uppercase tracking-widest">Est. Cost</p>
            <p className="text-cyber-blue text-4xl font-bold tracking-wider mt-1">
              ${estimate.estimated_cost_usd.toFixed(2)}
            </p>
          </div>
        </div>

        {estimate.warnings.length > 0 && (
          <div className="mt-4 flex flex-col gap-2">
            {estimate.warnings.map((warning, i) => (
              <div
                key={i}
                className="bg-yellow-900/50 border border-yellow-400/20 text-yellow-400 text-xs rounded-md p-3 flex items-start gap-3"
              >
                <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-px" />
                <p>{warning}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
