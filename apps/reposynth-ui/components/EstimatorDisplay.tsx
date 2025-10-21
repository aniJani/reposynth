// components/EstimatorDisplay.tsx
'use client';

import { useStore } from '@/lib/store';
import { Clock, DollarSign, Hash, AlertTriangle, Loader2 } from 'lucide-react';

export function EstimatorDisplay() {
  const { estimate, isEstimating, estimateError } = useStore();

  if (estimateError) {
    return (
      <div className="w-full max-w-4xl mx-auto bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center gap-2 text-red-700">
          <AlertTriangle className="h-5 w-5" />
          <span className="font-medium">Estimation Error</span>
        </div>
        <p className="text-sm text-red-600 mt-1">{estimateError}</p>
      </div>
    );
  }

  if (isEstimating) {
    return (
      <div className="w-full max-w-4xl mx-auto bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex items-center justify-center gap-3">
          <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
          <span className="text-blue-700 font-medium">Calculating estimate...</span>
        </div>
      </div>
    );
  }

  if (!estimate) {
    return (
      <div className="w-full max-w-4xl mx-auto bg-gray-50 border border-gray-200 rounded-lg p-6">
        <p className="text-center text-gray-500">
          Enter a repository URL to see the estimate
        </p>
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl mx-auto bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Estimated Cost & Time
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 text-gray-600 mb-1">
            <Hash className="h-4 w-4" />
            <span className="text-sm font-medium">Tokens</span>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {estimate.estimated_tokens.toLocaleString()}
          </p>
        </div>

        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 text-gray-600 mb-1">
            <Clock className="h-4 w-4" />
            <span className="text-sm font-medium">Time</span>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {Math.round(estimate.estimated_time_seconds)}s
          </p>
        </div>

        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-2 text-gray-600 mb-1">
            <DollarSign className="h-4 w-4" />
            <span className="text-sm font-medium">Cost</span>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            ${estimate.estimated_cost_usd.toFixed(4)}
          </p>
        </div>
      </div>

      {estimate.warnings.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-yellow-900 mb-1">Warnings</p>
              <ul className="text-xs text-yellow-700 space-y-1">
                {estimate.warnings.map((warning, i) => (
                  <li key={i}>• {warning}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
