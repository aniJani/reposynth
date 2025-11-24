// components/JobStatusDisplay.tsx
'use client';

import { useStore } from '@/lib/store';
import { useEffect, useState } from 'react';
import { getJobStatus } from '@/lib/api';
import { Download, ExternalLink, Copy, Check } from 'lucide-react';
import { copyToClipboard } from '@/lib/utils';

export function JobStatusDisplay() {
  const { currentJob, setCurrentJob } = useStore();
  const [copied, setCopied] = useState(false);

  const handleCopyJobId = async () => {
    if (!currentJob) return;
    const success = await copyToClipboard(currentJob.id);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Poll for job status updates
  useEffect(() => {
    if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await getJobStatus(currentJob.id);
        setCurrentJob(updated);
      } catch (error) {
        console.error('Failed to fetch job status:', error);
      }
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, [currentJob, setCurrentJob]);

  if (!currentJob) {
    return null;
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'text-yellow-400';
      case 'processing':
        return 'text-blue-400 animate-pulse';
      case 'completed':
        return 'text-green-400';
      case 'failed':
        return 'text-red-400';
      default:
        return 'text-zinc-400';
    }
  };

  return (
    <div className="w-full space-y-2 p-4 border border-zinc-800 rounded-lg bg-zinc-900 shadow-2xl backdrop-blur-sm">
      <div className="flex items-center justify-between">
        <p className="text-zinc-400 text-sm font-mono">[JobStatusDisplay]</p>
        <button
          onClick={handleCopyJobId}
          className="flex items-center gap-1 px-2 py-1 text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded transition-colors"
          title="Click to copy Job ID"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3" />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <Copy className="h-3 w-3" />
              <span>ID: {currentJob.id.slice(0, 8)}...</span>
            </>
          )}
        </button>
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 font-mono text-base mt-2">
        <div className="space-y-4">
          {/* Pending/Waiting state */}
          {currentJob.status === 'pending' && (
            <div className="flex items-start gap-4">
              <span className="text-yellow-400 mt-0.5">&gt;</span>
              <p className="text-yellow-400">WAITING_FOR_WORKER...</p>
            </div>
          )}

          {/* Processing state */}
          {currentJob.status === 'processing' && (
            <>
              <div className="flex items-start gap-4">
                <span className="text-green-400 mt-0.5">&gt;</span>
                <p className="text-green-400">WORKER_ACQUIRED</p>
              </div>
              <div className="flex items-start gap-4">
                <span className="text-blue-400 mt-0.5">&gt;</span>
                <p className="text-blue-400 animate-pulse">ANALYZING_REPOSITORY...</p>
              </div>
            </>
          )}

          {/* Completed state */}
          {currentJob.status === 'completed' && (
            <>
              <div className="flex items-start gap-4">
                <span className="text-green-400 mt-0.5">&gt;</span>
                <p className="text-green-400">ANALYSIS_COMPLETE</p>
              </div>
              <div className="flex items-start gap-4">
                <span className="text-green-400 mt-0.5">&gt;</span>
                <div className="flex-1">
                  <p className="text-green-400 mb-2">SUCCESS</p>
                  {currentJob.processing_time_seconds && (
                    <p className="text-zinc-400 text-sm mb-4">
                      Completed in {Math.round(currentJob.processing_time_seconds)}s
                    </p>
                  )}
                  <div className="flex flex-col sm:flex-row gap-4">
                    <a
                      href={currentJob.result_url}
                      download
                      className="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 px-4 bg-neon-purple text-zinc-50 text-sm font-bold leading-normal hover:bg-violet-600 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-900 focus:ring-neon-purple font-display"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      <span className="truncate">Download Pack</span>
                    </a>
                    <a
                      href={currentJob.result_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 px-4 bg-cyber-blue text-zinc-50 text-sm font-bold leading-normal hover:bg-blue-500 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-900 focus:ring-cyber-blue font-display"
                    >
                      <ExternalLink className="h-4 w-4 mr-2" />
                      <span className="truncate">View in Browser</span>
                    </a>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Failed state */}
          {currentJob.status === 'failed' && (
            <>
              <div className="flex items-start gap-4">
                <span className="text-yellow-400 mt-0.5">&gt;</span>
                <p className="text-yellow-400">WORKER_STARTED</p>
              </div>
              <div className="flex items-start gap-4">
                <span className="text-red-400 mt-0.5">&gt;</span>
                <div className="flex-1">
                  <p className="text-red-400 mb-2">FATAL_ERROR</p>
                  {currentJob.error_message && (
                    <div className="bg-zinc-800 border border-zinc-700 rounded-md p-4 text-red-300 text-sm overflow-x-auto">
                      <pre><code>{currentJob.error_message}</code></pre>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
