// components/JobProgressPanel.tsx
'use client';

import { useStore } from '@/lib/store';
import { useEffect, useState } from 'react';
import { getJobStatus } from '@/lib/api';
import { Download, Copy, Check, Loader2, CheckCircle, XCircle, Clock, Zap, Hash, RefreshCw } from 'lucide-react';

export function JobProgressPanel() {
  const { currentJob, setCurrentJob, config, setIsVibeDrawerOpen } = useStore();
  const [copied, setCopied] = useState(false);
  const [toonContent, setToonContent] = useState<string | null>(null);
  const [toonTokens, setToonTokens] = useState<number | null>(null);
  const [isLoadingToon, setIsLoadingToon] = useState(false);
  const [toonError, setToonError] = useState<string | null>(null);

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
    }, 3000);

    return () => clearInterval(interval);
  }, [currentJob, setCurrentJob]);

  // Auto-load TOON content when job completes (for toon format)
  useEffect(() => {
    if (currentJob?.status === 'completed' && config.output_format === 'toon' && !toonContent && !isLoadingToon && currentJob.result_url) {
      loadToonContent();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentJob?.status, config.output_format, currentJob?.result_url]);

  const loadToonContent = async () => {
    if (!currentJob?.result_url) return;
    
    setIsLoadingToon(true);
    setToonError(null);
    
    try {
      // The result_url from worker contains internal hostname (minio:9000)
      // Replace with public MinIO URL
      const MINIO_PUBLIC_URL = process.env.NEXT_PUBLIC_MINIO_URL || 'http://163.192.102.98:9000';
      const fetchUrl = currentJob.result_url.replace(/http:\/\/(minio|localhost):9000/g, MINIO_PUBLIC_URL);
      
      console.log('Fetching TOON from:', fetchUrl);
      
      const response = await fetch(fetchUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }
      const content = await response.text();
      setToonContent(content);
      
      // Estimate tokens (rough: ~4 chars per token)
      const estimatedTokens = Math.round(content.length / 4);
      setToonTokens(estimatedTokens);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load TOON content';
      setToonError(message);
      console.error('Failed to load TOON:', err);
    } finally {
      setIsLoadingToon(false);
    }
  };

  const handleCopyToon = async () => {
    if (!toonContent) return;

    try {
      await navigator.clipboard.writeText(toonContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleStartNew = () => {
    setCurrentJob(null);
    setToonContent(null);
    setToonTokens(null);
    setToonError(null);
  };

  // Helper to convert MinIO internal URL to public URL
  const getDownloadUrl = (resultUrl: string | undefined): string => {
    if (!resultUrl) return '#';
    // Replace internal minio hostname with public URL
    const MINIO_PUBLIC_URL = process.env.NEXT_PUBLIC_MINIO_URL || 'http://163.192.102.98:9000';
    return resultUrl.replace(/http:\/\/(minio|localhost):9000/g, MINIO_PUBLIC_URL);
  };

  // Handle download with proper filename
  const handleDownload = async (url: string, filename?: string) => {
    try {
      const downloadUrl = getDownloadUrl(url);
      const response = await fetch(downloadUrl);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename || downloadUrl.split('/').pop() || 'download';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error('Download failed:', err);
      // Fallback: open in new tab
      window.open(getDownloadUrl(url), '_blank');
    }
  };

  const formatTokenCount = (count: number) => {
    if (count >= 1000000) {
      return `${(count / 1000000).toFixed(2)}M`;
    } else if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}K`;
    }
    return count.toString();
  };

  if (!currentJob) return null;

  const isZipFormat = config.output_format === 'zip';

  return (
    <div className="w-full space-y-2 p-5 border border-zinc-800/50 rounded-xl bg-zinc-900/30 backdrop-blur-sm">
      <div className="flex items-center justify-between">
        <p className="text-zinc-500 text-xs font-medium uppercase tracking-wider">Job Progress</p>
        {currentJob.status === 'completed' && (
          <button
            onClick={handleStartNew}
            className="flex items-center gap-2 px-3 py-1.5 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800/50 rounded-lg transition-colors"
          >
            <RefreshCw className="h-3 w-3" />
            New Analysis
          </button>
        )}
      </div>

      {/* Progress States */}
      <div className="bg-zinc-950/50 border border-zinc-800/50 rounded-xl p-6 mt-3">
        {/* Pending */}
        {currentJob.status === 'pending' && (
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-12 h-12 rounded-full bg-amber-500/10 flex items-center justify-center">
                <Clock className="h-6 w-6 text-amber-400" />
              </div>
              <div className="absolute inset-0 rounded-full bg-amber-500/20 animate-ping" />
            </div>
            <div>
              <p className="text-white font-semibold text-lg">Waiting for Worker</p>
              <p className="text-zinc-500 text-sm">Your job is queued and will start shortly...</p>
            </div>
          </div>
        )}

        {/* Processing */}
        {currentJob.status === 'processing' && (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center">
                <Loader2 className="h-6 w-6 text-blue-400 animate-spin" />
              </div>
              <div>
                <p className="text-white font-semibold text-lg">Analyzing Repository</p>
                <p className="text-zinc-500 text-sm">Parsing code, building graphs, generating embeddings...</p>
              </div>
            </div>
            <div className="w-full bg-zinc-800/50 rounded-full h-1.5 overflow-hidden">
              <div className="bg-blue-500 h-full w-2/3 animate-pulse rounded-full" />
            </div>
          </div>
        )}

        {/* Failed */}
        {currentJob.status === 'failed' && (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center">
                <XCircle className="h-6 w-6 text-red-400" />
              </div>
              <div>
                <p className="text-white font-semibold text-lg">Analysis Failed</p>
                <p className="text-zinc-500 text-sm">Something went wrong during processing</p>
              </div>
            </div>
            {currentJob.error_message && (
              <div className="bg-red-950/30 border border-red-500/20 rounded-lg p-4">
                <pre className="text-red-300 text-sm font-mono whitespace-pre-wrap">{currentJob.error_message}</pre>
              </div>
            )}
            <button
              onClick={handleStartNew}
              className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg transition-colors text-sm"
            >
              <RefreshCw className="h-4 w-4" />
              Try Again
            </button>
          </div>
        )}

        {/* Completed */}
        {currentJob.status === 'completed' && (
          <div className="space-y-5">
            {/* Success Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center">
                  <CheckCircle className="h-6 w-6 text-emerald-400" />
                </div>
                <div>
                  <p className="text-white font-semibold text-lg">Analysis Complete</p>
                  {currentJob.processing_time_seconds && (
                    <p className="text-zinc-500 text-sm">
                      Completed in {Math.round(currentJob.processing_time_seconds)}s
                    </p>
                  )}
                </div>
              </div>
              
              {/* Token Count Badge (for TOON) */}
              {!isZipFormat && toonTokens && (
                <div className="flex items-center gap-2 px-4 py-2.5 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                  <Hash className="h-5 w-5 text-blue-400" />
                  <span className="text-blue-400 font-mono font-semibold text-lg">
                    {formatTokenCount(toonTokens)} tokens
                  </span>
                </div>
              )}
            </div>

            {/* ZIP Format Output */}
            {isZipFormat && (
              <div className="flex flex-col sm:flex-row gap-3">
                <button
                  onClick={() => currentJob.result_url && handleDownload(currentJob.result_url)}
                  className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-500 transition-colors"
                >
                  <Download className="h-5 w-5" />
                  Download ZIP Pack
                </button>
              </div>
            )}

            {/* TOON Format Output */}
            {!isZipFormat && (
              <div className="space-y-4">
                {/* Loading State */}
                {isLoadingToon && (
                  <div className="flex items-center gap-3 p-4 bg-zinc-900/50 border border-zinc-800/50 rounded-xl">
                    <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                    <span className="text-zinc-400">Loading TOON output...</span>
                  </div>
                )}

                {/* Error State */}
                {toonError && (
                  <div className="p-4 bg-red-950/30 border border-red-500/20 rounded-xl">
                    <p className="text-red-400 text-sm">{toonError}</p>
                    <button
                      onClick={loadToonContent}
                      className="mt-2 text-sm text-red-300 hover:text-red-200 underline"
                    >
                      Retry
                    </button>
                  </div>
                )}

                {/* TOON Content */}
                {toonContent && (
                  <>
                    {/* Action Buttons */}
                    <div className="flex flex-wrap gap-3">
                      <button
                        onClick={handleCopyToon}
                        className={`flex items-center gap-2 px-5 py-2.5 font-semibold rounded-xl transition-colors ${
                          copied
                            ? 'bg-emerald-600 text-white'
                            : 'bg-blue-600 text-white hover:bg-blue-500'
                        }`}
                      >
                        {copied ? (
                          <>
                            <Check className="h-4 w-4" />
                            Copied!
                          </>
                        ) : (
                          <>
                            <Copy className="h-4 w-4" />
                            Copy TOON
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => currentJob.result_url && handleDownload(currentJob.result_url)}
                        className="flex items-center gap-2 px-5 py-2.5 bg-zinc-800/80 text-zinc-200 font-semibold rounded-xl hover:bg-zinc-700 transition-colors"
                      >
                        <Download className="h-4 w-4" />
                        Download File
                      </button>
                      <button
                        onClick={() => setIsVibeDrawerOpen(true)}
                        className="flex items-center gap-2 px-5 py-2.5 bg-violet-500/10 text-violet-400 font-semibold rounded-xl hover:bg-violet-500/20 transition-colors border border-violet-500/20"
                      >
                        <Zap className="h-4 w-4" />
                        Advanced Modes
                      </button>
                    </div>

                    {/* Code Output - Full content */}
                    <div className="bg-[#0d1117] border border-zinc-800/50 rounded-xl overflow-hidden">
                      <div className="max-h-[500px] overflow-auto">
                        <div className="flex p-4">
                          <div className="text-right text-zinc-600 select-none pr-4 font-mono text-xs leading-6 border-r border-zinc-800/50">
                            {toonContent.split('\n').map((_, i) => (
                              <div key={i}>{i + 1}</div>
                            ))}
                          </div>
                          <pre className="text-zinc-300 text-sm whitespace-pre-wrap flex-1 font-mono leading-6 pl-4">
                            <code>{toonContent}</code>
                          </pre>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
