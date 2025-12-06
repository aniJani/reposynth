'use client';

import { VibeStationDrawer } from '@/components/VibeStationDrawer';
import { getJobStatus } from '@/lib/api';
import type { JobStatus } from '@/lib/store';
import { useStore } from '@/lib/store';
import { ArrowLeft, Check, CheckCircle, Clock, Copy, Download, Hash, Loader2, Share2, Sparkles, XCircle, Zap } from 'lucide-react';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

export default function ResultsPage() {
  const params = useParams();
  const jobId = params.jobId as string;
  const router = useRouter();

  const { setCurrentJob, isVibeDrawerOpen, setIsVibeDrawerOpen } = useStore();

  const handleBack = () => {
    setCurrentJob(null);
    router.push('/');
  };

  const [job, setJob] = useState<JobStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [toonContent, setToonContent] = useState<string | null>(null);
  const [toonTokens, setToonTokens] = useState<number | null>(null);
  const [isLoadingToon, setIsLoadingToon] = useState(false);
  const [toonError, setToonError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [linkCopied, setLinkCopied] = useState(false);

  // Fetch job status
  useEffect(() => {
    if (!jobId) return;

    const fetchJob = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const jobData = await getJobStatus(jobId);
        setJob(jobData);
        // Also set in global store for VibeStationDrawer
        setCurrentJob(jobData);
      } catch (err) {
        console.error('Failed to fetch job:', err);
        setError('Failed to load job. It may not exist or has been deleted.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchJob();
  }, [jobId, setCurrentJob]);

  // Poll for updates if job is still processing
  useEffect(() => {
    if (!job || job.status === 'completed' || job.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await getJobStatus(jobId);
        setJob(updated);
      } catch (error) {
        console.error('Failed to fetch job status:', error);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [job, jobId]);

  // Load TOON content when job completes
  useEffect(() => {
    if (job?.status === 'completed' && job.result_url && !toonContent && !isLoadingToon) {
      loadToonContent();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status, job?.result_url]);

  const loadToonContent = async () => {
    if (!job?.result_url) return;

    setIsLoadingToon(true);
    setToonError(null);

    try {
      const MINIO_PUBLIC_URL = process.env.NEXT_PUBLIC_MINIO_URL || 'https://reposynth.duckdns.org/storage';
      const fetchUrl = job.result_url.replace(/http:\/\/(minio|localhost):9000/g, MINIO_PUBLIC_URL);

      console.log('Fetching TOON from:', fetchUrl);

      const response = await fetch(fetchUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }
      const content = await response.text();
      setToonContent(content);

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

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setLinkCopied(true);
      setTimeout(() => setLinkCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy link:', err);
    }
  };

  const getDownloadUrl = (resultUrl: string | undefined): string => {
    if (!resultUrl) return '#';
    const MINIO_PUBLIC_URL = process.env.NEXT_PUBLIC_MINIO_URL || 'https://reposynth.duckdns.org/storage';
    return resultUrl.replace(/http:\/\/(minio|localhost):9000/g, MINIO_PUBLIC_URL);
  };

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

  const getRepoName = (url: string) => {
    try {
      const parts = url.replace(/\.git$/, '').split('/');
      return parts[parts.length - 1] || 'Repository';
    } catch {
      return 'Repository';
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="flex items-center gap-3">
          <Loader2 className="h-6 w-6 text-blue-400 animate-spin" />
          <span className="text-zinc-400">Loading analysis...</span>
        </div>
      </div>
    );
  }

  // Error state
  if (error || !job) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="text-center space-y-4">
          <XCircle className="h-12 w-12 text-red-400 mx-auto" />
          <p className="text-zinc-400">{error || 'Job not found'}</p>
          <button
            onClick={handleBack}
            className="inline-flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen bg-zinc-950 text-white">
      {/* Header */}
      <header className={`border-b border-zinc-800/50 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-10 transition-all duration-300 ${isVibeDrawerOpen ? 'mr-[50%] lg:mr-[40%]' : ''}`}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={handleBack}
              className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="h-4 w-4" />
              <span className="text-sm">Back</span>
            </button>
            <div className="h-4 w-px bg-zinc-700" />
            <h1 className="text-lg font-semibold text-white">
              {getRepoName(job.repo_url)}
            </h1>
          </div>
          <button
            onClick={handleCopyLink}
            className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg transition-colors ${linkCopied
                ? 'bg-emerald-500/10 text-emerald-400'
                : 'bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700'
              }`}
          >
            {linkCopied ? (
              <>
                <Check className="h-4 w-4" />
                Link Copied!
              </>
            ) : (
              <>
                <Share2 className="h-4 w-4" />
                Share
              </>
            )}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className={`max-w-6xl mx-auto px-6 py-8 transition-all duration-300 ${isVibeDrawerOpen ? 'mr-[50%] lg:mr-[40%]' : ''}`}>
        <div className="space-y-6">
          {/* Job Info */}
          <div className="flex items-center gap-3 text-sm text-zinc-500">
            <span className="font-mono">{job.id}</span>
            <span>|</span>
            <span>{job.mode} mode</span>
            {job.processing_time_seconds && (
              <>
                <span>|</span>
                <span>{Math.round(job.processing_time_seconds)}s</span>
              </>
            )}
          </div>

          {/* Progress Panel */}
          <div className="p-6 border border-zinc-800/50 rounded-xl bg-zinc-900/30 backdrop-blur-sm">
            {/* Pending */}
            {job.status === 'pending' && (
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
            {job.status === 'processing' && (
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
            {job.status === 'failed' && (
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
                {job.error_message && (
                  <div className="bg-red-950/30 border border-red-500/20 rounded-lg p-4">
                    <pre className="text-red-300 text-sm font-mono whitespace-pre-wrap">{job.error_message}</pre>
                  </div>
                )}
              </div>
            )}

            {/* Completed */}
            {job.status === 'completed' && (
              <div className="space-y-5">
                {/* Success Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center">
                      <CheckCircle className="h-6 w-6 text-emerald-400" />
                    </div>
                    <div>
                      <p className="text-white font-semibold text-lg">Analysis Complete</p>
                      {job.processing_time_seconds && (
                        <p className="text-zinc-500 text-sm">
                          Completed in {Math.round(job.processing_time_seconds)}s
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Token Count Badge */}
                  {toonTokens && (
                    <div className="flex items-center gap-2 px-4 py-2.5 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                      <Hash className="h-5 w-5 text-blue-400" />
                      <span className="text-blue-400 font-mono font-semibold text-lg">
                        {formatTokenCount(toonTokens)} tokens
                      </span>
                    </div>
                  )}
                </div>

                {/* TOON Content */}
                <div className="space-y-4">
                  {/* Loading State */}
                  {isLoadingToon && (
                    <div className="flex items-center gap-3 p-4 bg-zinc-900/50 border border-zinc-800/50 rounded-xl">
                      <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                      <span className="text-zinc-400">Loading output...</span>
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

                  {/* Content Display */}
                  {toonContent && (
                    <>
                      {/* Action Buttons */}
                      <div className="flex flex-wrap gap-3">
                        <button
                          onClick={handleCopyToon}
                          className={`flex items-center gap-2 px-5 py-2.5 font-semibold rounded-xl transition-colors ${copied
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
                              Copy Content
                            </>
                          )}
                        </button>
                        <button
                          onClick={() => job.result_url && handleDownload(job.result_url)}
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

                      {/* Code Output */}
                      <div className="bg-[#0d1117] border border-zinc-800/50 rounded-xl overflow-hidden">
                        <div className="max-h-[600px] overflow-auto">
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
              </div>
            )}
          </div>

          {/* Repository Link */}
          <div className="text-center">
            <a
              href={job.repo_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-zinc-500 hover:text-blue-400 transition-colors"
            >
              {job.repo_url}
            </a>
          </div>
        </div>
      </main>

      {/* Floating Vibe Station Button */}
      {job.status === 'completed' && !isVibeDrawerOpen && (
        <button
          onClick={() => setIsVibeDrawerOpen(true)}
          className="fixed bottom-8 right-8 flex items-center gap-3 px-6 py-4 bg-violet-600 text-white rounded-xl hover:bg-violet-500 transition-colors font-semibold text-base z-30"
        >
          <Sparkles className="h-5 w-5" />
          <span>Vibe Station</span>
        </button>
      )}

      {/* Vibe Station Drawer */}
      <VibeStationDrawer />
    </div>
  );
}
