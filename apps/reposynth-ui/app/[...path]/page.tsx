'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { useStore } from '@/lib/store';
import { createJob, getRateLimitStatus, getJobByRepo, getJobStatus } from '@/lib/api';
import type { JobStatus } from '@/lib/store';
import { VibeStationDrawer } from '@/components/VibeStationDrawer';
import { 
  Loader2, ArrowLeft, Check, CheckCircle, Clock, Copy, Download, Hash, 
  Share2, Sparkles, XCircle, Zap, RefreshCw, AlertTriangle 
} from 'lucide-react';

/**
 * Catch-all route handler for GitHub-style URL paths.
 * 
 * This page:
 * 1. Checks if a cached analysis exists for this repo
 * 2. If cached and up-to-date, shows results directly
 * 3. If cached but outdated (new commits), offers to re-analyze
 * 4. If not cached, auto-submits a new analysis job
 * 
 * Results are displayed directly at /owner/repo instead of redirecting.
 */
export default function RepoPage() {
  const router = useRouter();
  const params = useParams();
  
  // Page states
  const [pageState, setPageState] = useState<'loading' | 'analyzing' | 'results' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);
  const [needsReanalysis, setNeedsReanalysis] = useState(false);
  const [latestCommitDate, setLatestCommitDate] = useState<string | null>(null);
  
  // Job and content state
  const [job, setJob] = useState<JobStatus | null>(null);
  const [toonContent, setToonContent] = useState<string | null>(null);
  const [toonTokens, setToonTokens] = useState<number | null>(null);
  const [isLoadingToon, setIsLoadingToon] = useState(false);
  const [toonError, setToonError] = useState<string | null>(null);
  
  // UI state
  const [copied, setCopied] = useState(false);
  const [linkCopied, setLinkCopied] = useState(false);
  
  const {
    setRepoUrl,
    setConfig,
    setCurrentJob,
    setRateLimit,
    rateLimit,
    isVibeDrawerOpen,
    setIsVibeDrawerOpen,
  } = useStore();

  // Build repo info from path
  const pathSegments = params.path as string[];
  const owner = pathSegments?.[0] || '';
  const repo = pathSegments?.[1] || '';
  const repoPath = `${owner}/${repo}`;
  const githubUrl = `https://github.com/${repoPath}`;

  // Default config for auto-submit
  const fullConfig = {
    mode: 'full' as const,
    enable_ast: true,
    enable_imports: true,
    enable_complexity: true,
    enable_security: true,
    enable_embeddings: true,
    output_format: 'toon' as const,
  };

  // Initial load - check for cached results or start new analysis
  useEffect(() => {
    if (!pathSegments || pathSegments.length < 2) {
      router.replace('/');
      return;
    }

    const initializePage = async () => {
      setRepoUrl(githubUrl);
      setConfig(fullConfig);

      try {
        // First, check if we have cached results for this repo
        const cachedJob = await getJobByRepo(githubUrl);
        
        // We have cached results!
        setJob(cachedJob);
        setCurrentJob(cachedJob);
        setNeedsReanalysis(cachedJob.needs_reanalysis);
        setLatestCommitDate(cachedJob.latest_commit_date);
        setPageState('results');
        
      } catch (err: any) {
        // No cached results - need to analyze
        if (err.response?.status === 404) {
          await startNewAnalysis();
        } else {
          console.error('Error checking cache:', err);
          setError('Failed to check for existing analysis');
          setPageState('error');
        }
      }
    };

    initializePage();
  }, [pathSegments?.join('/')]);

  // Start a new analysis job
  const startNewAnalysis = async () => {
    // Check rate limit
    if (rateLimit && rateLimit.remaining !== -1 && rateLimit.remaining <= 0) {
      setError('Daily API limit reached. Please try again later.');
      setPageState('error');
      return;
    }

    setPageState('analyzing');
    setNeedsReanalysis(false);

    try {
      const response = await createJob({ repo_url: githubUrl, config: fullConfig });
      
      const newJob: JobStatus = {
        id: response.job_id,
        status: 'pending',
        repo_url: response.repo_url,
        mode: response.mode,
        created_at: new Date().toISOString(),
      };
      
      setJob(newJob);
      setCurrentJob(newJob);

      // Refresh rate limit
      try {
        const newRateLimit = await getRateLimitStatus();
        setRateLimit(newRateLimit);
      } catch {
        // Ignore
      }
    } catch (err: any) {
      console.error('Auto-submit error:', err);
      
      if (err.response?.status === 429) {
        setError('Daily API limit reached. Please try again later.');
      } else {
        setError(err.response?.data?.detail || 'Failed to start analysis');
      }
      setPageState('error');
    }
  };

  // Poll for job status during analysis
  useEffect(() => {
    if (pageState !== 'analyzing' || !job || job.status === 'completed' || job.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await getJobStatus(job.id);
        setJob(updated);
        setCurrentJob(updated);
        
        if (updated.status === 'completed') {
          setPageState('results');
        } else if (updated.status === 'failed') {
          setError(updated.error_message || 'Analysis failed');
          setPageState('error');
        }
      } catch (error) {
        console.error('Failed to fetch job status:', error);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [pageState, job?.id, job?.status]);

  // Load TOON content when job completes
  useEffect(() => {
    if (pageState === 'results' && job?.status === 'completed' && job.result_url && !toonContent && !isLoadingToon) {
      loadToonContent();
    }
  }, [pageState, job?.status, job?.result_url, toonContent, isLoadingToon]);

  const loadToonContent = async () => {
    if (!job?.result_url) return;

    setIsLoadingToon(true);
    setToonError(null);

    try {
      const MINIO_PUBLIC_URL = process.env.NEXT_PUBLIC_MINIO_URL || 'https://reposynth.duckdns.org/storage';
      const fetchUrl = job.result_url.replace(/http:\/\/(minio|localhost):9000/g, MINIO_PUBLIC_URL);

      const response = await fetch(fetchUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }
      const content = await response.text();
      setToonContent(content);
      setToonTokens(Math.round(content.length / 4));
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

  const handleDownload = async (url: string) => {
    try {
      const downloadUrl = getDownloadUrl(url);
      const response = await fetch(downloadUrl);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = `${repo}.toon`;
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
    if (count >= 1000000) return `${(count / 1000000).toFixed(2)}M`;
    if (count >= 1000) return `${(count / 1000).toFixed(1)}K`;
    return count.toString();
  };

  const handleBack = () => {
    setCurrentJob(null);
    router.push('/');
  };

  // ========== RENDER STATES ==========

  // Loading state - checking for cached results
  if (pageState === 'loading') {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-12 w-12 text-teal-500 animate-spin mx-auto mb-4" />
          <p className="text-zinc-400 text-lg">Checking for {repoPath}...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (pageState === 'error') {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="text-center p-8">
          <XCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-red-400 mb-4">Analysis Failed</h1>
          <p className="text-zinc-400 mb-6">{error}</p>
          <button
            onClick={handleBack}
            className="px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-500 transition-colors"
          >
            Go to Home
          </button>
        </div>
      </div>
    );
  }

  // Analyzing state - job in progress
  if (pageState === 'analyzing' && job?.status !== 'completed') {
    return (
      <div className="min-h-screen bg-zinc-950">
        <header className="border-b border-zinc-800/50 bg-zinc-900/50 backdrop-blur-sm">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center gap-4">
            <button onClick={handleBack} className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors">
              <ArrowLeft className="h-4 w-4" />
              <span className="text-sm">Back</span>
            </button>
            <div className="h-4 w-px bg-zinc-700" />
            <h1 className="text-lg font-semibold text-white">{repo}</h1>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-6 py-8">
          <div className="p-6 border border-zinc-800/50 rounded-xl bg-zinc-900/30">
            {job?.status === 'pending' && (
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

            {job?.status === 'processing' && (
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
          </div>

          <div className="text-center mt-8">
            <a href={githubUrl} target="_blank" rel="noopener noreferrer" className="text-sm text-zinc-500 hover:text-blue-400 transition-colors">
              {githubUrl}
            </a>
          </div>
        </main>
      </div>
    );
  }

  // Results state - analysis complete
  return (
    <div className="relative min-h-screen bg-zinc-950 text-white">
      <header className={`border-b border-zinc-800/50 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-10 transition-all duration-300 ${isVibeDrawerOpen ? 'mr-[50%] lg:mr-[40%]' : ''}`}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button onClick={handleBack} className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors">
              <ArrowLeft className="h-4 w-4" />
              <span className="text-sm">Back</span>
            </button>
            <div className="h-4 w-px bg-zinc-700" />
            <h1 className="text-lg font-semibold text-white">{repo}</h1>
          </div>
          <button
            onClick={handleCopyLink}
            className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg transition-colors ${
              linkCopied ? 'bg-emerald-500/10 text-emerald-400' : 'bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700'
            }`}
          >
            {linkCopied ? <><Check className="h-4 w-4" />Link Copied!</> : <><Share2 className="h-4 w-4" />Share</>}
          </button>
        </div>
      </header>

      <main className={`max-w-6xl mx-auto px-6 py-8 transition-all duration-300 ${isVibeDrawerOpen ? 'mr-[50%] lg:mr-[40%]' : ''}`}>
        <div className="space-y-6">
          {/* Job Info */}
          <div className="flex items-center gap-3 text-sm text-zinc-500">
            <span className="font-mono">{job?.id}</span>
            <span>|</span>
            <span>{job?.mode} mode</span>
            {job?.processing_time_seconds && (
              <>
                <span>|</span>
                <span>{Math.round(job.processing_time_seconds)}s</span>
              </>
            )}
          </div>

          {/* Reanalysis Banner */}
          {needsReanalysis && (
            <div className="p-4 border border-amber-500/30 bg-amber-500/10 rounded-xl flex items-center justify-between">
              <div className="flex items-center gap-3">
                <AlertTriangle className="h-5 w-5 text-amber-400" />
                <div>
                  <p className="text-amber-200 font-medium">New commits detected</p>
                  <p className="text-amber-300/70 text-sm">
                    Repository has been updated since this analysis
                    {latestCommitDate && ` (${new Date(latestCommitDate).toLocaleDateString()})`}
                  </p>
                </div>
              </div>
              <button
                onClick={startNewAnalysis}
                className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-black font-semibold rounded-lg hover:bg-amber-400 transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
                Re-analyze
              </button>
            </div>
          )}

          {/* Results Panel */}
          <div className="p-6 border border-zinc-800/50 rounded-xl bg-zinc-900/30 backdrop-blur-sm">
            <div className="space-y-5">
              {/* Success Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center">
                    <CheckCircle className="h-6 w-6 text-emerald-400" />
                  </div>
                  <div>
                    <p className="text-white font-semibold text-lg">Analysis Complete</p>
                    {job?.processing_time_seconds && (
                      <p className="text-zinc-500 text-sm">Completed in {Math.round(job.processing_time_seconds)}s</p>
                    )}
                  </div>
                </div>

                {toonTokens && (
                  <div className="flex items-center gap-2 px-4 py-2.5 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                    <Hash className="h-5 w-5 text-blue-400" />
                    <span className="text-blue-400 font-mono font-semibold text-lg">{formatTokenCount(toonTokens)} tokens</span>
                  </div>
                )}
              </div>

              {/* TOON Content */}
              <div className="space-y-4">
                {isLoadingToon && (
                  <div className="flex items-center gap-3 p-4 bg-zinc-900/50 border border-zinc-800/50 rounded-xl">
                    <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                    <span className="text-zinc-400">Loading output...</span>
                  </div>
                )}

                {toonError && (
                  <div className="p-4 bg-red-950/30 border border-red-500/20 rounded-xl">
                    <p className="text-red-400 text-sm">{toonError}</p>
                    <button onClick={loadToonContent} className="mt-2 text-sm text-red-300 hover:text-red-200 underline">Retry</button>
                  </div>
                )}

                {toonContent && (
                  <>
                    <div className="flex flex-wrap gap-3">
                      <button
                        onClick={handleCopyToon}
                        className={`flex items-center gap-2 px-5 py-2.5 font-semibold rounded-xl transition-colors ${
                          copied ? 'bg-emerald-600 text-white' : 'bg-blue-600 text-white hover:bg-blue-500'
                        }`}
                      >
                        {copied ? <><Check className="h-4 w-4" />Copied!</> : <><Copy className="h-4 w-4" />Copy Content</>}
                      </button>
                      <button
                        onClick={() => job?.result_url && handleDownload(job.result_url)}
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

                    <div className="bg-[#0d1117] border border-zinc-800/50 rounded-xl overflow-hidden">
                      <div className="max-h-[600px] overflow-auto">
                        <div className="flex p-4">
                          <div className="text-right text-zinc-600 select-none pr-4 font-mono text-xs leading-6 border-r border-zinc-800/50">
                            {toonContent.split('\n').map((_, i) => <div key={i}>{i + 1}</div>)}
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
          </div>

          <div className="text-center">
            <a href={githubUrl} target="_blank" rel="noopener noreferrer" className="text-sm text-zinc-500 hover:text-blue-400 transition-colors">
              {githubUrl}
            </a>
          </div>
        </div>
      </main>

      {/* Floating Vibe Station Button */}
      {job?.status === 'completed' && !isVibeDrawerOpen && (
        <button
          onClick={() => setIsVibeDrawerOpen(true)}
          className="fixed bottom-8 right-8 flex items-center gap-3 px-6 py-4 bg-violet-600 text-white rounded-xl hover:bg-violet-500 transition-colors font-semibold text-base z-30"
        >
          <Sparkles className="h-5 w-5" />
          <span>Vibe Station</span>
        </button>
      )}

      <VibeStationDrawer />
    </div>
  );
}
