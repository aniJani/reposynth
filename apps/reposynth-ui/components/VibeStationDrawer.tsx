// components/VibeStationDrawer.tsx
'use client';

import { useStore } from '@/lib/store';
import { generateVibePrompt, getJobFiles } from '@/lib/api';
import { useState, useEffect } from 'react';
import { X, Copy, CheckCircle, PlayCircle, AlertCircle, Loader2, Sparkles, Hash, FileText, Layers, Download } from 'lucide-react';

export function VibeStationDrawer() {
  const {
    currentJob,
    isVibeDrawerOpen,
    setIsVibeDrawerOpen,
    vibeMode,
    setVibeMode,
    vibeQuery,
    setVibeQuery,
    vibeEntryPoint,
    setVibeEntryPoint,
    vibePrompt,
    setVibePrompt,
    vibeMetadata,
    setVibeMetadata,
    vibeFileList,
    setVibeFileList,
    isGeneratingPrompt,
    setIsGeneratingPrompt,
  } = useStore();

  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [bundleSubMode, setBundleSubMode] = useState<'file' | 'search'>('file');

  useEffect(() => {
    if (vibeMode === 'bundle' && currentJob?.id) {
      if (vibeFileList.files.length === 0) {
        setIsLoadingFiles(true);
        getJobFiles(currentJob.id)
          .then((data) => {
            setVibeFileList(data);
            if (data.roots.length > 0 && !vibeEntryPoint) {
              setVibeEntryPoint(data.roots[0]);
            }
          })
          .catch((err) => console.error('Failed to load files:', err))
          .finally(() => setIsLoadingFiles(false));
      }
    }
  }, [vibeMode, currentJob?.id, setVibeEntryPoint, vibeEntryPoint, vibeFileList.files.length, setVibeFileList]);

  const handleGeneratePrompt = async () => {
    if (!currentJob || currentJob.status !== 'completed') {
      setError('Please complete a job first before generating a prompt');
      return;
    }

    if (vibeMode === 'focus' && !vibeQuery.trim()) {
      setError('Please enter a query for Focus mode');
      return;
    }

    if (vibeMode === 'bundle') {
      if (bundleSubMode === 'file' && !vibeEntryPoint.trim()) {
        setError('Please select an entry point file for Bundle mode');
        return;
      }
      if (bundleSubMode === 'search' && !vibeQuery.trim()) {
        setError('Please enter a search query for Bundle mode');
        return;
      }
    }

    setIsGeneratingPrompt(true);
    setError(null);
    setVibePrompt('');

    try {
      const response = await generateVibePrompt({
        job_id: currentJob.id,
        mode: vibeMode,
        query: (vibeMode === 'focus' || (vibeMode === 'bundle' && bundleSubMode === 'search')) ? vibeQuery : undefined,
        entry_point: (vibeMode === 'bundle' && bundleSubMode === 'file') ? vibeEntryPoint : undefined,
        max_files: 5,
        max_depth: 3,
      });

      setVibePrompt(response.prompt);
      setVibeMetadata(response.metadata);
    } catch (err: unknown) {
      const message = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to generate prompt';
      setError(message);
      console.error('Prompt generation error:', err);
    } finally {
      setIsGeneratingPrompt(false);
    }
  };

  const handleCopyPrompt = async () => {
    if (!vibePrompt) return;

    try {
      await navigator.clipboard.writeText(vibePrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Close on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsVibeDrawerOpen(false);
    };
    if (isVibeDrawerOpen) {
      window.addEventListener('keydown', handleEscape);
      return () => window.removeEventListener('keydown', handleEscape);
    }
  }, [isVibeDrawerOpen, setIsVibeDrawerOpen]);

  if (!isVibeDrawerOpen) return null;

  // Replace internal minio hostname with public URL
  const getDownloadUrl = (resultUrl: string | undefined): string => {
    if (!resultUrl) return '#';
    const MINIO_PUBLIC_URL = process.env.NEXT_PUBLIC_MINIO_URL || 'http://163.192.102.98:9000';
    return resultUrl.replace(/http:\/\/(minio|localhost):9000/g, MINIO_PUBLIC_URL);
  };

  // Handle download with proper filename
  const handleDownload = async (url: string) => {
    try {
      const downloadUrl = getDownloadUrl(url);
      const response = await fetch(downloadUrl);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = downloadUrl.split('/').pop() || 'download';
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

  return (
    <>
      {/* Drawer - No backdrop, slides in from right */}
      <div className="fixed inset-y-0 right-0 w-full md:w-1/2 lg:w-[40%] bg-[#0d1117] z-50 overflow-hidden flex flex-col border-l border-zinc-800/50 animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="px-6 py-4 border-b border-zinc-800/50 flex items-center justify-between bg-[#0d1117]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-violet-500/10 flex items-center justify-center">
              <Sparkles className="text-violet-400 h-5 w-5" />
            </div>
            <div>
              <h1 className="text-white text-lg font-semibold">
                Vibe Station
              </h1>
              <p className="text-xs text-zinc-500">Advanced Context Modes</p>
            </div>
          </div>
          <button
            onClick={() => setIsVibeDrawerOpen(false)}
            className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors text-zinc-400 hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Mode Tabs */}
            <div className="space-y-3">
              <p className="text-zinc-500 text-xs font-medium uppercase tracking-wider">Compression Mode</p>
              <div className="flex border border-zinc-800/50 rounded-xl overflow-hidden bg-zinc-900/30">
                {(['blueprint', 'focus', 'bundle'] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setVibeMode(mode)}
                    className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                      vibeMode === mode
                        ? 'bg-blue-600 text-white'
                        : 'bg-transparent text-zinc-400 hover:bg-zinc-800/50 hover:text-white'
                    }`}
                  >
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </button>
                ))}
              </div>
              <p className="text-xs text-zinc-600 px-1">
                {vibeMode === 'blueprint' && 'Structure only (5-10K tokens) - Architecture overview'}
                {vibeMode === 'focus' && 'Query-based (20-50K tokens) - Relevant files only'}
                {vibeMode === 'bundle' && 'Dependency tree (50-200K+ tokens) - Full context'}
              </p>
            </div>

            {/* Mode-Specific Inputs */}
            {vibeMode === 'focus' && (
              <div className="space-y-3 p-4 border border-zinc-800/50 rounded-xl bg-zinc-900/30">
                <p className="text-zinc-500 text-xs font-medium uppercase tracking-wider">Focus Query</p>
                <textarea
                  value={vibeQuery}
                  onChange={(e) => setVibeQuery(e.target.value)}
                  placeholder="e.g., How does authentication work? or Fix the login bug..."
                  className="w-full resize-none bg-[#0d1117] border border-zinc-800/50 rounded-xl p-4 text-zinc-300 placeholder:text-zinc-600 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-colors font-mono text-sm"
                  rows={4}
                />
              </div>
            )}

            {vibeMode === 'bundle' && (
              <div className="space-y-4 p-4 border border-zinc-800/50 rounded-xl bg-zinc-900/30">
                <p className="text-zinc-500 text-xs font-medium uppercase tracking-wider">Bundle Config</p>
                <div className="flex rounded-xl bg-[#0d1117] p-1 border border-zinc-800/50">
                  <button
                    onClick={() => setBundleSubMode('file')}
                    className={`flex-1 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      bundleSubMode === 'file'
                        ? 'bg-zinc-800 text-white'
                        : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    Select File
                  </button>
                  <button
                    onClick={() => setBundleSubMode('search')}
                    className={`flex-1 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      bundleSubMode === 'search'
                        ? 'bg-zinc-800 text-white'
                        : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    Search Tree
                  </button>
                </div>

                {bundleSubMode === 'file' ? (
                  <div>
                    {isLoadingFiles ? (
                      <div className="flex items-center gap-2 text-zinc-500 px-4 py-3 border border-zinc-800/50 rounded-xl bg-[#0d1117]">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading file list...
                      </div>
                    ) : (
                      <select
                        value={vibeEntryPoint}
                        onChange={(e) => setVibeEntryPoint(e.target.value)}
                        className="w-full bg-[#0d1117] border border-zinc-800/50 rounded-xl p-3 text-zinc-300 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-colors"
                      >
                        <option value="">Select entry point...</option>
                        {vibeFileList.roots.length > 0 && (
                          <optgroup label="Suggested Entry Points">
                            {vibeFileList.roots.map((f) => (
                              <option key={f} value={f}>
                                {f}
                              </option>
                            ))}
                          </optgroup>
                        )}
                        <optgroup label="All Files">
                          {vibeFileList.files.map((f) => (
                            <option key={f} value={f}>
                              {f}
                            </option>
                          ))}
                        </optgroup>
                      </select>
                    )}
                    {vibeFileList.files.length === 0 && !isLoadingFiles && (
                      <p className="text-sm text-amber-500 mt-2">
                        ⚠️ No files found. The job might have completed without generating a graph.
                      </p>
                    )}
                  </div>
                ) : (
                  <div>
                    <textarea
                      value={vibeQuery}
                      onChange={(e) => setVibeQuery(e.target.value)}
                      placeholder="e.g., Payment Service or Auth Controller"
                      className="w-full resize-none bg-[#0d1117] border border-zinc-800/50 rounded-xl p-4 text-zinc-300 placeholder:text-zinc-600 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-colors font-mono text-sm"
                      rows={2}
                    />
                    <p className="text-xs text-zinc-500 mt-2">
                      We&apos;ll find the most relevant file and build the tree from there
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Generate Button */}
            <button
              onClick={handleGeneratePrompt}
              disabled={isGeneratingPrompt || !currentJob || currentJob.status !== 'completed'}
              className={`w-full flex items-center justify-center h-12 px-4 font-semibold text-base rounded-xl transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#0d1117] ${
                isGeneratingPrompt || !currentJob || currentJob.status !== 'completed'
                  ? 'bg-zinc-800/50 text-zinc-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-500 focus:ring-blue-500'
              }`}
            >
              {isGeneratingPrompt ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  GENERATING...
                </>
              ) : (
                <>
                  <PlayCircle className="h-5 w-5 mr-2" />
                  GENERATE PROMPT
                </>
              )}
            </button>

            {/* Info/Error Messages */}
            {(!currentJob || currentJob.status !== 'completed') && (
              <div className="flex items-start gap-3 p-4 bg-blue-500/5 border border-blue-500/20 rounded-xl">
                <AlertCircle className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-blue-300">
                  Complete a job first to use the Vibe Station. Submit a repository for analysis.
                </p>
              </div>
            )}

            {error && (
              <div className="flex items-start gap-3 p-4 bg-red-500/5 border border-red-500/20 rounded-xl">
                <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-300">{error}</p>
              </div>
            )}

            {/* Output Section */}
            {vibePrompt && (
              <div className="space-y-4 p-4 border border-zinc-800/50 rounded-xl bg-zinc-900/30">
                {/* Token Count Header */}
                {vibeMetadata && (
                  <div className="flex items-center justify-between pb-4 border-b border-zinc-800/50">
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                        <Hash className="h-5 w-5 text-blue-400" />
                        <span className="text-blue-400 font-mono font-semibold text-lg">
                          {formatTokenCount(vibeMetadata.token_estimate)} tokens
                        </span>
                      </div>
                      {vibeMetadata.files_included !== undefined && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
                          <FileText className="h-4 w-4 text-zinc-400" />
                          <span className="text-zinc-300 font-mono font-medium text-sm">
                            {vibeMetadata.files_included} files
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleCopyPrompt}
                        className={`flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-xl transition-colors ${
                          copied
                            ? 'bg-emerald-600 text-white'
                            : 'bg-blue-600 text-white hover:bg-blue-500'
                        }`}
                      >
                        {copied ? (
                          <>
                            <CheckCircle className="h-4 w-4" />
                            Copied!
                          </>
                        ) : (
                          <>
                            <Copy className="h-4 w-4" />
                            Copy
                          </>
                        )}
                      </button>
                      {currentJob?.result_url && (
                        <button
                          onClick={() => handleDownload(currentJob.result_url!)}
                          className="flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-xl bg-zinc-800/80 text-zinc-200 hover:bg-zinc-700 transition-colors"
                        >
                          <Download className="h-4 w-4" />
                          Download
                        </button>
                      )}
                    </div>
                  </div>
                )}

                {/* Code Output */}
                <div className="bg-[#0d1117] border border-zinc-800/50 rounded-xl overflow-hidden">
                  <div className="max-h-[500px] overflow-auto">
                    <div className="flex p-4">
                      <div className="text-right text-zinc-600 select-none pr-4 font-mono text-xs leading-6 border-r border-zinc-800/50">
                        {vibePrompt.split('\n').map((_, i) => (
                          <div key={i}>{i + 1}</div>
                        ))}
                      </div>
                      <pre className="text-zinc-300 text-sm whitespace-pre-wrap flex-1 font-mono leading-6 pl-4">
                        <code>{vibePrompt}</code>
                      </pre>
                    </div>
                  </div>
                </div>

                {/* Metadata Footer */}
                {vibeMetadata && (
                  <div className="flex items-center gap-6 text-xs text-zinc-500 px-1">
                    <div className="flex items-center gap-1">
                      <Layers className="h-3 w-3" />
                      <span>Mode: {vibeMetadata.mode}</span>
                    </div>
                    {vibeMetadata.query && (
                      <div className="truncate max-w-xs">
                        Query: &quot;{vibeMetadata.query}&quot;
                      </div>
                    )}
                    {vibeMetadata.entry_point && (
                      <div className="truncate max-w-xs">
                        Entry: {vibeMetadata.entry_point}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Empty State */}
            {!vibePrompt && !error && currentJob?.status === 'completed' && (
              <div className="p-8 border border-zinc-800/50 rounded-xl bg-zinc-900/30 text-center">
                <div className="text-zinc-600 mb-3">
                  <PlayCircle className="h-12 w-12 mx-auto opacity-50" />
                </div>
                <p className="text-zinc-500 text-sm">
                  Select a mode and click Generate to create your LLM context prompt
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
