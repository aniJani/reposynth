// components/VibeStationDrawer.tsx
'use client';

import { useStore } from '@/lib/store';
import { generateVibePrompt, getJobFiles } from '@/lib/api';
import { useState, useEffect } from 'react';
import { X, Copy, CheckCircle, PlayCircle, AlertCircle, Loader2 } from 'lucide-react';

interface VibeStationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export function VibeStationDrawer({ isOpen, onClose }: VibeStationDrawerProps) {
  const {
    currentJob,
    config,
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
      const tokenLimit = getEffectiveTokenLimit();

      // Map analysis mode to filter strategy:
      // - 'hybrid' (Balanced) -> 'aggressive' (exclude API files)
      // - 'full' (Deep Dive) -> 'minimal' (include minimal API signatures)
      // - 'semantic' -> 'minimal' (fallback)
      const filterStrategy = config.mode === 'hybrid' ? 'aggressive' : 'minimal';

      // Map frontend modes to backend modes:
      // - 'prompt' mode with query -> 'focus' (search-based)
      // - 'prompt' mode without query -> 'blueprint' (full project overview)
      // - 'file' mode -> 'bundle' (dependency tree from file)
      let backendMode: 'blueprint' | 'focus' | 'bundle';
      let query: string | undefined;
      let entryPoint: string | undefined;

      if (vibeMode === 'prompt') {
        if (vibeQuery.trim()) {
          backendMode = 'focus';
          query = vibeQuery;
        } else {
          backendMode = 'blueprint';
        }
      } else {
        // file mode
        backendMode = 'bundle';
        entryPoint = vibeEntryPoint;
      }

      const response = await generateVibePrompt({
        job_id: currentJob.id,
        mode: vibeMode,
        query: (vibeMode === 'focus' || (vibeMode === 'bundle' && bundleSubMode === 'search')) ? vibeQuery : undefined,
        entry_point: (vibeMode === 'bundle' && bundleSubMode === 'file') ? vibeEntryPoint : undefined,
        max_files: 5,
        max_depth: 3,
        token_limit: tokenLimit,
        minify_source: true,
        keep_docs: false,
        filter_strategy: filterStrategy,
      });

      setVibePrompt(response.prompt);
      setVibeMetadata(response.metadata);
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Failed to generate prompt';
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
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleEscape);
      return () => window.removeEventListener('keydown', handleEscape);
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="fixed inset-y-0 right-0 w-full md:w-3/4 lg:w-2/3 xl:w-1/2 bg-zinc-900 z-50 overflow-hidden flex flex-col shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between bg-gradient-to-r from-neon-purple/10 to-cyber-blue/10">
          <h1 className="text-zinc-200 text-xl font-bold font-display tracking-tight">
            Vibe Station
          </h1>
          <button
            onClick={onClose}
            className="p-2 hover:bg-zinc-800 rounded-lg transition-colors text-zinc-400 hover:text-zinc-200"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Mode Tabs */}
            <div>
              <label className="text-zinc-400 text-xs font-semibold tracking-wider mb-2 block">
                COMPRESSION MODE
              </label>
              <div className="flex border border-zinc-800 rounded-lg overflow-hidden">
                {(['blueprint', 'focus', 'bundle'] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setVibeMode(mode)}
                    className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                      vibeMode === mode
                        ? 'bg-cyber-blue text-white'
                        : 'bg-zinc-900 text-zinc-400 hover:bg-zinc-800 hover:text-white'
                    }`}
                  >
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Mode-Specific Inputs */}
            {vibeMode === 'focus' && (
              <div>
                <label
                  className="text-zinc-400 text-xs font-semibold tracking-wider mb-2 block"
                  htmlFor="vibe-query"
                >
                  FOCUS_MODE // QUERY
                </label>
                <textarea
                  id="vibe-query"
                  value={vibeQuery}
                  onChange={(e) => setVibeQuery(e.target.value)}
                  placeholder="e.g., Analyze the component structure and data flow..."
                  className="w-full resize-none bg-zinc-950 border border-zinc-800 rounded-lg p-4 text-zinc-300 placeholder:text-zinc-600 focus:ring-2 focus:ring-cyber-blue focus:border-cyber-blue transition-colors font-mono text-sm"
                  rows={6}
                />
              </div>
            )}

            {vibeMode === 'bundle' && (
              <div className="space-y-4">
                <div className="flex rounded-lg bg-zinc-950 p-1 border border-zinc-800">
                  <button
                    onClick={() => setBundleSubMode('file')}
                    className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-all ${
                      bundleSubMode === 'file'
                        ? 'bg-zinc-800 text-white shadow-sm'
                        : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    Select File
                  </button>
                  <button
                    onClick={() => setBundleSubMode('search')}
                    className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-all ${
                      bundleSubMode === 'search'
                        ? 'bg-zinc-800 text-white shadow-sm'
                        : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    Search Tree
                  </button>
                </div>

                {bundleSubMode === 'file' ? (
                  <div>
                    <label
                      className="text-zinc-400 text-xs font-semibold tracking-wider mb-2 block"
                      htmlFor="vibe-entrypoint"
                    >
                      BUNDLE_MODE // ENTRY_POINT
                    </label>
                    {isLoadingFiles ? (
                      <div className="flex items-center gap-2 text-zinc-500 px-4 py-3 border border-zinc-800 rounded-lg bg-zinc-900/50">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading file list...
                      </div>
                    ) : (
                      <select
                        id="vibe-entrypoint"
                        value={vibeEntryPoint}
                        onChange={(e) => setVibeEntryPoint(e.target.value)}
                        className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-3 text-zinc-300 focus:ring-2 focus:ring-cyber-blue focus:border-cyber-blue transition-colors"
                      >
                        <option value="">Select a file...</option>
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
                      <p className="text-sm text-amber-600 mt-2">
                        ⚠️ No files found. The job might have completed without generating a graph.
                      </p>
                    )}
                  </div>
                ) : (
                  <div>
                    <label className="text-zinc-400 text-xs font-semibold tracking-wider mb-2 block">
                      BUNDLE_MODE // SEARCH_QUERY
                    </label>
                    <textarea
                      value={vibeQuery}
                      onChange={(e) => setVibeQuery(e.target.value)}
                      placeholder="e.g., Payment Service or Auth Controller"
                      className="w-full resize-none bg-zinc-950 border border-zinc-800 rounded-lg p-4 text-zinc-300 placeholder:text-zinc-600 focus:ring-2 focus:ring-cyber-blue focus:border-cyber-blue transition-colors font-mono text-sm"
                      rows={3}
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
              className={`w-full flex items-center justify-center h-12 px-4 font-bold text-base rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-900 ${
                isGeneratingPrompt || !currentJob || currentJob.status !== 'completed'
                  ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                  : 'bg-neon-purple text-white hover:bg-violet-600 focus:ring-neon-purple shadow-lg'
              }`}
            >
              <PlayCircle className="h-5 w-5 mr-2" />
              {isGeneratingPrompt ? 'GENERATING...' : 'GENERATE PROMPT'}
            </button>

            {/* Info/Error Messages */}
            {(!currentJob || currentJob.status !== 'completed') && (
              <div className="flex items-start gap-2 p-4 bg-blue-900/20 border border-blue-400/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-blue-400">
                  Complete a job first to use the Vibe Station. Submit a repository for analysis.
                </p>
              </div>
            )}

            {error && (
              <div className="flex items-start gap-2 p-4 bg-red-900/50 border border-red-400/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-400">{error}</p>
              </div>
            )}

            {/* Output Section */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-zinc-400 text-xs font-semibold tracking-wider">
                  OUTPUT // GENERATED_PROMPT
                </label>
                {vibePrompt && (
                  <button
                    onClick={handleCopyPrompt}
                    className="flex items-center gap-2 px-3 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors font-medium"
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
                )}
              </div>

              <div className="bg-zinc-950 border border-zinc-800 rounded-lg h-[400px] overflow-auto">
                {vibePrompt ? (
                  <div className="flex p-4">
                    <div className="text-right text-zinc-600 select-none pr-4 font-mono text-xs leading-6">
                      {vibePrompt.split('\n').map((_, i) => (
                        <div key={i}>{i + 1}</div>
                      ))}
                    </div>
                    <pre className="text-zinc-300 text-sm whitespace-pre-wrap flex-1 font-mono leading-6">
                      <code>{vibePrompt}</code>
                    </pre>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-zinc-600">
                    <p className="text-sm">Output will appear here after generation...</p>
                  </div>
                )}
              </div>

              {vibeMetadata && (
                <div className="mt-4 p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <p className="text-zinc-400 text-xs font-semibold tracking-wider mb-2">
                    METADATA
                  </p>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span className="text-zinc-500">Mode:</span>
                      <span className="ml-2 text-zinc-300">{vibeMetadata.mode}</span>
                    </div>
                    <div>
                      <span className="text-zinc-500">Tokens:</span>
                      <span className="ml-2 text-zinc-300">
                        {vibeMetadata.token_estimate.toLocaleString()}
                      </span>
                    </div>
                    {vibeMetadata.files_included !== undefined && (
                      <div>
                        <span className="text-zinc-500">Files:</span>
                        <span className="ml-2 text-zinc-300">{vibeMetadata.files_included}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
