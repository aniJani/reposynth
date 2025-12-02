// components/VibeCodingPanel.tsx
'use client';

import { useStore } from '@/lib/store';
import { generateVibePrompt, getJobFiles } from '@/lib/api';
import { useState, useEffect } from 'react';
import { Sparkles, Loader2, Copy, CheckCircle, AlertCircle, Settings2, Zap, FileText, MessageSquare } from 'lucide-react';

// Context window presets for different LLM models
const CONTEXT_PRESETS = {
  unlimited: { label: 'Unlimited', tokens: null, description: 'No optimization - include all files' },
  'gpt-4': { label: 'GPT-4', tokens: 8192, description: '8K context window' },
  'gpt-4-32k': { label: 'GPT-4 32K', tokens: 32768, description: '32K context window' },
  'gpt-4-turbo': { label: 'GPT-4 Turbo', tokens: 128000, description: '128K context window' },
  'claude-3-haiku': { label: 'Claude Haiku', tokens: 200000, description: '200K context window' },
  'claude-3-sonnet': { label: 'Claude Sonnet', tokens: 200000, description: '200K context window' },
  'gemini-1.5': { label: 'Gemini 1.5', tokens: 1000000, description: '1M context window' },
} as const;

type ContextPresetKey = keyof typeof CONTEXT_PRESETS;

export function VibeCodingPanel() {
  const {
    currentJob,
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
  const [contextPreset, setContextPreset] = useState<ContextPresetKey>('unlimited');
  const [showAdvancedContext, setShowAdvancedContext] = useState(false);

  // Reset file list when job changes
  useEffect(() => {
    setVibeFileList({ files: [], roots: [] });
  }, [currentJob?.id, setVibeFileList]);

  // Fetch files when in file mode (only if not already loaded)
  useEffect(() => {
    if (vibeMode === 'file' && currentJob?.id) {
      // Only fetch if we don't have files yet
      if (vibeFileList.files.length === 0) {
        setIsLoadingFiles(true);
        getJobFiles(currentJob.id)
          .then((data) => {
            setVibeFileList(data);
            // Auto-select first root if available and no entry point set
            if (data.roots.length > 0 && !vibeEntryPoint) {
              setVibeEntryPoint(data.roots[0]);
            }
          })
          .catch((err) => console.error('Failed to load files:', err))
          .finally(() => setIsLoadingFiles(false));
      }
    }
  }, [vibeMode, currentJob?.id, vibeFileList.files.length, vibeEntryPoint, setVibeEntryPoint, setVibeFileList]);

  // Get effective token limit from preset
  const getEffectiveTokenLimit = (): number | undefined => {
    if (contextPreset === 'unlimited') return undefined;
    const preset = CONTEXT_PRESETS[contextPreset];
    return preset.tokens ?? undefined;
  };

  const handleGeneratePrompt = async () => {
    if (!currentJob || currentJob.status !== 'completed') {
      setError('Please complete a job first before generating a prompt');
      return;
    }

    // File mode requires a file selection
    if (vibeMode === 'file' && !vibeEntryPoint.trim()) {
      setError('Please select an entry point file');
      return;
    }

    setIsGeneratingPrompt(true);
    setError(null);
    setVibePrompt('');

    try {
      const tokenLimit = getEffectiveTokenLimit();
      
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
        mode: backendMode,
        query,
        entry_point: entryPoint,
        max_files: 5,
        max_depth: 3,
        token_limit: tokenLimit,
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

  return (
    <div className="w-full space-y-2 p-4 border border-zinc-800 rounded-md bg-zinc-900/50">
      <p className="text-zinc-400 text-sm font-mono">[VibeCodingPanel]</p>

      <div className="w-full bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden mt-2">
        {/* Header */}
        <header className="px-4 py-3 border-b border-zinc-800">
          <h1 className="text-zinc-400 text-sm tracking-wider font-mono">
            VIBE_STATION // PROMPT_ENGINE
          </h1>
        </header>

        <div className="flex flex-col md:flex-row">
          {/* Left Panel - Configuration */}
          <div className="w-full md:w-1/2 p-4 border-b md:border-b-0 md:border-r border-zinc-800">
            {/* Mode Tabs - 2 modes: Prompt and File Selection */}
            <div className="flex border-b border-zinc-800 -mx-4 px-4">
              <button
                onClick={() => setVibeMode('prompt')}
                className={`flex items-center gap-2 px-4 py-2 text-sm transition-colors ${
                  vibeMode === 'prompt'
                    ? 'border-b-2 border-teal-500 text-white bg-zinc-800/50'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                <MessageSquare className="h-4 w-4" />
                Prompt
              </button>
              <button
                onClick={() => setVibeMode('file')}
                className={`flex items-center gap-2 px-4 py-2 text-sm transition-colors ${
                  vibeMode === 'file'
                    ? 'border-b-2 border-teal-500 text-white bg-zinc-800/50'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                <FileText className="h-4 w-4" />
                File Selection
              </button>
            </div>

            <div className="space-y-6 pt-4">
              {/* Prompt Mode */}
              {vibeMode === 'prompt' && (
                <div>
                  <label className="block text-sm font-medium text-zinc-400 mb-2">
                    What are you working on? <span className="text-zinc-500">(optional)</span>
                  </label>
                  <textarea
                    value={vibeQuery}
                    onChange={(e) => setVibeQuery(e.target.value)}
                    placeholder="Leave empty for full project context, or describe what you're working on..."
                    className="w-full px-4 py-3 border border-zinc-700 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent resize-none bg-zinc-800 text-zinc-200 placeholder-zinc-500"
                    rows={3}
                  />
                  <p className="text-sm text-zinc-500 mt-2">
                    {vibeQuery.trim() 
                      ? 'Will search for relevant code based on your query'
                      : 'Will generate a compressed overview of the entire project'
                    }
                  </p>
                </div>
              )}

              {/* File Selection Mode */}
              {vibeMode === 'file' && (
                <div>
                  <label className="block text-sm font-medium text-zinc-400 mb-2">
                    Entry Point File
                  </label>
                  {isLoadingFiles ? (
                    <div className="flex items-center gap-2 text-zinc-400 px-4 py-3 border border-zinc-700 rounded-lg bg-zinc-800">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Loading file list...
                    </div>
                  ) : (
                    <select
                      value={vibeEntryPoint}
                      onChange={(e) => setVibeEntryPoint(e.target.value)}
                      className="w-full px-4 py-3 border border-zinc-700 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent bg-zinc-800 text-zinc-200"
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
                    <p className="text-sm text-amber-500 mt-2">
                      No files found in the analysis pack. The job might have completed without generating a graph.
                    </p>
                  )}
                  <p className="text-sm text-zinc-500 mt-2">
                    Manually select the root file for the dependency tree
                  </p>
                </div>
              )}

              {/* Context Window Optimizer - Available for both modes */}
              <div className="border border-zinc-700 rounded-lg p-3 bg-zinc-800/50">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Zap className="h-4 w-4 text-yellow-400" />
                    <span className="text-sm font-medium text-zinc-200">Context Optimizer</span>
                  </div>
                  <button
                    onClick={() => setShowAdvancedContext(!showAdvancedContext)}
                    className="text-xs text-zinc-400 hover:text-zinc-200 flex items-center gap-1"
                  >
                    <Settings2 className="h-3 w-3" />
                    {showAdvancedContext ? 'Hide' : 'More Models'}
                  </button>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {(Object.keys(CONTEXT_PRESETS) as ContextPresetKey[]).slice(0, 4).map((key) => (
                    <button
                      key={key}
                      onClick={() => setContextPreset(key)}
                      className={`px-3 py-2 text-xs rounded-md transition-all ${
                        contextPreset === key
                          ? 'bg-teal-600 text-white ring-2 ring-teal-400'
                          : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                      }`}
                    >
                      {CONTEXT_PRESETS[key].label}
                    </button>
                  ))}
                </div>

                {showAdvancedContext && (
                  <div className="mt-3 grid grid-cols-3 gap-2">
                    {(Object.keys(CONTEXT_PRESETS) as ContextPresetKey[]).slice(4).map((key) => (
                      <button
                        key={key}
                        onClick={() => setContextPreset(key)}
                        className={`px-3 py-2 text-xs rounded-md transition-all ${
                          contextPreset === key
                            ? 'bg-teal-600 text-white ring-2 ring-teal-400'
                            : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                        }`}
                      >
                        {CONTEXT_PRESETS[key].label}
                      </button>
                    ))}
                  </div>
                )}

                <p className="text-xs text-zinc-500 mt-2">
                  {CONTEXT_PRESETS[contextPreset].description}
                  {contextPreset !== 'unlimited' && (
                    <span className="text-yellow-400 ml-1">
                      (Graph-Knapsack optimization active)
                    </span>
                  )}
                </p>
              </div>

              {/* Generate Button */}
              <button
                onClick={handleGeneratePrompt}
                disabled={
                  isGeneratingPrompt ||
                  !currentJob ||
                  currentJob.status !== 'completed' ||
                  (vibeMode === 'file' && !vibeEntryPoint.trim())
                }
                className={`w-full px-6 py-4 rounded-lg font-semibold text-lg transition-all flex items-center justify-center gap-3 ${
                  isGeneratingPrompt ||
                  !currentJob ||
                  currentJob.status !== 'completed' ||
                  (vibeMode === 'file' && !vibeEntryPoint.trim())
                    ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                    : 'bg-teal-600 text-white hover:bg-teal-500'
                }`}
              >
                {isGeneratingPrompt ? (
                  <>
                    <Loader2 className="h-6 w-6 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-6 w-6" />
                    Generate Prompt
                  </>
                )}
              </button>

              {/* Info/Error Messages */}
              {(!currentJob || currentJob.status !== 'completed') && (
                <div className="flex items-start gap-2 p-4 bg-blue-900/20 border border-blue-400/20 rounded-lg">
                  <AlertCircle className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-blue-400">
                    Complete a job first to use the Vibe Station. Submit a repository for analysis
                    above.
                  </p>
                </div>
              )}

              {error && (
                <div className="flex items-start gap-2 p-4 bg-red-900/50 border border-red-400/20 rounded-lg">
                  <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-red-400">{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Output */}
          <div className="w-full md:w-1/2 p-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-zinc-400 text-xs font-semibold tracking-wider">
                OUTPUT // GENERATED_PROMPT
              </label>
              {vibePrompt && (
                <button
                  onClick={handleCopyPrompt}
                  className="flex items-center gap-2 px-3 py-1 bg-green-600 text-white text-xs rounded-md hover:bg-green-700 transition-colors"
                >
                  {copied ? (
                    <>
                      <CheckCircle className="h-3 w-3" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      Copy
                    </>
                  )}
                </button>
              )}
            </div>

            <div className="bg-zinc-950 border border-zinc-800 rounded-md h-[296px] overflow-auto">
              {vibePrompt ? (
                <div className="flex p-3">
                  <div className="text-right text-zinc-600 select-none pr-4 font-mono text-sm">
                    {vibePrompt.split('\n').map((_, i) => (
                      <div key={i}>{i + 1}</div>
                    ))}
                  </div>
                  <pre className="text-zinc-300 text-sm whitespace-pre-wrap flex-1">
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
              <div className="mt-3 p-3 bg-zinc-900 border border-zinc-800 rounded-md">
                <p className="text-zinc-400 text-xs font-semibold tracking-wider mb-2">
                  METADATA
                </p>
                <div className="space-y-1 text-xs text-zinc-300">
                  <p>
                    <span className="text-zinc-500">Mode:</span> {vibeMetadata.mode}
                  </p>
                  <p>
                    <span className="text-zinc-500">Tokens:</span>{' '}
                    {vibeMetadata.token_estimate.toLocaleString()}
                  </p>
                  {vibeMetadata.files_included !== undefined && (
                    <p>
                      <span className="text-zinc-500">Files:</span> {vibeMetadata.files_included}
                    </p>
                  )}
                  
                  {/* Optimization Stats */}
                  {vibeMetadata.optimization?.optimization_applied && (
                    <div className="mt-2 pt-2 border-t border-zinc-700">
                      <p className="text-yellow-400 text-xs font-semibold mb-1">
                        ⚡ Context Optimization Applied
                      </p>
                      <p>
                        <span className="text-zinc-500">Budget:</span>{' '}
                        {vibeMetadata.optimization.token_budget?.toLocaleString() ?? 'N/A'} tokens
                      </p>
                      <p>
                        <span className="text-zinc-500">Used:</span>{' '}
                        {vibeMetadata.optimization.tokens_used?.toLocaleString() ?? 'N/A'} tokens
                      </p>
                      {vibeMetadata.optimization.files_pruned !== undefined && vibeMetadata.optimization.files_pruned > 0 && (
                        <p className="text-amber-400">
                          <span className="text-zinc-500">Pruned:</span>{' '}
                          {vibeMetadata.optimization.files_pruned} files
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
