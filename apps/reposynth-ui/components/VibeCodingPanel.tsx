// components/VibeCodingPanel.tsx
'use client';

import { useStore } from '@/lib/store';
import { generateVibePrompt, getJobFiles } from '@/lib/api';
import { useState, useEffect } from 'react';
import { Sparkles, Loader2, Copy, CheckCircle, PlayCircle, AlertCircle } from 'lucide-react';

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
    isGeneratingPrompt,
    setIsGeneratingPrompt,
  } = useStore();

  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileList, setFileList] = useState<{ files: string[]; roots: string[] }>({
    files: [],
    roots: [],
  });
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);

  useEffect(() => {
    if (vibeMode === 'bundle' && currentJob?.id) {
      setIsLoadingFiles(true);
      getJobFiles(currentJob.id)
        .then((data) => {
          setFileList(data);
          if (data.roots.length > 0 && !vibeEntryPoint) {
            setVibeEntryPoint(data.roots[0]);
          }
        })
        .catch((err) => console.error('Failed to load files:', err))
        .finally(() => setIsLoadingFiles(false));
    }
  }, [vibeMode, currentJob?.id, setVibeEntryPoint, vibeEntryPoint]);

  const handleGeneratePrompt = async () => {
    if (!currentJob || currentJob.status !== 'completed') {
      setError('Please complete a job first before generating a prompt');
      return;
    }

    if (vibeMode === 'focus' && !vibeQuery.trim()) {
      setError('Please enter a query for Focus mode');
      return;
    }

    if (vibeMode === 'bundle' && !vibeEntryPoint.trim()) {
      setError('Please enter an entry point file path for Bundle mode');
      return;
    }

    setIsGeneratingPrompt(true);
    setError(null);
    setVibePrompt('');

    try {
      const response = await generateVibePrompt({
        job_id: currentJob.id,
        mode: vibeMode,
        query: vibeMode === 'focus' ? vibeQuery : undefined,
        entry_point: vibeMode === 'bundle' ? vibeEntryPoint : undefined,
        max_files: 5,
        max_depth: 3,
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
            {/* Mode Tabs */}
            <div className="flex border-b border-zinc-800 -mx-4 px-4">
              {(['blueprint', 'focus', 'bundle'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setVibeMode(mode)}
                  className={`px-4 py-2 text-sm transition-colors ${
                    vibeMode === mode
                      ? 'border-b-2 border-cyber-blue text-white bg-zinc-800/50'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>

            <div className="space-y-6 pt-4">
              {/* Focus Mode - Query */}
              {vibeMode === 'focus' && (
                <div>
                  <label
                    className="text-zinc-400 text-xs font-semibold tracking-wider"
                    htmlFor="query-textarea"
                  >
                    FOCUS_MODE // QUERY
                  </label>
                  <textarea
                    id="query-textarea"
                    value={vibeQuery}
                    onChange={(e) => setVibeQuery(e.target.value)}
                    placeholder="e.g., 'Analyze the component structure and data flow...'"
                    className="mt-2 form-textarea w-full resize-none bg-zinc-950 border border-zinc-800 rounded-md p-3 text-zinc-300 placeholder:text-zinc-600 focus:ring-2 focus:ring-cyber-blue focus:border-cyber-blue transition-colors"
                    rows={5}
                  />
                </div>
              )}

              {/* Bundle Mode - Entry Point */}
              {vibeMode === 'bundle' && (
                <div>
                  <label
                    className="text-zinc-400 text-xs font-semibold tracking-wider"
                    htmlFor="entrypoint-select"
                  >
                    BUNDLE_MODE // ENTRY_POINT
                  </label>
                  <select
                    id="entrypoint-select"
                    value={vibeEntryPoint}
                    onChange={(e) => setVibeEntryPoint(e.target.value)}
                    className="mt-2 form-select w-full bg-zinc-950 border border-zinc-800 rounded-md p-3 text-zinc-300 focus:ring-2 focus:ring-cyber-blue focus:border-cyber-blue transition-colors"
                  >
                    <option value="">Select a file...</option>
                    {fileList.roots.length > 0 && (
                      <optgroup label="⭐ Suggested Entry Points">
                        {fileList.roots.map((f) => (
                          <option key={f} value={f}>
                            {f}
                          </option>
                        ))}
                      </optgroup>
                    )}
                    <optgroup label="All Files">
                      {fileList.files.map((f) => (
                        <option key={f} value={f}>
                          {f}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                </div>
              )}

              {/* Generate Button */}
              <button
                onClick={handleGeneratePrompt}
                disabled={isGeneratingPrompt || !currentJob || currentJob.status !== 'completed'}
                className={`w-full flex items-center justify-center h-10 px-4 font-bold text-sm rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-900 ${
                  isGeneratingPrompt || !currentJob || currentJob.status !== 'completed'
                    ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                    : 'bg-neon-purple text-white hover:bg-violet-600 focus:ring-neon-purple'
                }`}
              >
                <PlayCircle className="h-4 w-4 mr-2" style={{ fontVariationSettings: "'wght' 600" }} />
                {isGeneratingPrompt ? 'GENERATING...' : 'GENERATE PROMPT'}
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
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
