
// components/VibeCodingPanel.tsx
'use client';

import { useStore } from '@/lib/store';
import { generateVibePrompt, getJobFiles } from '@/lib/api';
import { useState, useEffect } from 'react';
import {
  Sparkles,
  Loader2,
  Copy,
  CheckCircle,
  Search,
  FileCode,
  Layout,
  AlertCircle,
} from 'lucide-react';

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
          // Auto-select first root if available and no entry point set
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

    // Validate mode-specific requirements
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

  const getModeIcon = (mode: string) => {
    switch (mode) {
      case 'blueprint':
        return <Layout className="h-5 w-5" />;
      case 'focus':
        return <Search className="h-5 w-5" />;
      case 'bundle':
        return <FileCode className="h-5 w-5" />;
    }
  };

  const getModeDescription = (mode: string) => {
    switch (mode) {
      case 'blueprint':
        return 'Structure only (5-10K tokens) - Architecture overview without source code';
      case 'focus':
        return 'Structure + relevant files (20-50K tokens) - Query-based file selection';
      case 'bundle':
        return 'Structure + dependency slice (50-200K+ tokens) - Complete dependency tree';
    }
  };

  const formatTokenCount = (count: number) => {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}K`;
    }
    return count.toString();
  };

  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-blue-600 px-6 py-4">
          <div className="flex items-center gap-3">
            <Sparkles className="h-6 w-6 text-white" />
            <div>
              <h2 className="text-xl font-bold text-white">Vibe Station</h2>
              <p className="text-sm text-purple-100">
                Generate LLM-optimized prompts from your analysis pack
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
          {/* Left Column: Configuration */}
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Compression Mode
              </h3>

              {/* Mode Selection */}
              <div className="space-y-3">
                {(['blueprint', 'focus', 'bundle'] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setVibeMode(mode)}
                    className={`w-full p-4 rounded-lg border-2 transition-all text-left ${
                      vibeMode === mode
                        ? 'border-purple-600 bg-purple-50'
                        : 'border-gray-200 hover:border-purple-300 bg-white'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={`flex-shrink-0 mt-0.5 ${
                          vibeMode === mode ? 'text-purple-600' : 'text-gray-400'
                        }`}
                      >
                        {getModeIcon(mode)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span
                            className={`font-semibold capitalize ${
                              vibeMode === mode ? 'text-purple-900' : 'text-gray-900'
                            }`}
                          >
                            {mode} Mode
                          </span>
                          {vibeMode === mode && (
                            <CheckCircle className="h-4 w-4 text-purple-600" />
                          )}
                        </div>
                        <p className="text-sm text-gray-600 mt-1">
                          {getModeDescription(mode)}
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Mode-Specific Inputs */}
            {vibeMode === 'focus' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  What are you working on?
                </label>
                <textarea
                  value={vibeQuery}
                  onChange={(e) => setVibeQuery(e.target.value)}
                  placeholder="e.g., How does authentication work? or Fix the login bug"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                  rows={3}
                />
              </div>
            )}

            {vibeMode === 'bundle' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Entry Point File
                </label>
                {isLoadingFiles ? (
                  <div className="flex items-center gap-2 text-gray-500 px-4 py-3 border border-gray-300 rounded-lg bg-gray-50">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading file list...
                  </div>
                ) : (
                  <select
                    value={vibeEntryPoint}
                    onChange={(e) => setVibeEntryPoint(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white"
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
                )}
                <p className="text-sm text-gray-500 mt-2">
                  The starting point for dependency traversal
                </p>
              </div>
            )}

            {/* Generate Button */}
            <button
              onClick={handleGeneratePrompt}
              disabled={
                isGeneratingPrompt ||
                !currentJob ||
                currentJob.status !== 'completed'
              }
              className={`w-full px-6 py-4 rounded-lg font-semibold text-lg transition-all flex items-center justify-center gap-3 ${
                isGeneratingPrompt ||
                !currentJob ||
                currentJob.status !== 'completed'
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-purple-600 text-white hover:bg-purple-700 shadow-lg hover:shadow-xl'
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

            {!currentJob || currentJob.status !== 'completed' ? (
              <div className="flex items-start gap-2 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <AlertCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-blue-700">
                  Complete a job first to use the Vibe Station. Submit a repository for
                  analysis above.
                </p>
              </div>
            ) : null}

            {error && (
              <div className="flex items-start gap-2 p-4 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column: Output */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Output</h3>
              {vibePrompt && vibeMetadata && (
                <div className="flex items-center gap-4">
                  <div className="text-sm text-gray-600">
                    <span className="font-semibold">
                      {formatTokenCount(vibeMetadata.token_estimate)}
                    </span>{' '}
                    tokens
                  </div>
                  <button
                    onClick={handleCopyPrompt}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    {copied ? (
                      <>
                        <CheckCircle className="h-4 w-4" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="h-4 w-4" />
                        Copy to Clipboard
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>

            {vibePrompt ? (
              <div className="relative">
                <textarea
                  value={vibePrompt}
                  readOnly
                  className="w-full h-[500px] px-4 py-3 border border-gray-300 rounded-lg bg-gray-50 font-mono text-sm text-gray-900 resize-none focus:outline-none"
                />
                {vibeMetadata && (
                  <div className="mt-3 p-4 bg-gray-100 rounded-lg">
                    <h4 className="text-sm font-semibold text-gray-900 mb-2">
                      Metadata
                    </h4>
                    <div className="space-y-1 text-sm text-gray-600">
                      <p>
                        <span className="font-medium">Mode:</span> {vibeMetadata.mode}
                      </p>
                      <p>
                        <span className="font-medium">Tokens:</span>{' '}
                        {vibeMetadata.token_estimate.toLocaleString()}
                      </p>
                      {vibeMetadata.files_included !== undefined && (
                        <p>
                          <span className="font-medium">Files:</span>{' '}
                          {vibeMetadata.files_included}
                        </p>
                      )}
                      {vibeMetadata.description && (
                        <p className="text-gray-500 italic">{vibeMetadata.description}</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-[500px] border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center">
                <div className="text-center px-6">
                  <Sparkles className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500 font-medium">No prompt generated yet</p>
                  <p className="text-sm text-gray-400 mt-2">
                    Select a mode and click "Generate Prompt" to create your LLM-optimized
                    context
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
