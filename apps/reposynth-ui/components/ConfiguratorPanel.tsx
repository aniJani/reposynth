// components/ConfiguratorPanel.tsx
'use client';

import { useStore } from '@/lib/store';
import { Settings, Zap, Layers, BarChart, Shield, Brain } from 'lucide-react';
import { useEffect, useCallback, useRef } from 'react';
import { estimateTokens } from '@/lib/api';

export function ConfiguratorPanel() {
  const {
    config,
    setConfig,
    repoUrl,
    setEstimate,
    setIsEstimating,
    setEstimateError,
  } = useStore();

  // Use a ref to track the debounce timeout
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Debounced estimation
  useEffect(() => {
    if (!repoUrl) {
      setEstimate(null);
      return;
    }

    // Clear previous timeout
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Set new debounced call
    debounceTimerRef.current = setTimeout(async () => {
      setIsEstimating(true);
      setEstimateError(null);
      try {
        const estimate = await estimateTokens({ repo_url: repoUrl, config });
        setEstimate(estimate);
      } catch (error) {
        setEstimateError('Failed to fetch estimate');
        console.error('Estimation error:', error);
      } finally {
        setIsEstimating(false);
      }
    }, 500);

    // Cleanup on unmount or dependency change
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [repoUrl, config, setEstimate, setIsEstimating, setEstimateError]);

  const modeConfig = [
    { value: 'semantic', label: 'Semantic', time: '~30s', description: 'Lightweight analysis' },
    { value: 'hybrid', label: 'Hybrid', time: '~2m', description: 'Balanced detail' },
    { value: 'full', label: 'Full', time: '~10m', description: 'Deep inspection' },
  ] as const;

  const formatConfig = [
    { value: 'zip', label: 'ZIP' },
    { value: 'markdown', label: 'Markdown' },
    { value: 'toon', label: 'TOON' },
  ] as const;

  const featureConfig = [
    { key: 'enable_ast', label: 'AST', icon: Layers },
    { key: 'enable_imports', label: 'Import Graph', icon: Zap },
    { key: 'enable_complexity', label: 'Complexity', icon: BarChart },
    { key: 'enable_security', label: 'Security', icon: Shield },
    { key: 'enable_embeddings', label: 'Embeddings', icon: Brain },
  ] as const;

  // Configuration presets
  const presets = {
    quickScan: {
      mode: 'semantic' as const,
      enable_ast: true,
      enable_imports: false,
      enable_complexity: false,
      enable_security: false,
      enable_embeddings: true,
      output_format: 'markdown' as const,
    },
    balanced: {
      mode: 'hybrid' as const,
      enable_ast: true,
      enable_imports: true,
      enable_complexity: true,
      enable_security: false,
      enable_embeddings: true,
      output_format: 'zip' as const,
    },
    deepDive: {
      mode: 'full' as const,
      enable_ast: true,
      enable_imports: true,
      enable_complexity: true,
      enable_security: true,
      enable_embeddings: true,
      output_format: 'zip' as const,
    },
  };

  const applyPreset = (preset: keyof typeof presets) => {
    setConfig(presets[preset]);
  };

  return (
    <div className="w-full space-y-2 p-4 border border-zinc-800 rounded-md bg-zinc-900/50">
      <div className="flex items-center justify-between">
        <p className="text-zinc-400 text-sm font-mono">[ConfiguratorPanel]</p>

        {/* Presets Dropdown */}
        <select
          onChange={(e) => e.target.value && applyPreset(e.target.value as keyof typeof presets)}
          className="text-xs bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-zinc-300 focus:ring-2 focus:ring-primary focus:border-primary"
          defaultValue=""
        >
          <option value="" disabled>Quick Presets</option>
          <option value="quickScan">Quick Scan</option>
          <option value="balanced">Balanced</option>
          <option value="deepDive">Deep Dive</option>
        </select>
      </div>

      <div className="w-full space-y-8 pt-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Mode Selector */}
          <div className="col-span-1">
            <h3 className="font-display text-zinc-400 text-sm tracking-wider uppercase mb-3 px-1">
              Mode Selector
            </h3>
            <div className="grid grid-cols-3 gap-2">
              {modeConfig.map((mode) => (
                <button
                  key={mode.value}
                  onClick={() => setConfig({ mode: mode.value })}
                  className={`relative flex flex-col items-center justify-center p-4 rounded-lg border-2 transition-colors cursor-pointer text-left h-32 ${
                    config.mode === mode.value
                      ? 'border-neon-purple bg-zinc-800 glow-purple'
                      : 'bg-zinc-900 border-zinc-800 hover:bg-zinc-800'
                  }`}
                >
                  <span
                    className={`absolute top-2 right-2 text-xs font-mono px-2 py-0.5 rounded-full ${
                      config.mode === mode.value
                        ? 'bg-neon-purple/10 text-neon-purple'
                        : 'bg-zinc-700/50 text-zinc-400'
                    }`}
                  >
                    {mode.time}
                  </span>
                  <p className="font-display font-bold text-zinc-200">{mode.label}</p>
                  <p className="text-xs text-zinc-400 mt-1">{mode.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Output Format */}
          <div className="col-span-1">
            <h3 className="font-display text-zinc-400 text-sm tracking-wider uppercase mb-3 px-1">
              Output Format
            </h3>
            <div className="flex items-center bg-zinc-900 border border-zinc-800 rounded-lg p-1 space-x-1">
              {formatConfig.map((format) => (
                <button
                  key={format.value}
                  onClick={() => setConfig({ output_format: format.value })}
                  className={`flex-1 text-center py-2 rounded-md font-mono text-sm transition-colors ${
                    config.output_format === format.value
                      ? 'bg-zinc-800 text-zinc-200'
                      : 'text-zinc-400 hover:bg-zinc-800'
                  }`}
                >
                  {format.label}
                </button>
              ))}
            </div>
          </div>

          {/* Feature Toggles */}
          <div className="col-span-1 md:col-span-3">
            <h3 className="font-display text-zinc-400 text-sm tracking-wider uppercase mb-3 px-1">
              Feature Toggles
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
              {featureConfig.map((feature) => {
                const isEnabled = config[feature.key];
                return (
                  <button
                    key={feature.key}
                    onClick={() => setConfig({ [feature.key]: !isEnabled })}
                    className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${
                      isEnabled
                        ? 'bg-cyber-blue/10 border-cyber-blue/50 hover:bg-cyber-blue/20'
                        : 'bg-zinc-900 border-zinc-800 hover:bg-zinc-800 hover:border-cyber-blue/40'
                    }`}
                  >
                    <span
                      className={`font-mono text-sm ${
                        isEnabled ? 'text-cyber-blue' : 'text-zinc-200'
                      }`}
                    >
                      {feature.label}
                    </span>
                    <div
                      className={`w-10 h-5 flex items-center rounded-full p-1 transition-colors ${
                        isEnabled
                          ? 'bg-cyber-blue/50 justify-end'
                          : 'bg-zinc-700 justify-start'
                      }`}
                    >
                      <div
                        className={`w-3 h-3 rounded-full shadow-md ${
                          isEnabled ? 'bg-zinc-100' : 'bg-zinc-400'
                        }`}
                      />
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
