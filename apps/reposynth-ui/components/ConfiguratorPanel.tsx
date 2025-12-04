// components/ConfiguratorPanel.tsx
'use client';

import { useStore } from '@/lib/store';
import { Zap, Layers, BarChart, Shield, Brain } from 'lucide-react';

export function ConfiguratorPanel() {
  const {
    config,
    setConfig,
  } = useStore();

  const modeConfig = [
    { value: 'hybrid', label: 'Hybrid', time: '~30s', description: 'Balanced detail' },
    { value: 'full', label: 'Full', time: '~1m', description: 'Deep inspection' },
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
    balanced: {
      mode: 'hybrid' as const,
      enable_ast: true,
      enable_imports: true,
      enable_complexity: true,
      enable_security: false,
      enable_embeddings: true,
      output_format: 'toon' as const,
    },
    deepDive: {
      mode: 'full' as const,
      enable_ast: true,
      enable_imports: true,
      enable_complexity: true,
      enable_security: true,
      enable_embeddings: true,
      output_format: 'toon' as const,
    },
  };

  const applyPreset = (preset: keyof typeof presets) => {
    setConfig(presets[preset]);
  };

  return (
    <div className="w-full space-y-2 p-5 border border-zinc-800 rounded-lg bg-zinc-900/80">
      <div className="flex items-center justify-between">
        <p className="text-zinc-400 text-sm font-mono">[ConfiguratorPanel]</p>

        {/* Presets Dropdown */}
        <select
          onChange={(e) => e.target.value && applyPreset(e.target.value as keyof typeof presets)}
          className="text-xs bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-zinc-300 focus:ring-2 focus:ring-primary focus:border-primary"
          defaultValue=""
        >
          <option value="" disabled>Quick Presets</option>
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
            <div className="grid grid-cols-2 gap-2">
              {modeConfig.map((mode) => (
                <button
                  key={mode.value}
                  onClick={() => setConfig({ mode: mode.value })}
                  className={`relative flex flex-col items-center justify-center p-4 rounded-lg border transition-all cursor-pointer text-left h-32 ${
                    config.mode === mode.value
                      ? 'border-accent bg-accent/10'
                      : 'bg-zinc-900 border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800/50'
                  }`}
                >
                  <span
                    className={`absolute top-2 right-2 text-xs font-mono px-2 py-0.5 rounded ${
                      config.mode === mode.value
                        ? 'bg-accent/20 text-accent'
                        : 'bg-zinc-800 text-zinc-500'
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
                    className={`flex items-center justify-between p-3 rounded-lg border transition-all ${
                      isEnabled
                        ? 'bg-accent/10 border-accent/40'
                        : 'bg-zinc-900 border-zinc-700 hover:border-zinc-600'
                    }`}
                  >
                    <span
                      className={`font-mono text-sm ${
                        isEnabled ? 'text-accent' : 'text-zinc-300'
                      }`}
                    >
                      {feature.label}
                    </span>
                    <div
                      className={`w-9 h-5 flex items-center rounded-full p-0.5 transition-colors ${
                        isEnabled
                          ? 'bg-accent justify-end'
                          : 'bg-zinc-600 justify-start'
                      }`}
                    >
                      <div
                        className={`w-4 h-4 rounded-full transition-colors ${
                          isEnabled ? 'bg-white' : 'bg-zinc-400'
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
