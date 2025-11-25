// components/ConfiguratorPanel.tsx
'use client';

import { useStore } from '@/lib/store';
import { Settings, Zap, Layers, BarChart, Shield, Brain } from 'lucide-react';
import { useEffect } from 'react';
import { estimateTokens } from '@/lib/api';
import { debounce } from '@/lib/utils';

export function ConfiguratorPanel() {
  const {
    config,
    setConfig,
    repoUrl,
    setEstimate,
    setIsEstimating,
    setEstimateError,
  } = useStore();

  // Debounced estimation
  useEffect(() => {
    if (!repoUrl) {
      setEstimate(null);
      return;
    }

    const fetchEstimate = debounce(async () => {
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

    fetchEstimate();
  }, [repoUrl, config, setEstimate, setIsEstimating, setEstimateError]);

  return (
    <div className="w-full max-w-4xl mx-auto bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center gap-2 mb-6">
        <Settings className="h-5 w-5 text-gray-600" />
        <h2 className="text-lg font-semibold text-gray-900">
          Packaging Configuration
        </h2>
      </div>

      {/* Analysis Mode */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Analysis Mode
        </label>
        <div className="grid grid-cols-3 gap-3">
          {['semantic', 'hybrid', 'full'].map((mode) => (
            <button
              key={mode}
              onClick={() => setConfig({ mode: mode as any })}
              className={`px-4 py-3 rounded-lg border-2 transition-all ${
                config.mode === mode
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200 bg-white hover:border-gray-300 text-gray-700'
              }`}
            >
              <div className="font-medium capitalize">{mode}</div>
              <div className="text-xs mt-1" style={{ color: config.mode === mode ? '#1d4ed8' : '#6b7280' }}>
                {mode === 'semantic' && '~30s'}
                {mode === 'hybrid' && '~60s'}
                {mode === 'full' && '~120s'}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Output Format */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Output Format
        </label>
        <div className="grid grid-cols-3 gap-3">
          {[
            { value: 'zip', label: 'ZIP Archive', description: 'Complete package' },
            { value: 'markdown', label: 'Markdown', description: 'Human-readable document' },
            { value: 'toon', label: 'TOON', description: 'Token-Optimized Object Notation' }
          ].map((format) => (
            <button
              key={format.value}
              onClick={() => setConfig({ output_format: format.value as any })}
              className={`px-4 py-3 rounded-lg border-2 transition-all ${
                config.output_format === format.value
                  ? 'border-green-500 bg-green-50 text-green-700'
                  : 'border-gray-200 bg-white hover:border-gray-300 text-gray-700'
              }`}
            >
              <div className="font-medium">{format.label}</div>
              <div className="text-xs mt-1" style={{ color: config.output_format === format.value ? '#15803d' : '#6b7280' }}>
                {format.description}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Feature Toggles */}
      <div className="space-y-3">
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Features
        </label>

        <FeatureToggle
          icon={<Layers className="h-4 w-4" />}
          label="AST Parsing"
          description="Parse abstract syntax trees for structural analysis"
          enabled={config.enable_ast}
          onChange={(enabled) => setConfig({ enable_ast: enabled })}
        />

        <FeatureToggle
          icon={<Zap className="h-4 w-4" />}
          label="Import Graph"
          description="Analyze module dependencies and relationships"
          enabled={config.enable_imports}
          onChange={(enabled) => setConfig({ enable_imports: enabled })}
        />

        <FeatureToggle
          icon={<BarChart className="h-4 w-4" />}
          label="Code Complexity"
          description="Calculate cyclomatic complexity metrics"
          enabled={config.enable_complexity}
          onChange={(enabled) => setConfig({ enable_complexity: enabled })}
        />

        <FeatureToggle
          icon={<Shield className="h-4 w-4" />}
          label="Security Scanning"
          description="Detect secrets and security vulnerabilities"
          enabled={config.enable_security}
          onChange={(enabled) => setConfig({ enable_security: enabled })}
        />

        <FeatureToggle
          icon={<Brain className="h-4 w-4" />}
          label="Semantic Embeddings"
          description="Generate vector embeddings for semantic search"
          enabled={config.enable_embeddings}
          onChange={(enabled) => setConfig({ enable_embeddings: enabled })}
        />
      </div>
    </div>
  );
}

interface FeatureToggleProps {
  icon: React.ReactNode;
  label: string;
  description: string;
  enabled: boolean;
  onChange: (enabled: boolean) => void;
}

function FeatureToggle({
  icon,
  label,
  description,
  enabled,
  onChange,
}: FeatureToggleProps) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-lg hover:bg-gray-50 transition-colors">
      <div className="flex items-center h-5">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        />
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium text-gray-900">{label}</span>
        </div>
        <p className="text-xs text-gray-500 mt-1">{description}</p>
      </div>
    </div>
  );
}
