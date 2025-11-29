import { create } from 'zustand';

export interface JobConfiguration {
  mode: 'semantic' | 'hybrid' | 'full';
  enable_ast: boolean;
  enable_imports: boolean;
  enable_complexity: boolean;
  enable_security: boolean;
  enable_embeddings: boolean;
  output_format: 'zip' | 'markdown' | 'json' | 'toon';
}

export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset_at: string;
}

export interface EstimateResponse {
  estimated_tokens: number;
  estimated_time_seconds: number;
  estimated_cost_usd: number;
  mode: string;
  features_enabled: Record<string, boolean>;
  warnings: string[];
  rate_limit?: RateLimitInfo;
}

export interface JobStatus {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  repo_url: string;
  mode: string;
  created_at: string;
  result_url?: string;
  error_message?: string;
  processing_time_seconds?: number;
}

export interface VibeMetadata {
  mode: string;
  token_estimate: number;
  files_included?: number;
  description?: string;
  query?: string;
  entry_point?: string;
}

interface StoreState {
  // Rate Limit Info
  rateLimit: RateLimitInfo | null;
  setRateLimit: (rateLimit: RateLimitInfo | null) => void;

  // Repository URL
  repoUrl: string;
  setRepoUrl: (url: string) => void;

  // Configuration
  config: JobConfiguration;
  setConfig: (config: Partial<JobConfiguration>) => void;

  // Estimation
  estimate: EstimateResponse | null;
  setEstimate: (estimate: EstimateResponse | null) => void;
  isEstimating: boolean;
  setIsEstimating: (isEstimating: boolean) => void;
  estimateError: string | null;
  setEstimateError: (error: string | null) => void;

  // Job submission
  isSubmitting: boolean;
  setIsSubmitting: (isSubmitting: boolean) => void;
  currentJob: JobStatus | null;
  setCurrentJob: (job: JobStatus | null) => void;
  submitError: string | null;
  setSubmitError: (error: string | null) => void;

  // Vibe Coding
  vibeMode: 'blueprint' | 'focus' | 'bundle';
  setVibeMode: (mode: 'blueprint' | 'focus' | 'bundle') => void;
  vibeQuery: string;
  setVibeQuery: (query: string) => void;
  vibeEntryPoint: string;
  setVibeEntryPoint: (entryPoint: string) => void;
  vibePrompt: string;
  setVibePrompt: (prompt: string) => void;
  vibeMetadata: VibeMetadata | null;
  setVibeMetadata: (metadata: VibeMetadata | null) => void;
  vibeFileList: { files: string[]; roots: string[] };
  setVibeFileList: (list: { files: string[]; roots: string[] }) => void;
  isGeneratingPrompt: boolean;
  setIsGeneratingPrompt: (isGenerating: boolean) => void;
  isVibeDrawerOpen: boolean;
  setIsVibeDrawerOpen: (isOpen: boolean) => void;
}

export const useStore = create<StoreState>((set) => ({
  // Rate Limit Info
  rateLimit: null,
  setRateLimit: (rateLimit) => set({ rateLimit }),

  // Repository URL
  repoUrl: '',
  setRepoUrl: (url) => set({ repoUrl: url }),

  // Configuration - default values
  config: {
    mode: 'semantic',
    enable_ast: true,
    enable_imports: true,
    enable_complexity: true,
    enable_security: false,
    enable_embeddings: true,
    output_format: 'zip',
  },
  setConfig: (newConfig) =>
    set((state) => ({
      config: { ...state.config, ...newConfig },
    })),

  // Estimation
  estimate: null,
  setEstimate: (estimate) => set({ estimate }),
  isEstimating: false,
  setIsEstimating: (isEstimating) => set({ isEstimating }),
  estimateError: null,
  setEstimateError: (error) => set({ estimateError: error }),

  // Job submission
  isSubmitting: false,
  setIsSubmitting: (isSubmitting) => set({ isSubmitting }),
  currentJob: null,
  setCurrentJob: (job) => set({ currentJob: job }),
  submitError: null,
  setSubmitError: (error) => set({ submitError: error }),

  // Vibe Coding
  vibeMode: 'blueprint',
  setVibeMode: (mode) => set({ vibeMode: mode }),
  vibeQuery: '',
  setVibeQuery: (query) => set({ vibeQuery: query }),
  vibeEntryPoint: '',
  setVibeEntryPoint: (entryPoint) => set({ vibeEntryPoint: entryPoint }),
  vibePrompt: '',
  setVibePrompt: (prompt) => set({ vibePrompt: prompt }),
  vibeMetadata: null,
  setVibeMetadata: (metadata) => set({ vibeMetadata: metadata }),
  vibeFileList: { files: [], roots: [] },
  setVibeFileList: (list) => set({ vibeFileList: list }),
  isGeneratingPrompt: false,
  setIsGeneratingPrompt: (isGenerating) => set({ isGeneratingPrompt: isGenerating }),
  isVibeDrawerOpen: false,
  setIsVibeDrawerOpen: (isOpen) => set({ isVibeDrawerOpen: isOpen }),
}));
