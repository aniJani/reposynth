import axios from 'axios';
import type { JobConfiguration, EstimateResponse, JobStatus, VibeMetadata, RateLimitInfo } from './store';

// API base URL - defaults to localhost:8000, can be overridden via environment variable
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Generate or retrieve a unique device ID for rate limiting.
 * This helps distinguish different devices behind the same IP (NAT/proxy).
 * The ID is stored in localStorage and persists across sessions.
 */
function getDeviceId(): string {
  if (typeof window === 'undefined') {
    // Server-side rendering - return empty string
    return '';
  }
  
  const STORAGE_KEY = 'reposynth_device_id';
  let deviceId = localStorage.getItem(STORAGE_KEY);
  
  if (!deviceId) {
    // Generate a new unique device ID
    deviceId = `${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
    localStorage.setItem(STORAGE_KEY, deviceId);
  }
  
  return deviceId;
}

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add device ID header to all requests (for unique device tracking behind same IP)
apiClient.interceptors.request.use((config) => {
  const deviceId = getDeviceId();
  if (deviceId) {
    config.headers['X-Device-ID'] = deviceId;
  }
  return config;
});

// ============================================================================
// Rate Limit API
// ============================================================================

export interface RateLimitStatusResponse {
  client_id: string;
  daily_limit: number | string;  // Can be "unlimited" in dev mode
  remaining_calls: number | string;  // Can be "unlimited" in dev mode
  reset_at: string | null;
  message: string;
}

export async function getRateLimitStatus(): Promise<RateLimitInfo> {
  const response = await apiClient.get<RateLimitStatusResponse>('/rate-limit-status');
  
  // Handle unlimited (dev mode) response
  if (response.data.daily_limit === 'unlimited' || response.data.daily_limit === -1) {
    return {
      limit: -1,
      remaining: -1,
      reset_at: null,
    };
  }
  
  return {
    limit: response.data.daily_limit as number,
    remaining: response.data.remaining_calls as number,
    reset_at: response.data.reset_at,
  };
}

// ============================================================================
// Estimation API
// ============================================================================

export interface EstimateTokensRequest {
  repo_url: string;
  config: JobConfiguration;
}

export async function estimateTokens(request: EstimateTokensRequest): Promise<EstimateResponse> {
  const response = await apiClient.post<EstimateResponse>('/estimate', request);
  return response.data;
}

// ============================================================================
// Job Queue API
// ============================================================================

export interface CreateJobRequest {
  repo_url: string;
  config: JobConfiguration;
}

export interface CreateJobResponse {
  job_id: string;
  status: string;
  repo_url: string;
  mode: string;
  config: JobConfiguration;
  message: string;
}

export async function createJob(request: CreateJobRequest): Promise<CreateJobResponse> {
  const response = await apiClient.post<CreateJobResponse>(
    `/jobs?repo_url=${encodeURIComponent(request.repo_url)}`,
    request.config
  );
  return response.data;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await apiClient.get<JobStatus>(`/jobs/${jobId}`);
  return response.data;
}

// ============================================================================
// Vibe Coding / Prompt Generation API
// ============================================================================

export interface GenerateVibePromptRequest {
  job_id: string;
  mode: 'blueprint' | 'focus' | 'bundle';  // Backend modes - frontend maps to these
  query?: string;
  entry_point?: string;
  max_files?: number;
  max_depth?: number;
  token_limit?: number;  // Optional token budget for context optimization
}

export interface GenerateVibePromptResponse {
  prompt: string;
  metadata: VibeMetadata;
}

export async function generateVibePrompt(
  request: GenerateVibePromptRequest
): Promise<GenerateVibePromptResponse> {
  const response = await apiClient.post<GenerateVibePromptResponse>('/vibe-prompt', request);
  return response.data;
}

export interface JobFilesResponse {
  files: string[];
  roots: string[];
}

export async function getJobFiles(jobId: string): Promise<JobFilesResponse> {
  const response = await apiClient.get<JobFilesResponse>(`/jobs/${jobId}/files`);
  return response.data;
}

// ============================================================================
// Recent Jobs API (for homepage bubbles)
// ============================================================================

export interface RecentJob {
  id: string;
  repo_url: string;
  status: string;
  mode: string;
  created_at: string;
}

export interface RecentJobsResponse {
  jobs: RecentJob[];
  total: number;
}

export async function getRecentJobs(limit: number = 10): Promise<RecentJobsResponse> {
  const response = await apiClient.get<RecentJobsResponse>('/jobs', {
    params: {
      status_filter: 'completed',
      limit,
    },
  });
  return response.data;
}
