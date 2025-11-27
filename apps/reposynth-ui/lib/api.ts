import axios from 'axios';
import type { JobConfiguration, EstimateResponse, JobStatus, VibeMetadata } from './store';

// API base URL - defaults to localhost:8000, can be overridden via environment variable
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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
  mode: 'blueprint' | 'focus' | 'bundle';
  query?: string;
  entry_point?: string;
  max_files?: number;
  max_depth?: number;
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
