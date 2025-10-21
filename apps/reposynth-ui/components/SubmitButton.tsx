// components/SubmitButton.tsx
'use client';

import { useStore } from '@/lib/store';
import { createJob } from '@/lib/api';
import { Rocket, Loader2 } from 'lucide-react';

export function SubmitButton() {
  const {
    repoUrl,
    config,
    isSubmitting,
    setIsSubmitting,
    setCurrentJob,
    setSubmitError,
  } = useStore();

  const handleSubmit = async () => {
    if (!repoUrl) {
      setSubmitError('Please enter a repository URL');
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);

    try {
      const response = await createJob({ repo_url: repoUrl, config });
      
      // Set initial job status
      setCurrentJob({
        id: response.job_id,
        status: 'pending',
        repo_url: response.repo_url,
        mode: response.mode,
        created_at: new Date().toISOString(),
      });
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Failed to submit job';
      setSubmitError(message);
      console.error('Job submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const isDisabled = !repoUrl || isSubmitting;

  return (
    <button
      onClick={handleSubmit}
      disabled={isDisabled}
      className={`w-full max-w-4xl mx-auto px-6 py-4 rounded-lg font-semibold text-lg transition-all flex items-center justify-center gap-3 ${
        isDisabled
          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
          : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg hover:shadow-xl'
      }`}
    >
      {isSubmitting ? (
        <>
          <Loader2 className="h-6 w-6 animate-spin" />
          Submitting...
        </>
      ) : (
        <>
          <Rocket className="h-6 w-6" />
          Generate Pack
        </>
      )}
    </button>
  );
}
