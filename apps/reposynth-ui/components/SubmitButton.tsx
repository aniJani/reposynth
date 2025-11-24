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
      className={`flex min-w-[84px] w-full cursor-pointer items-center justify-center overflow-hidden rounded-md h-12 px-5 text-base font-bold leading-normal tracking-[0.015em] transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-950 font-display ${
        isDisabled
          ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed opacity-50'
          : 'bg-primary text-zinc-950 hover:bg-teal-400 focus:ring-primary'
      }`}
    >
      {isSubmitting ? (
        <>
          <Loader2 className="h-5 w-5 mr-2 animate-spin" />
          <span className="truncate">Submitting...</span>
        </>
      ) : (
        <span className="truncate">Submit Analysis Job</span>
      )}
    </button>
  );
}
