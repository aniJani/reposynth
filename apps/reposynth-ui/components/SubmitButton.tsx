// components/SubmitButton.tsx
'use client';

import { useStore } from '@/lib/store';
import { createJob, getRateLimitStatus } from '@/lib/api';
import { Rocket, Loader2 } from 'lucide-react';

export function SubmitButton() {
  const {
    repoUrl,
    config,
    isSubmitting,
    setIsSubmitting,
    setCurrentJob,
    setSubmitError,
    rateLimit,
    setRateLimit,
  } = useStore();

  const handleSubmit = async () => {
    if (!repoUrl) {
      setSubmitError('Please enter a repository URL');
      return;
    }

    // Check if rate limit is exhausted (skip check if unlimited, indicated by -1)
    if (rateLimit && rateLimit.remaining !== -1 && rateLimit.remaining <= 0) {
      setSubmitError('Daily API limit reached. Please wait for reset.');
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

      // Refresh rate limit info after successful job submission
      try {
        const newRateLimit = await getRateLimitStatus();
        setRateLimit(newRateLimit);
      } catch {
        // Ignore rate limit refresh errors
      }
    } catch (error: any) {
      // Handle rate limit errors specifically
      if (error.response?.status === 429) {
        const detail = error.response?.data?.detail;
        if (detail && typeof detail === 'object') {
          setRateLimit({
            limit: detail.limit || 5,
            remaining: 0,
            reset_at: detail.reset_at || new Date(Date.now() + 86400000).toISOString(),
          });
          setSubmitError(`Daily limit exceeded. Resets at ${new Date(detail.reset_at).toLocaleTimeString()}`);
        } else {
          setSubmitError('Daily API limit reached. Please try again later.');
        }
      } else {
        const message = error.response?.data?.detail || 'Failed to submit job';
        setSubmitError(typeof message === 'string' ? message : JSON.stringify(message));
      }
      console.error('Job submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Rate limit is exhausted only if remaining is 0 or less (but not -1 which means unlimited)
  const isRateLimitExhausted = rateLimit ? (rateLimit.remaining !== -1 && rateLimit.remaining <= 0) : false;
  const isDisabled = !repoUrl || isSubmitting || isRateLimitExhausted;

  return (
    <button
      onClick={handleSubmit}
      disabled={isDisabled}
      className={`flex min-w-[84px] w-full cursor-pointer items-center justify-center overflow-hidden rounded-md h-12 px-5 text-base font-bold leading-normal tracking-[0.015em] transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-950 font-display ${
        isDisabled
          ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed opacity-50'
          : 'bg-teal-600 text-white hover:bg-teal-500 focus:ring-teal-600'
      }`}
    >
      {isSubmitting ? (
        <>
          <Loader2 className="h-5 w-5 mr-2 animate-spin" />
          <span className="truncate">Submitting...</span>
        </>
      ) : isRateLimitExhausted ? (
        <>
          <Rocket className="h-5 w-5 mr-2 opacity-50" />
          <span className="truncate">Daily Limit Reached</span>
        </>
      ) : (
        <>
          <Rocket className="h-5 w-5 mr-2" />
          <span className="truncate">Submit Analysis Job</span>
        </>
      )}
    </button>
  );
}
