// components/RateLimitBanner.tsx
'use client';

import { useEffect, useState } from 'react';
import { useStore } from '@/lib/store';
import { getRateLimitStatus } from '@/lib/api';
import { Clock, AlertTriangle, CheckCircle } from 'lucide-react';

export function RateLimitBanner() {
  const { rateLimit, setRateLimit } = useStore();
  const [timeUntilReset, setTimeUntilReset] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch rate limit status on mount
  useEffect(() => {
    async function fetchRateLimit() {
      try {
        setIsLoading(true);
        const status = await getRateLimitStatus();
        setRateLimit(status);
        setError(null);
      } catch (err: any) {
        // Handle 429 rate limit exceeded
        if (err.response?.status === 429) {
          const detail = err.response?.data?.detail;
          if (detail && typeof detail === 'object') {
            setRateLimit({
              limit: detail.limit || 5,
              remaining: 0,
              reset_at: detail.reset_at || new Date(Date.now() + 86400000).toISOString(),
            });
          }
        } else {
          setError('Unable to fetch rate limit status');
        }
      } finally {
        setIsLoading(false);
      }
    }

    fetchRateLimit();
    
    // Refresh every 5 minutes
    const interval = setInterval(fetchRateLimit, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [setRateLimit]);

  // Update countdown timer
  useEffect(() => {
    // Skip countdown if unlimited (reset_at is null) or no rate limit
    if (!rateLimit?.reset_at) return;

    function updateCountdown() {
      const now = new Date();
      const reset = new Date(rateLimit!.reset_at!);
      const diff = reset.getTime() - now.getTime();

      if (diff <= 0) {
        setTimeUntilReset('Resetting...');
        // Refetch after reset
        setTimeout(() => {
          getRateLimitStatus().then(setRateLimit).catch(() => {});
        }, 1000);
        return;
      }

      const hours = Math.floor(diff / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));

      if (hours > 0) {
        setTimeUntilReset(`${hours}h ${minutes}m`);
      } else {
        setTimeUntilReset(`${minutes}m`);
      }
    }

    updateCountdown();
    const interval = setInterval(updateCountdown, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [rateLimit?.reset_at, setRateLimit]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 bg-zinc-800/50 rounded-lg text-xs text-zinc-400">
        <div className="w-3 h-3 border border-zinc-500 border-t-transparent rounded-full animate-spin" />
        <span>Loading...</span>
      </div>
    );
  }

  if (error) {
    // Show error state instead of hiding
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 bg-red-900/30 border border-red-700/50 rounded-lg text-xs text-red-300">
        <AlertTriangle className="h-3 w-3" />
        <span>API unavailable</span>
      </div>
    );
  }

  if (!rateLimit) return null;

  // Check if rate limiting is disabled (unlimited mode, indicated by -1)
  const isUnlimited = rateLimit.limit === -1 || rateLimit.remaining === -1;
  
  if (isUnlimited) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 bg-teal-900/30 border border-teal-700/50 rounded-lg text-xs text-teal-300">
        <CheckCircle className="h-3 w-3" />
        <span>Development Mode - Unlimited API calls</span>
      </div>
    );
  }

  const isExhausted = rateLimit.remaining === 0;
  const isLow = rateLimit.remaining <= 2 && rateLimit.remaining > 0;

  return (
    <div
      className={`flex items-center gap-3 px-4 py-2 rounded-lg text-sm ${
        isExhausted
          ? 'bg-red-900/30 border border-red-700/50 text-red-300'
          : isLow
          ? 'bg-yellow-900/30 border border-yellow-700/50 text-yellow-300'
          : 'bg-zinc-800/50 border border-zinc-700/50 text-zinc-300'
      }`}
    >
      {isExhausted ? (
        <AlertTriangle className="h-4 w-4 text-red-400 flex-shrink-0" />
      ) : isLow ? (
        <AlertTriangle className="h-4 w-4 text-yellow-400 flex-shrink-0" />
      ) : (
        <CheckCircle className="h-4 w-4 text-green-400 flex-shrink-0" />
      )}

      <div className="flex items-center gap-4">
        <span className="font-mono">
          <span className={`font-bold ${isExhausted ? 'text-red-300' : isLow ? 'text-yellow-300' : 'text-zinc-200'}`}>
            {rateLimit.remaining}
          </span>
          <span className="text-zinc-500">/{rateLimit.limit}</span>
          <span className="ml-1 text-zinc-400">calls left</span>
        </span>

        <div className="flex items-center gap-1.5 text-zinc-400">
          <Clock className="h-3.5 w-3.5" />
          <span className="text-xs">
            Resets in <span className="font-mono text-zinc-300">{timeUntilReset}</span>
          </span>
        </div>
      </div>

      {isExhausted && (
        <span className="text-xs text-red-400 ml-2">
          Daily limit reached
        </span>
      )}
    </div>
  );
}
