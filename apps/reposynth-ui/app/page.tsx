'use client';

import { UrlInput } from '@/components/UrlInput';
import { ConfiguratorPanel } from '@/components/ConfiguratorPanel';
import { SubmitButton } from '@/components/SubmitButton';
import { JobProgressPanel } from '@/components/JobProgressPanel';
import { VibeStationDrawer } from '@/components/VibeStationDrawer';
import { RateLimitBanner } from '@/components/RateLimitBanner';
import { Sparkles } from 'lucide-react';
import Image from 'next/image';
import { useEffect, useRef } from 'react';
import { useStore } from '@/lib/store';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function Home() {
  const { currentJob, repoUrl, isVibeDrawerOpen, setIsVibeDrawerOpen } = useStore();
  const prevJobStatusRef = useRef<string | null>(null);
  const router = useRouter();

  // Auto-redirect to results page when job completes
  useEffect(() => {
    if (currentJob?.status === 'completed' && prevJobStatusRef.current !== 'completed') {
      // Redirect to shareable results page
      router.push(`/results/${currentJob.id}`);
    }
    prevJobStatusRef.current = currentJob?.status ?? null;
  }, [currentJob?.status, currentJob?.id, router]);

  const showVibeButton = currentJob?.status === 'completed';
  const showConfiguration = repoUrl.trim().length > 0;
  const hasActiveJob = currentJob && currentJob.status !== 'failed';

  return (
    <div className="relative flex h-screen w-full flex-col overflow-hidden">
      {/* Fixed Header - Top Left */}
      <header className="fixed top-0 left-0 right-0 z-20 bg-zinc-950/80 backdrop-blur-md">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Image 
                src="/logo.png" 
                alt="RepoSynth Logo" 
                width={32} 
                height={32}
              />
              <div>
                <h1 className="text-lg font-bold font-display tracking-tight text-zinc-200">
                  RepoSynth
                </h1>
                <p className="text-xs text-zinc-500">Repository Synthesis & LLM Context Engine</p>
              </div>
            </div>
            <RateLimitBanner />
          </div>
        </div>
      </header>

      {/* Main Content - slides left when vibe drawer opens */}
      <main 
        className={`flex-1 pt-20 overflow-y-auto transition-all duration-300 ease-in-out ${
          isVibeDrawerOpen ? 'mr-[50%] lg:mr-[40%]' : 'mr-0'
        }`}
      >
        {!showConfiguration ? (
          /* Centered URL Input - Initial State */
          <div className="flex items-center justify-center h-full px-4">
            <div className="w-full max-w-2xl">
              <div className="text-center mb-8">
                <h2 className="text-4xl md:text-5xl font-bold text-zinc-200 mb-4 font-display tracking-tight">
                  Analyze a Repository
                </h2>
                <p className="text-zinc-400 text-lg">
                  Enter a GitHub repository URL to begin
                </p>
              </div>
              <UrlInput />
            </div>
          </div>
        ) : (
          /* Configuration / Job Progress - After URL Entry */
          <div className="max-w-5xl mx-auto px-4 md:px-8 py-4">
            <div className="space-y-6">
              {/* URL Input - Compact */}
              <div className="w-full">
                <UrlInput />
              </div>

              {/* Show Configuration Panel OR Job Progress Panel */}
              {!hasActiveJob ? (
                <>
                  {/* Configuration Panel */}
                  <div className="w-full">
                    <ConfiguratorPanel />
                  </div>

                  {/* Submit Button */}
                  <div className="w-full pt-2">
                    <SubmitButton />
                  </div>
                </>
              ) : (
                /* Job Progress Panel - replaces config when job is active */
                <div className="w-full">
                  <JobProgressPanel />
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer - Minimal */}
      <footer className={`py-4 text-center border-t border-zinc-800/50 bg-zinc-950/50 transition-all duration-300 ease-in-out ${
        isVibeDrawerOpen ? 'mr-[50%] lg:mr-[40%]' : 'mr-0'
      }`}>
        <div className="text-xs text-zinc-600">
          <Link href="/about" className="hover:text-zinc-400 transition-colors">
            About
          </Link>
        </div>
      </footer>

      {/* Floating Vibe Station Button - only show when drawer is closed */}
      {showVibeButton && !isVibeDrawerOpen && (
        <button
          onClick={() => setIsVibeDrawerOpen(true)}
          className="fixed bottom-8 right-8 flex items-center gap-3 px-6 py-4 bg-violet-600 text-white rounded-xl hover:bg-violet-500 transition-colors font-semibold text-base z-30"
        >
          <Sparkles className="h-5 w-5" />
          <span>Vibe Station</span>
        </button>
      )}

      {/* Vibe Station Drawer - no backdrop blur */}
      <VibeStationDrawer />
    </div>
  );
}
