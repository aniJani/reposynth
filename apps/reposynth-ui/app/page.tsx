'use client';

import { UrlInput } from '@/components/UrlInput';
import { ConfiguratorPanel } from '@/components/ConfiguratorPanel';
import { EstimatorDisplay } from '@/components/EstimatorDisplay';
import { SubmitButton } from '@/components/SubmitButton';
import { JobStatusDisplay } from '@/components/JobStatusDisplay';
import { VibeStationDrawer } from '@/components/VibeStationDrawer';
import { Database, Sparkles } from 'lucide-react';
import { useState } from 'react';
import { useStore } from '@/lib/store';

export default function Home() {
  const [isVibeDrawerOpen, setIsVibeDrawerOpen] = useState(false);
  const { currentJob, repoUrl } = useStore();

  const showVibeButton = currentJob?.status === 'completed';
  const showConfiguration = repoUrl.trim().length > 0;

  return (
    <div className="relative flex min-h-screen w-full flex-col overflow-x-hidden">
      {/* Fixed Header - Top Left */}
      <header className="fixed top-0 left-0 right-0 z-20 bg-zinc-950/80 backdrop-blur-md">
        <div className="px-6 py-4">
          <div className="flex items-center gap-3">
            <Database className="text-primary h-6 w-6" />
            <div>
              <h1 className="text-lg font-bold font-display tracking-tight text-zinc-200">
                RepoSynth
              </h1>
              <p className="text-xs text-zinc-500">Repository Synthesis & LLM Context Engine</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 pt-20">
        {!showConfiguration ? (
          /* Centered URL Input - Initial State */
          <div className="flex items-center justify-center min-h-[calc(100vh-5rem)] px-4">
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
          /* Configuration Panels - After URL Entry */
          <div className="max-w-5xl mx-auto px-4 md:px-8 pb-16">
            <div className="space-y-8">
              {/* URL Input - Compact */}
              <div className="w-full">
                <UrlInput />
              </div>

              {/* Configuration Panel */}
              <div className="w-full">
                <ConfiguratorPanel />
              </div>

              {/* Estimator Display */}
              <div className="w-full">
                <EstimatorDisplay />
              </div>

              {/* Submit Button */}
              <div className="w-full pt-4">
                <SubmitButton />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer - Minimal */}
      <footer className="py-6 text-center border-t border-zinc-800/50 bg-zinc-950/50">
        <div className="flex items-center justify-center gap-6 text-xs text-zinc-600">
          <a href="#" className="hover:text-zinc-400 transition-colors">
            Documentation
          </a>
          <span>•</span>
          <a href="#" className="hover:text-zinc-400 transition-colors">
            About
          </a>
          <span>•</span>
          <a href="#" className="hover:text-zinc-400 transition-colors">
            GitHub
          </a>
        </div>
      </footer>

      {/* Fixed Job Status Display - Bottom Left */}
      {currentJob && (
        <div className="fixed bottom-8 left-8 w-full max-w-2xl z-30">
          <JobStatusDisplay />
        </div>
      )}

      {/* Floating Vibe Station Button */}
      {showVibeButton && (
        <button
          onClick={() => setIsVibeDrawerOpen(true)}
          className="fixed bottom-8 right-8 flex items-center gap-3 px-6 py-4 bg-gradient-to-r from-neon-purple to-violet-600 text-white rounded-full shadow-2xl hover:shadow-neon-purple/50 hover:scale-105 transition-all font-bold text-base z-30 glow-purple"
        >
          <Sparkles className="h-5 w-5" />
          <span>Vibe Station</span>
        </button>
      )}

      {/* Vibe Station Drawer */}
      <VibeStationDrawer
        isOpen={isVibeDrawerOpen}
        onClose={() => setIsVibeDrawerOpen(false)}
      />
    </div>
  );
}
