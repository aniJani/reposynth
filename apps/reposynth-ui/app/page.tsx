import { UrlInput } from '@/components/UrlInput';
import { ConfiguratorPanel } from '@/components/ConfiguratorPanel';
import { EstimatorDisplay } from '@/components/EstimatorDisplay';
import { SubmitButton } from '@/components/SubmitButton';
import { JobStatusDisplay } from '@/components/JobStatusDisplay';
import { VibeCodingPanel } from '@/components/VibeCodingPanel';
import { Database } from 'lucide-react';

export default function Home() {
  return (
    <div className="relative flex min-h-screen w-full flex-col overflow-x-hidden">
      <div className="flex h-full grow flex-col">
        <div className="flex flex-1 justify-center items-start p-4 md:p-8">
          <div className="flex flex-col w-full max-w-5xl flex-1">
            {/* Header */}
            <header className="flex flex-col items-center justify-center text-center py-10">
              <div className="flex items-center gap-3">
                <Database className="text-primary text-3xl h-8 w-8" />
                <h1 className="text-zinc-200 text-3xl font-bold font-display tracking-tight">
                  RepoSynth
                </h1>
              </div>
              <p className="text-zinc-400 mt-2 text-base">
                Repository Synthesis & LLM Context Engine
              </p>
            </header>

            {/* Main Content */}
            <main className="flex-grow flex flex-col items-center w-full space-y-8">
              {/* URL Input */}
              <div className="w-full space-y-2">
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
              <div className="w-full pt-2">
                <SubmitButton />
              </div>

              {/* Job Status Display */}
              <div className="w-full">
                <JobStatusDisplay />
              </div>

              {/* Vibe Station */}
              <div className="w-full">
                <VibeCodingPanel />
              </div>
            </main>

            {/* Footer */}
            <footer className="flex flex-col gap-6 px-5 py-10 text-center mt-16">
              <div className="flex flex-wrap items-center justify-center gap-6">
                <a
                  className="text-zinc-500 hover:text-zinc-300 transition-colors text-sm font-normal leading-normal min-w-40"
                  href="#"
                >
                  Documentation
                </a>
                <a
                  className="text-zinc-500 hover:text-zinc-300 transition-colors text-sm font-normal leading-normal min-w-40"
                  href="#"
                >
                  About
                </a>
                <a
                  className="text-zinc-500 hover:text-zinc-300 transition-colors text-sm font-normal leading-normal min-w-40"
                  href="#"
                >
                  GitHub
                </a>
              </div>
              <p className="text-zinc-600 text-sm font-normal leading-normal">
                © 2024 RepoSynth. All rights reserved.
              </p>
            </footer>
          </div>
        </div>
      </div>
    </div>
  );
}
