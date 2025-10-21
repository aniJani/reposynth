import { UrlInput } from '@/components/UrlInput';
import { ConfiguratorPanel } from '@/components/ConfiguratorPanel';
import { EstimatorDisplay } from '@/components/EstimatorDisplay';
import { SubmitButton } from '@/components/SubmitButton';
import { JobStatusDisplay } from '@/components/JobStatusDisplay';
import { Package } from 'lucide-react';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <Package className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-3xl font-bold text-gray-900">RepoSynth</h1>
              <p className="text-sm text-gray-500">
                AI-Powered Repository Analysis
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="space-y-8">
          {/* URL Input */}
          <section>
            <UrlInput />
          </section>

          {/* Configuration Panel */}
          <section>
            <ConfiguratorPanel />
          </section>

          {/* Estimator Display */}
          <section>
            <EstimatorDisplay />
          </section>

          {/* Submit Button */}
          <section>
            <SubmitButton />
          </section>

          {/* Job Status Display */}
          <section>
            <JobStatusDisplay />
          </section>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-16 py-8 bg-white border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            RepoSynth © 2025 - Built with Next.js and FastAPI
          </p>
        </div>
      </footer>
    </main>
  );
}
