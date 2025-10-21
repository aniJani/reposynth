// components/UrlInput.tsx
'use client';

import { useStore } from '@/lib/store';
import { Github } from 'lucide-react';

export function UrlInput() {
  const { repoUrl, setRepoUrl } = useStore();

  return (
    <div className="w-full max-w-4xl mx-auto">
      <label htmlFor="repo-url" className="block text-sm font-medium text-gray-700 mb-2">
        GitHub Repository URL
      </label>
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Github className="h-5 w-5 text-gray-400" />
        </div>
        <input
          type="text"
          id="repo-url"
          value={repoUrl}
          onChange={(e) => setRepoUrl(e.target.value)}
          placeholder="https://github.com/username/repository"
          className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-base text-gray-900 bg-white placeholder-gray-400"
        />
      </div>
      <p className="mt-2 text-sm text-gray-500">
        Enter a public GitHub repository URL to analyze
      </p>
    </div>
  );
}
