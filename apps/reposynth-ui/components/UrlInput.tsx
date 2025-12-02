//components/UrlInput.tsx
'use client';

import { useStore } from '@/lib/store';
import { AlertCircle, CheckCircle } from 'lucide-react';
import { useMemo } from 'react';

export function UrlInput() {
  const { repoUrl, setRepoUrl } = useStore();

  const validation = useMemo(() => {
    if (!repoUrl) return { valid: null, message: '' };

    // Check if it's a GitHub URL
    if (!repoUrl.includes('github.com')) {
      return { valid: false, message: 'Only GitHub repositories are supported' };
    }

    // Check if it's a PR or issue URL
    if (repoUrl.includes('/pull/') || repoUrl.includes('/issues/')) {
      return { valid: false, message: 'Please provide repository URL, not PR/issue' };
    }

    // Check if it's missing the repo name
    if (repoUrl.match(/github\.com\/[^/]+\/?$/)) {
      return { valid: false, message: 'Missing repository name' };
    }

    // Check for valid GitHub repo pattern
    const match = repoUrl.match(/github\.com\/([^/]+)\/([^/]+)/);
    if (!match) {
      return { valid: false, message: 'Invalid GitHub URL format' };
    }

    const [, owner, repo] = match;
    if (!owner || !repo) {
      return { valid: false, message: 'Invalid repository format' };
    }

    return { valid: true, message: `${owner}/${repo.replace(/\.git$/, '')}` };
  }, [repoUrl]);

  const borderColor = validation.valid === false
    ? 'border-red-500'
    : validation.valid === true
      ? 'border-green-500'
      : 'border-zinc-800 hover:border-zinc-700';

  return (
    <div className="w-full">
      <label htmlFor="repo-url" className="flex flex-col">
        <p className="text-zinc-400 text-sm font-mono leading-normal pb-2 px-1">
          [UrlInput]
        </p>
        <div className="relative flex items-center group">
          <span className="absolute left-4 text-zinc-400 select-none">&gt;</span>
          <input
            type="text"
            id="repo-url"
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            placeholder="e.g., https://github.com/facebook/react"
            className={`form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-md text-zinc-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-950 focus:ring-primary border bg-zinc-900 h-12 placeholder:text-zinc-600 pl-8 pr-10 text-base font-normal leading-normal transition-colors ${borderColor}`}
          />
          {validation.valid !== null && (
            <div className="absolute right-3">
              {validation.valid ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <AlertCircle className="h-5 w-5 text-red-500" />
              )}
            </div>
          )}
        </div>
        {validation.message && (
          <div className={`mt-2 text-sm px-1 flex items-center gap-2 ${validation.valid ? 'text-green-400' : 'text-red-400'
            }`}>
            {validation.valid ? (
              <span>Repository: {validation.message}</span>
            ) : (
              <span>{validation.message}</span>
            )}
          </div>
        )}
      </label>
    </div>
  );
}
