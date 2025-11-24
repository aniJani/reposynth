// components/UrlInput.tsx
'use client';

import { useStore } from '@/lib/store';

export function UrlInput() {
  const { repoUrl, setRepoUrl } = useStore();

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
            className="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-md text-zinc-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-950 focus:ring-primary border border-zinc-800 bg-zinc-900 h-12 placeholder:text-zinc-600 pl-8 pr-4 text-base font-normal leading-normal transition-colors hover:border-zinc-700"
          />
        </div>
      </label>
    </div>
  );
}
