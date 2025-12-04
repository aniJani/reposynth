'use client';

import { useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { useStore } from '@/lib/store';

/**
 * Catch-all route handler for GitHub-style URL paths.
 *
 * Supports formats like:
 * - /facebook/react (owner/repo)
 * - /facebook/react/tree/main/src (owner/repo with branch/path)
 *
 * This route pre-fills the URL input and redirects to the home page
 * for configuration before analysis.
 */
export default function RepoRedirectPage() {
  const router = useRouter();
  const params = useParams();
  const { setRepoUrl } = useStore();

  useEffect(() => {
    const pathSegments = params.path as string[];

    if (!pathSegments || pathSegments.length < 2) {
      // Not enough segments for owner/repo, redirect to home
      router.replace('/');
      return;
    }

    // Extract owner and repo from path
    const owner = pathSegments[0];
    const repo = pathSegments[1];

    // Build the full GitHub URL including any additional path segments
    // e.g., /facebook/react/tree/main/src -> https://github.com/facebook/react/tree/main/src
    const fullPath = pathSegments.join('/');
    const githubUrl = `https://github.com/${fullPath}`;

    // Set the repo URL in the store
    setRepoUrl(githubUrl);

    // Redirect to home page where the URL will be pre-filled
    router.replace('/');
  }, [params.path, router, setRepoUrl]);

  // Show a brief loading state while redirecting
  return (
    <div className="flex items-center justify-center h-screen bg-zinc-950">
      <div className="text-center">
        <div className="animate-pulse text-zinc-400 text-lg">
          Loading repository...
        </div>
      </div>
    </div>
  );
}
