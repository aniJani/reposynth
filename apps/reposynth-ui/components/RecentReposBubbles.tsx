'use client';

import { useEffect, useState, useMemo } from 'react';
import Link from 'next/link';
import { getRecentJobs, RecentJob } from '@/lib/api';

interface BubblePosition {
  x: number;
  y: number;
  delay: number;
  duration: number;
}

/**
 * Extract owner/repo from a GitHub URL.
 */
function parseGitHubUrl(url: string): { owner: string; repo: string } | null {
  const match = url.match(/github\.com\/([^/]+)\/([^/]+)/);
  if (!match) return null;
  return {
    owner: match[1],
    repo: match[2].replace(/\.git$/, ''),
  };
}

/**
 * Generate random positions for bubbles that don't overlap.
 */
function generateBubblePositions(count: number): BubblePosition[] {
  const positions: BubblePosition[] = [];

  for (let i = 0; i < count; i++) {
    // Distribute bubbles across the width, avoiding the center where the input is
    const side = i % 2 === 0 ? 'left' : 'right';
    const x = side === 'left'
      ? 5 + Math.random() * 25 // 5-30% from left
      : 65 + Math.random() * 25; // 65-90% from left

    // Vertical position - spread across height, avoiding the very center
    const y = 15 + (Math.random() * 70); // 15-85% from top

    positions.push({
      x,
      y,
      delay: Math.random() * 2, // Random animation delay 0-2s
      duration: 3 + Math.random() * 2, // Animation duration 3-5s
    });
  }

  return positions;
}

export function RecentReposBubbles() {
  const [recentRepos, setRecentRepos] = useState<RecentJob[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchRecentRepos() {
      try {
        const response = await getRecentJobs(8); // Get up to 8 recent repos
        // Filter to unique repos and shuffle for randomness
        const uniqueRepos = response.jobs.reduce((acc: RecentJob[], job) => {
          const parsed = parseGitHubUrl(job.repo_url);
          if (parsed && !acc.some(j => parseGitHubUrl(j.repo_url)?.repo === parsed.repo)) {
            acc.push(job);
          }
          return acc;
        }, []);

        // Shuffle the array
        const shuffled = uniqueRepos.sort(() => Math.random() - 0.5);
        setRecentRepos(shuffled.slice(0, 6)); // Show max 6 bubbles
      } catch (error) {
        console.error('Failed to fetch recent repos:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchRecentRepos();
  }, []);

  const bubblePositions = useMemo(
    () => generateBubblePositions(recentRepos.length),
    [recentRepos.length]
  );

  if (isLoading || recentRepos.length === 0) {
    return null;
  }

  return (
    <>
    {/* Hint text - positioned below the main content */}
    <div className="absolute bottom-8 left-0 right-0 text-center pointer-events-none z-10">
      <p className="text-zinc-500 text-sm">
        or click a recently analyzed repo
      </p>
    </div>

    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {recentRepos.map((job, index) => {
        const parsed = parseGitHubUrl(job.repo_url);
        if (!parsed) return null;

        const position = bubblePositions[index];
        const repoPath = `/${parsed.owner}/${parsed.repo}`;

        return (
          <Link
            key={job.id}
            href={repoPath}
            className="pointer-events-auto absolute group"
            style={{
              left: `${position.x}%`,
              top: `${position.y}%`,
              animation: `float ${position.duration}s ease-in-out ${position.delay}s infinite`,
            }}
          >
            <div className="relative px-4 py-2 bg-zinc-800/60 backdrop-blur-sm border border-zinc-700/50 rounded-full text-sm text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/70 hover:border-zinc-600 transition-all duration-200 shadow-lg hover:shadow-xl hover:scale-105 cursor-pointer">
              <span className="font-medium text-zinc-300 group-hover:text-white">
                {parsed.owner}
              </span>
              <span className="text-zinc-500">/</span>
              <span className="text-teal-400 group-hover:text-teal-300">
                {parsed.repo}
              </span>
            </div>
          </Link>
        );
      })}

      <style jsx global>{`
        @keyframes float {
          0%, 100% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-10px);
          }
        }
      `}</style>
    </div>
    </>
  );
}
