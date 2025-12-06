'use client';

import { useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import { getRecentJobs, RecentJob } from '@/lib/api';

interface BubblePosition {
  x: number;
  y: number;
  delay: number;
  duration: number;
}

interface DisplayBubble {
  job: RecentJob;
  position: BubblePosition;
  opacity: number;
  state: 'entering' | 'visible' | 'exiting';
  key: string;
}

// Popular GitHub repos to show as examples when there aren't enough real ones
const SAMPLE_REPOS: RecentJob[] = [
  { id: 'sample-1', repo_url: 'https://github.com/facebook/react', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-2', repo_url: 'https://github.com/vercel/next.js', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-3', repo_url: 'https://github.com/microsoft/vscode', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-4', repo_url: 'https://github.com/tailwindlabs/tailwindcss', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-5', repo_url: 'https://github.com/openai/whisper', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-6', repo_url: 'https://github.com/nodejs/node', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-7', repo_url: 'https://github.com/rust-lang/rust', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-8', repo_url: 'https://github.com/golang/go', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-9', repo_url: 'https://github.com/python/cpython', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-10', repo_url: 'https://github.com/denoland/deno', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-11', repo_url: 'https://github.com/sveltejs/svelte', status: 'completed', mode: 'standard', created_at: '' },
  { id: 'sample-12', repo_url: 'https://github.com/vuejs/vue', status: 'completed', mode: 'standard', created_at: '' },
];

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
 * Generate a position on the RIGHT side only.
 */
function generateRightPosition(index: number): BubblePosition {
  // Stagger vertically based on index
  const baseY = 20 + (index * 18); // 20%, 38%, 56%, 74%
  const y = baseY + (Math.random() * 10 - 5); // Add some randomness
  
  return {
    x: 75 + Math.random() * 15, // 75-90% from left (right side)
    y: Math.min(Math.max(y, 15), 80), // Clamp between 15-80%
    delay: 0,
    duration: 3 + Math.random() * 2,
  };
}

const MAX_VISIBLE_BUBBLES = 4;
const MIN_CYCLE_INTERVAL = 2000; // Min 2 seconds between changes
const MAX_CYCLE_INTERVAL = 5000; // Max 5 seconds between changes
const FADE_DURATION = 600;

export function RecentReposBubbles() {
  const [allRepos, setAllRepos] = useState<RecentJob[]>([]);
  const [displayBubbles, setDisplayBubbles] = useState<DisplayBubble[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch recent repos and merge with samples
  useEffect(() => {
    async function fetchRecentRepos() {
      try {
        const response = await getRecentJobs(20);
        // Filter to unique repos
        const uniqueRepos = response.jobs.reduce((acc: RecentJob[], job) => {
          const parsed = parseGitHubUrl(job.repo_url);
          if (parsed && !acc.some(j => parseGitHubUrl(j.repo_url)?.repo === parsed.repo)) {
            acc.push(job);
          }
          return acc;
        }, []);

        // Merge with sample repos, avoiding duplicates
        const realUrls = new Set(uniqueRepos.map(r => r.repo_url));
        const samplesToAdd = SAMPLE_REPOS.filter(s => !realUrls.has(s.repo_url));
        
        setAllRepos([...uniqueRepos, ...samplesToAdd]);
      } catch (error) {
        console.error('Failed to fetch recent repos:', error);
        // On error, use sample repos
        setAllRepos(SAMPLE_REPOS);
      } finally {
        setIsLoading(false);
      }
    }

    fetchRecentRepos();
  }, []);

  // Initialize display bubbles when repos are loaded
  useEffect(() => {
    if (allRepos.length === 0) return;

    const shuffled = [...allRepos].sort(() => Math.random() - 0.5);
    const initialRepos = shuffled.slice(0, MAX_VISIBLE_BUBBLES);

    const initialBubbles: DisplayBubble[] = initialRepos.map((job, i) => ({
      job,
      position: generateRightPosition(i),
      opacity: 1,
      state: 'visible' as const,
      key: `${job.id}-${Date.now()}-${i}`,
    }));

    setDisplayBubbles(initialBubbles);
  }, [allRepos]);

  // Cycle bubbles with random intervals
  useEffect(() => {
    if (allRepos.length === 0 || displayBubbles.length === 0) return;

    const scheduleCycle = () => {
      const randomInterval = MIN_CYCLE_INTERVAL + Math.random() * (MAX_CYCLE_INTERVAL - MIN_CYCLE_INTERVAL);
      
      timeoutRef.current = setTimeout(() => {
        setDisplayBubbles((current) => {
          const visibleBubbles = current.filter(b => b.state !== 'exiting');
          if (visibleBubbles.length === 0) return current;

          // Pick a random bubble to exit
          const exitIndex = Math.floor(Math.random() * visibleBubbles.length);
          const bubbleToExit = visibleBubbles[exitIndex];

          // Find a repo that's not currently displayed
          const displayedRepoIds = new Set(current.map(b => b.job.id));
          let availableRepos = allRepos.filter(r => !displayedRepoIds.has(r.id));

          if (availableRepos.length === 0) {
            const shuffled = [...allRepos].sort(() => Math.random() - 0.5);
            const nonDisplayed = shuffled.find(r => r.id !== bubbleToExit.job.id);
            if (nonDisplayed) availableRepos = [nonDisplayed];
          }

          if (availableRepos.length === 0) return current;

          const newRepo = availableRepos[Math.floor(Math.random() * availableRepos.length)];
          const exitedIndex = visibleBubbles.findIndex(b => b.key === bubbleToExit.key);

          const newBubble: DisplayBubble = {
            job: newRepo,
            position: generateRightPosition(exitedIndex >= 0 ? exitedIndex : 0),
            opacity: 0,
            state: 'entering',
            key: `${newRepo.id}-${Date.now()}`,
          };

          return current.map(b =>
            b.key === bubbleToExit.key
              ? { ...b, state: 'exiting' as const, opacity: 0 }
              : b
          ).concat(newBubble);
        });

        scheduleCycle(); // Schedule next cycle
      }, randomInterval);
    };

    scheduleCycle();

    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [allRepos, displayBubbles.length]);

  // Transition entering bubbles to visible, remove exited bubbles
  useEffect(() => {
    // Transition entering bubbles to visible after a brief delay
    const enteringBubbles = displayBubbles.filter(b => b.state === 'entering');
    if (enteringBubbles.length > 0) {
      const timeout = setTimeout(() => {
        setDisplayBubbles(current =>
          current.map(b =>
            b.state === 'entering'
              ? { ...b, state: 'visible' as const, opacity: 1 }
              : b
          )
        );
      }, 50); // Small delay to trigger CSS transition
      return () => clearTimeout(timeout);
    }
  }, [displayBubbles]);

  // Remove exited bubbles after animation completes
  useEffect(() => {
    const exitingBubbles = displayBubbles.filter(b => b.state === 'exiting');
    if (exitingBubbles.length > 0) {
      const timeout = setTimeout(() => {
        setDisplayBubbles(current =>
          current.filter(b => b.state !== 'exiting')
        );
      }, FADE_DURATION);
      return () => clearTimeout(timeout);
    }
  }, [displayBubbles]);

  if (isLoading) {
    return null;
  }

  return (
    <>
      {/* Right side container for bubbles */}
      <div className="absolute right-0 top-0 bottom-0 w-[35%] overflow-hidden pointer-events-none">
        {/* Watermark text */}
        <div className="absolute top-8 right-8 text-right pointer-events-none select-none">
          <p className="text-zinc-700/40 text-2xl font-bold tracking-wider uppercase">
            Recent
          </p>
          <p className="text-zinc-700/40 text-2xl font-bold tracking-wider uppercase -mt-1">
            Synthesis
          </p>
        </div>

        {/* Bubbles */}
        {displayBubbles.map((bubble) => {
          const parsed = parseGitHubUrl(bubble.job.repo_url);
          if (!parsed) return null;

          const repoPath = `/${parsed.owner}/${parsed.repo}`;

          return (
            <Link
              key={bubble.key}
              href={repoPath}
              className="pointer-events-auto absolute group"
              style={{
                right: `${100 - bubble.position.x - 10}%`,
                top: `${bubble.position.y}%`,
                animation: bubble.state === 'visible' 
                  ? `float ${bubble.position.duration}s ease-in-out ${bubble.position.delay}s infinite`
                  : undefined,
                opacity: bubble.opacity,
                transition: `opacity ${FADE_DURATION}ms ease-in-out, transform 300ms ease-out`,
                transform: bubble.state === 'entering' ? 'scale(0.8) translateX(20px)' : 'scale(1) translateX(0)',
              }}
            >
              <div className="relative px-3 py-2 bg-zinc-800/60 backdrop-blur-sm border border-zinc-700/50 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/70 hover:border-teal-600/50 transition-all duration-200 shadow-lg hover:shadow-xl hover:shadow-teal-500/10 hover:scale-105 cursor-pointer whitespace-nowrap">
                <span className="text-zinc-500">github.com/</span>
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
              transform: translateY(-8px);
            }
          }
        `}</style>
      </div>
    </>
  );
}
