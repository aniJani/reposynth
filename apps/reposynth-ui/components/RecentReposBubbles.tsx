'use client';

import { useEffect, useState, useMemo, useCallback } from 'react';
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
  key: string; // Unique key for animation tracking
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
 * Generate a single random position for a bubble.
 */
function generateSinglePosition(): BubblePosition {
  // Distribute across left or right side randomly
  const side = Math.random() > 0.5 ? 'left' : 'right';
  const x = side === 'left'
    ? 5 + Math.random() * 25 // 5-30% from left
    : 65 + Math.random() * 25; // 65-90% from left

  const y = 15 + (Math.random() * 70); // 15-85% from top

  return {
    x,
    y,
    delay: 0, // No delay for dynamically added bubbles
    duration: 3 + Math.random() * 2, // Animation duration 3-5s
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

const MAX_VISIBLE_BUBBLES = 6;
const CYCLE_INTERVAL = 4000; // Swap a bubble every 4 seconds
const FADE_DURATION = 800; // Fade animation duration in ms

export function RecentReposBubbles() {
  const [allRepos, setAllRepos] = useState<RecentJob[]>([]);
  const [displayBubbles, setDisplayBubbles] = useState<DisplayBubble[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [cycleCounter, setCycleCounter] = useState(0);

  // Fetch recent repos
  useEffect(() => {
    async function fetchRecentRepos() {
      try {
        const response = await getRecentJobs(20); // Get more repos for cycling
        // Filter to unique repos
        const uniqueRepos = response.jobs.reduce((acc: RecentJob[], job) => {
          const parsed = parseGitHubUrl(job.repo_url);
          if (parsed && !acc.some(j => parseGitHubUrl(j.repo_url)?.repo === parsed.repo)) {
            acc.push(job);
          }
          return acc;
        }, []);

        setAllRepos(uniqueRepos);
      } catch (error) {
        console.error('Failed to fetch recent repos:', error);
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
    const positions = generateBubblePositions(initialRepos.length);

    const initialBubbles: DisplayBubble[] = initialRepos.map((job, i) => ({
      job,
      position: positions[i],
      opacity: 1,
      state: 'visible' as const,
      key: `${job.id}-${Date.now()}-${i}`,
    }));

    setDisplayBubbles(initialBubbles);
  }, [allRepos]);

  // Cycle bubbles - fade one out and bring a new one in
  useEffect(() => {
    if (allRepos.length <= MAX_VISIBLE_BUBBLES || displayBubbles.length === 0) return;

    const interval = setInterval(() => {
      setDisplayBubbles((current) => {
        // Find bubbles that are visible (not exiting)
        const visibleBubbles = current.filter(b => b.state !== 'exiting');
        if (visibleBubbles.length === 0) return current;

        // Pick a random bubble to exit
        const exitIndex = Math.floor(Math.random() * visibleBubbles.length);
        const bubbleToExit = visibleBubbles[exitIndex];

        // Find a repo that's not currently displayed
        const displayedRepoIds = new Set(current.map(b => b.job.id));
        const availableRepos = allRepos.filter(r => !displayedRepoIds.has(r.id));

        if (availableRepos.length === 0) {
          // If no new repos available, shuffle from all repos
          const shuffled = [...allRepos].sort(() => Math.random() - 0.5);
          const nonDisplayed = shuffled.find(r => r.id !== bubbleToExit.job.id);
          if (!nonDisplayed) return current;
          availableRepos.push(nonDisplayed);
        }

        const newRepo = availableRepos[Math.floor(Math.random() * availableRepos.length)];

        // Mark the selected bubble as exiting and add a new entering bubble
        const newBubble: DisplayBubble = {
          job: newRepo,
          position: generateSinglePosition(),
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

      setCycleCounter(c => c + 1);
    }, CYCLE_INTERVAL);

    return () => clearInterval(interval);
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

  if (isLoading || displayBubbles.length === 0) {
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
                left: `${bubble.position.x}%`,
                top: `${bubble.position.y}%`,
                animation: bubble.state === 'visible' 
                  ? `float ${bubble.position.duration}s ease-in-out ${bubble.position.delay}s infinite`
                  : undefined,
                opacity: bubble.opacity,
                transition: `opacity ${FADE_DURATION}ms ease-in-out, transform 200ms ease-out`,
                transform: bubble.state === 'entering' ? 'scale(0.8)' : 'scale(1)',
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
