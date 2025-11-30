'use client';

import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';

export default function AboutPage() {
  return (
    <div className="min-h-screen w-full bg-zinc-950 flex flex-col">
      {/* Header */}
      <header className="bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800/50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
              <Image 
                src="/logo.png" 
                alt="RepoSynth Logo" 
                width={32} 
                height={32}
              />
              <div>
                <h1 className="text-lg font-bold font-display tracking-tight text-zinc-200">
                  RepoSynth
                </h1>
                <p className="text-xs text-zinc-500">Repository Synthesis & LLM Context Engine</p>
              </div>
            </Link>
            <Link 
              href="/" 
              className="flex items-center gap-2 text-zinc-400 hover:text-zinc-200 transition-colors text-sm"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="max-w-2xl w-full">
          <h2 className="text-4xl font-bold text-zinc-200 mb-6 font-display tracking-tight">
            About RepoSynth
          </h2>
          
          <div className="space-y-6 text-zinc-400">
            <p className="text-lg leading-relaxed">
              RepoSynth is a powerful repository analysis tool designed to help developers 
              understand and navigate complex codebases. It synthesizes repository data into 
              meaningful insights optimized for LLM context.
            </p>
            
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-zinc-300">Features</h3>
              <ul className="space-y-2 list-disc list-inside">
                <li>Deep repository structure analysis</li>
                <li>Intelligent code summarization</li>
                <li>LLM-optimized context generation</li>
                <li>Support for multiple programming languages</li>
                <li>Interactive vibe station for exploration</li>
              </ul>
            </div>

            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-zinc-300">How It Works</h3>
              <p className="leading-relaxed">
                Simply paste a GitHub repository URL, configure your analysis preferences, 
                and let RepoSynth do the heavy lifting. The tool parses the repository, 
                builds an understanding of its structure, and generates comprehensive 
                summaries that can be used as context for AI-assisted development.
              </p>
            </div>

            <div className="pt-4">
              <Link 
                href="/"
                className="inline-flex items-center gap-2 px-6 py-3 bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors font-semibold"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-6 text-center border-t border-zinc-800/50 bg-zinc-950/50">
        <div className="text-xs text-zinc-600">
          <Link href="/about" className="hover:text-zinc-400 transition-colors">
            About
          </Link>
        </div>
      </footer>
    </div>
  );
}
