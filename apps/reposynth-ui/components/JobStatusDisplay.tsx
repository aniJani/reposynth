// components/JobStatusDisplay.tsx
'use client';

import { useStore } from '@/lib/store';
import { useEffect } from 'react';
import { getJobStatus } from '@/lib/api';
import {
  CheckCircle,
  Clock,
  Loader2,
  XCircle,
  Download,
  ExternalLink,
} from 'lucide-react';

export function JobStatusDisplay() {
  const { currentJob, setCurrentJob } = useStore();

  // Poll for job status updates
  useEffect(() => {
    if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await getJobStatus(currentJob.id);
        setCurrentJob(updated);
      } catch (error) {
        console.error('Failed to fetch job status:', error);
      }
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, [currentJob, setCurrentJob]);

  if (!currentJob) {
    return null;
  }

  const getStatusIcon = () => {
    switch (currentJob.status) {
      case 'pending':
        return <Clock className="h-6 w-6 text-yellow-500" />;
      case 'processing':
        return <Loader2 className="h-6 w-6 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-6 w-6 text-green-500" />;
      case 'failed':
        return <XCircle className="h-6 w-6 text-red-500" />;
    }
  };

  const getStatusColor = () => {
    switch (currentJob.status) {
      case 'pending':
        return 'border-yellow-200 bg-yellow-50';
      case 'processing':
        return 'border-blue-200 bg-blue-50';
      case 'completed':
        return 'border-green-200 bg-green-50';
      case 'failed':
        return 'border-red-200 bg-red-50';
    }
  };

  return (
    <div className={`w-full max-w-4xl mx-auto border rounded-lg p-6 ${getStatusColor()}`}>
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">{getStatusIcon()}</div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2">
            <h3 className="text-lg font-semibold text-gray-900 capitalize">
              {currentJob.status}
            </h3>
            <span className="text-sm text-gray-500">
              Job ID: {currentJob.id.slice(0, 8)}
            </span>
          </div>

          <p className="text-sm text-gray-600 mb-3">
            {currentJob.repo_url}
          </p>

          {currentJob.status === 'processing' && (
            <div className="flex items-center gap-2 text-sm text-blue-700">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Analyzing repository...</span>
            </div>
          )}

          {currentJob.status === 'completed' && currentJob.result_url && (
            <div className="flex gap-3">
              {/* Determine output format from URL extension */}
              {currentJob.result_url.endsWith('.md') || currentJob.result_url.endsWith('.json') ? (
                <>
                  <a
                    href={currentJob.result_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    <ExternalLink className="h-4 w-4" />
                    View in Browser
                  </a>
                  <button
                    onClick={() => {
                      // Force download by creating a temporary link
                      fetch(currentJob.result_url!)
                        .then(res => res.blob())
                        .then(blob => {
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = currentJob.result_url!.split('/').pop() || 'download';
                          document.body.appendChild(a);
                          a.click();
                          window.URL.revokeObjectURL(url);
                          document.body.removeChild(a);
                        });
                    }}
                    className="inline-flex items-center gap-2 px-4 py-2 border border-green-600 text-green-700 rounded-lg hover:bg-green-50 transition-colors"
                  >
                    <Download className="h-4 w-4" />
                    Download File
                  </button>
                </>
              ) : (
                <>
                  <a
                    href={currentJob.result_url}
                    download
                    className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    <Download className="h-4 w-4" />
                    Download Pack
                  </a>
                  <a
                    href={currentJob.result_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-4 py-2 border border-green-600 text-green-700 rounded-lg hover:bg-green-50 transition-colors"
                  >
                    <ExternalLink className="h-4 w-4" />
                    View in Browser
                  </a>
                </>
              )}
            </div>
          )}

          {currentJob.status === 'completed' && currentJob.processing_time_seconds && (
            <p className="text-sm text-gray-600 mt-2">
              Completed in {Math.round(currentJob.processing_time_seconds)}s
            </p>
          )}

          {currentJob.status === 'failed' && currentJob.error_message && (
            <div className="mt-2 p-3 bg-red-100 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700 font-medium">Error:</p>
              <p className="text-sm text-red-600 mt-1">
                {currentJob.error_message}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
