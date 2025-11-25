# RepoSynth Frontend

Modern Next.js frontend for RepoSynth repository analysis.

## Quick Start

```powershell
# Install dependencies
npm install

# Start development server
npm run dev

# Open browser to http://localhost:3000
```

## Features

- **Real-Time Token Estimation** - Instant feedback as you configure
- **Interactive Configuration** - Toggle modes and features
- **Live Job Monitoring** - Automatic status updates
- **Beautiful UI** - Modern design with Tailwind CSS
- **Type-Safe** - Complete TypeScript implementation

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development

```powershell
# Type checking
npm run type-check

# Linting
npm run lint

# Build for production
npm run build

# Start production server
npm start
```

## Architecture

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Zustand** - State management
- **Axios** - HTTP client
- **Lucide React** - Icon library

## Backend Required

This frontend requires the RepoSynth backend to be running:

```powershell
cd E:\SummerProjects\reposynth
docker-compose up
```

Backend should be available at http://localhost:8000

## Components

- **UrlInput** - GitHub URL input field
- **ConfiguratorPanel** - Mode selector + feature toggles
- **EstimatorDisplay** - Real-time cost/time display
- **SubmitButton** - Job submission with loading state
- **JobStatusDisplay** - Live job status with polling

## State Management

Uses Zustand for global state:
- Repository URL
- Configuration (mode, features)
- Estimates (tokens, time, cost)
- Current job status
- Loading states
- Errors

## API Integration

All API calls go through `lib/api.ts`:
- `estimateTokens()` - POST /estimate
- `createJob()` - POST /jobs
- `getJobStatus()` - GET /jobs/{id}
- `listJobs()` - GET /jobs

## Troubleshooting

### Can't connect to backend
Check `.env.local` has correct API URL and backend is running.

### TypeScript errors
Run `npm install` to ensure all dependencies are installed.

### Styles not working
Restart dev server after changing Tailwind config.

## License

MIT
