#!/bin/bash
# API startup script with automatic database migration

echo "🚀 Starting RepoSynth API..."

# Run database migration
echo "📊 Running database migrations..."
cd /app/packages/python-orchestrator
python -m orchestrator.migrations.add_config_column

if [ $? -ne 0 ]; then
    echo "✗ Migration failed! API will not start."
    exit 1
fi

echo "✓ Migrations completed successfully"

# Start the API server
echo "🌐 Starting API server on port 8000..."
cd /app/packages/python-orchestrator
uvicorn orchestrator.api:app --host 0.0.0.0 --port 8000
