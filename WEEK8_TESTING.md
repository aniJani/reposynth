# Week 8 Testing Guide

## Prerequisites

Before testing, ensure you have:
- Docker Desktop installed and running
- PowerShell (Windows) or bash (Linux/Mac)
- curl (optional, for command-line testing)

## Starting the System

### 1. Build and Start All Services

```powershell
# Navigate to project root
cd E:\SummerProjects\reposynth

# Build Docker images (first time or after code changes)
docker-compose build

# Start all services
docker-compose up
```

**Expected output:**
```
✓ Starting reposynth-postgres...
✓ Starting reposynth-redis...
✓ Starting reposynth-minio...
✓ Starting reposynth-api...
✓ Starting reposynth-worker...
```

### 2. Verify Services Are Running

```powershell
# Check service status
docker-compose ps

# All services should show "Up" status
```

Open these URLs in your browser to verify:
- **API Documentation**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health
- **MinIO Console**: http://localhost:9001 (login: minioadmin / minioadmin123)

## Testing Job Submission

### Test 1: Submit a Job (PowerShell)

```powershell
# Submit a job using Invoke-RestMethod
$headers = @{"Content-Type"="application/json"}
$response = Invoke-RestMethod -Uri "http://localhost:8000/jobs?repo_url=https://github.com/jquense/yup&mode=semantic" -Method POST -Headers $headers

# Save the job ID
$jobId = $response.job_id
Write-Host "Job ID: $jobId"
Write-Host "Status: $($response.status)"
```

**Expected output:**
```
Job ID: abc-123-def-456
Status: pending
```

### Test 2: Check Job Status

```powershell
# Check the job status
Invoke-RestMethod -Uri "http://localhost:8000/jobs/$jobId"
```

**Status progression:**
1. `pending` - Job is queued
2. `processing` - Worker is analyzing the repository
3. `completed` - Job finished successfully (or `failed` if error)

### Test 3: Monitor Job Progress

```powershell
# Poll for status updates every 10 seconds
while ($true) {
    $job = Invoke-RestMethod -Uri "http://localhost:8000/jobs/$jobId"
    Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] Status: $($job.status)"
    
    if ($job.status -in @("completed", "failed")) {
        Write-Host "`nFinal Result:"
        $job | ConvertTo-Json -Depth 10
        break
    }
    
    Start-Sleep -Seconds 10
}
```

### Test 4: Download the Result

Once the job is completed:

```powershell
# Get the result URL
$job = Invoke-RestMethod -Uri "http://localhost:8000/jobs/$jobId"
$resultUrl = $job.result_url

Write-Host "Download URL: $resultUrl"

# Download the pack (replace with actual URL)
Invoke-WebRequest -Uri $resultUrl -OutFile "result.zip"

# Extract and view
Expand-Archive -Path "result.zip" -DestinationPath "result" -Force
Get-Content "result\pack\repoBrief.md"
```

## Using the Swagger UI (Interactive API)

1. Open http://localhost:8000/docs
2. Click on `POST /jobs` endpoint
3. Click "Try it out"
4. Enter parameters:
   - `repo_url`: https://github.com/jquense/yup
   - `mode`: semantic
5. Click "Execute"
6. Copy the `job_id` from the response
7. Click on `GET /jobs/{job_id}` endpoint
8. Paste the job ID and execute to check status

## Testing Different Modes

### Semantic Mode (Fast - ~30 seconds)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/jobs?repo_url=https://github.com/jquense/yup&mode=semantic" -Method POST
```

### Hybrid Mode (Medium - ~60 seconds)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/jobs?repo_url=https://github.com/jquense/yup&mode=hybrid" -Method POST
```

### Full Mode (Slow - ~90 seconds)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/jobs?repo_url=https://github.com/jquense/yup&mode=full" -Method POST
```

## Monitoring and Debugging

### View Logs

```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f worker
docker-compose logs -f postgres
docker-compose logs -f redis
docker-compose logs -f minio
```

### Check Database

```powershell
# Connect to PostgreSQL (note: using port 5433 to avoid conflict with local PostgreSQL)
docker-compose exec postgres psql -U repouser -d reposynth

# List all jobs
SELECT id, status, repo_url, created_at FROM jobs ORDER BY created_at DESC;

# Exit psql
\q
```

### Check Redis Queue

```powershell
# Connect to Redis
docker-compose exec redis redis-cli

# Check queue length
LLEN dramatiq:default.DQ

# Exit redis-cli
exit
```

### Check MinIO (S3 Storage)

1. Open http://localhost:9001
2. Login: minioadmin / minioadmin123
3. Navigate to "Buckets" → "reposynth-packs"
4. Browse uploaded pack files
5. Download and inspect packs

## Common Issues

### Issue: Port already in use

**Solution:** Edit `docker-compose.yml` to use different ports:
```yaml
ports:
  - "8001:8000"  # Change 8000 to 8001
```

### Issue: Worker not processing jobs

**Check:**
```powershell
# View worker logs
docker-compose logs worker

# Restart worker
docker-compose restart worker
```

### Issue: Database connection failed

**Solution:**
```powershell
# Restart database
docker-compose restart postgres

# Wait 10 seconds for initialization
Start-Sleep -Seconds 10

# Restart API and worker
docker-compose restart api worker
```

### Issue: MinIO bucket not created

**Solution:**
```powershell
# Recreate bucket manually
docker-compose exec minio mc alias set myminio http://localhost:9000 minioadmin minioadmin123
docker-compose exec minio mc mb myminio/reposynth-packs --ignore-existing
```

## Cleanup

### Stop All Services
```powershell
docker-compose down
```

### Remove All Data (Fresh Start)
```powershell
# Stop and remove volumes
docker-compose down -v

# Remove temp files
Remove-Item -Recurse -Force temp_repos, worker_packs, .reposynth_cache -ErrorAction SilentlyContinue
```

### Remove Docker Images
```powershell
# Remove images to rebuild from scratch
docker-compose down --rmi all -v
```

## Performance Testing

### Test Multiple Jobs

```powershell
# Submit 5 jobs
$jobs = @()
1..5 | ForEach-Object {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/jobs?repo_url=https://github.com/jquense/yup&mode=semantic" -Method POST
    $jobs += $response.job_id
    Write-Host "Submitted job $($_): $($response.job_id)"
}

# Check all job statuses
Start-Sleep -Seconds 30
$jobs | ForEach-Object {
    $job = Invoke-RestMethod -Uri "http://localhost:8000/jobs/$_"
    Write-Host "$($_): $($job.status)"
}
```

## Definition of Done ✅

Your Week 8 implementation is complete when:

1. ✅ `docker-compose up` starts all 6 services without errors
2. ✅ http://localhost:8000/docs shows the API documentation
3. ✅ You can submit a job via POST /jobs
4. ✅ The job status progresses: pending → processing → completed
5. ✅ The result pack appears in MinIO console
6. ✅ You can download and extract the pack
7. ✅ The pack contains all expected files (repoBrief.md, manifest.json, etc.)
8. ✅ Multiple jobs can be processed concurrently

## Next Steps

Once testing is complete:
- Review the generated packs in MinIO
- Check the database for job history
- Experiment with different repositories
- Try all three modes (semantic, hybrid, full)
- Review the worker logs to understand the pipeline stages

Congratulations! You've successfully built a distributed, cloud-ready analysis system! 🎉
