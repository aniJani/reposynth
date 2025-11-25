# Week 9 Frontend - Quick Test Guide

## ✅ System Status Check

All services should be running:

```powershell
# Check Docker services
docker-compose ps

# Expected output:
# reposynth-api         Running
# reposynth-worker      Running
# reposynth-postgres    Running
# reposynth-redis       Running  
# reposynth-minio       Running
```

```powershell
# Check API health
Invoke-RestMethod http://localhost:8000/health

# Check frontend
# Open http://localhost:3000 in browser
```

## 🎯 Testing the Frontend

### 1. Open Frontend
Navigate to: **http://localhost:3000**

### 2. Enter Repository URL
Try these test repositories:
- `https://github.com/jquense/yup` (Small, fast - ~30 seconds)
- `https://github.com/pallets/click` (Medium)
- `https://github.com/psf/requests` (Larger)

### 3. Watch Real-Time Estimation
As you type and change settings, the token estimate updates automatically:
- **Semantic Mode**: ~15K-50K tokens ($0.03-$0.10)
- **Hybrid Mode**: ~50K-150K tokens ($0.10-$0.30)
- **Full Mode**: ~150K+ tokens ($0.30+)

### 4. Configure Features
Toggle these on/off and watch estimates update:
- ✅ **AST Parsing**: Syntax tree analysis
- ✅ **Import Graph**: Dependency mapping
- ✅ **Complexity**: Code metrics
- ❌ **Security Scans**: Vulnerability detection (adds ~30% time)
- ✅ **Embeddings**: AI vector representations (adds ~40% time)

### 5. Submit Job
Click **"Generate Pack"**
- Button shows loading spinner
- Job status appears below
- Status updates every 3 seconds

### 6. Monitor Progress
Watch the status indicator:
- 🟡 **Pending**: Job queued
- 🔵 **Processing**: Worker is analyzing (~30-90 seconds)
- 🟢 **Completed**: Pack ready for download
- 🔴 **Failed**: Check error message

### 7. Download Result
When completed:
- Click **"Download Pack"** button
- ZIP file downloads automatically
- Extract and explore the pack contents

## 🧪 Quick API Test

Test job submission via API:

```powershell
# Submit a job
$body = @{
    mode = "semantic"
    enable_ast = $true
    enable_imports = $true
    enable_complexity = $true
    enable_security = $false
    enable_embeddings = $true
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri "http://localhost:8000/jobs?repo_url=https://github.com/jquense/yup" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

# Save job ID
$jobId = $response.job_id
Write-Host "Job ID: $jobId"

# Check status
Invoke-RestMethod -Uri "http://localhost:8000/jobs/$jobId"

# Poll until complete
while ($true) {
    $job = Invoke-RestMethod -Uri "http://localhost:8000/jobs/$jobId"
    Write-Host "Status: $($job.status)"
    
    if ($job.status -in @("completed", "failed")) {
        $job | ConvertTo-Json -Depth 10
        break
    }
    
    Start-Sleep -Seconds 5
}

# Download result
if ($job.result_url) {
    Invoke-WebRequest -Uri $job.result_url -OutFile "pack.zip"
    Write-Host "✓ Pack downloaded to pack.zip"
}
```

## 🔍 Expected Results

### Semantic Mode (yup repository)
- **Processing Time**: ~30-40 seconds
- **Pack Size**: ~230KB
- **Contents**:
  - `repoBrief.md` (AI-generated summary)
  - `manifest.json` (file metadata)
  - `import_graph.json` (dependencies)
  - `name_registry.json` (symbols)
  - `vectors.faiss` (embeddings)
  - `ast_raw/` (syntax trees)

### Hybrid Mode
- **Processing Time**: ~60-90 seconds
- **Pack Size**: ~500KB-2MB
- **Additional Contents**:
  - `security_report.json` (vulnerabilities)
  - `variable_registry.json` (detailed symbols)

### Full Mode
- **Processing Time**: ~90-120 seconds
- **Pack Size**: ~2MB-10MB
- **Additional Contents**:
  - Complete span storage
  - Full variable tracking
  - Comprehensive embeddings

## 🐛 Troubleshooting

### Frontend Won't Load
```powershell
# Check if frontend dev server is running
# Should see: "✓ Ready in 32.5s"

# If not running:
cd E:\SummerProjects\reposynth\apps\reposynth-ui
npm run dev
```

### API Errors
```powershell
# Check API logs
docker-compose logs api --tail 50

# Restart API
docker-compose restart api

# Check database migration ran
docker-compose logs api | Select-String "migration"
# Should see: "✓ Successfully added 'config' column"
```

### Job Stays "Pending"
```powershell
# Check worker logs
docker-compose logs worker --tail 50

# Restart worker
docker-compose restart worker

# Check Redis connection
docker-compose logs redis
```

### Estimation Not Working
- **Check console**: Open browser DevTools (F12) → Console tab
- **Look for errors**: Network tab should show `/estimate` requests
- **Verify API URL**: Check `apps/reposynth-ui/.env.local` has:
  ```
  NEXT_PUBLIC_API_URL=http://localhost:8000
  ```

### Download Button Not Appearing
- **Check job status**: Should be "completed"
- **Verify result_url**: Check API response has valid MinIO URL
- **Test URL**: Copy result_url and paste in browser

## 📊 Monitoring Tools

### MinIO Console (S3 Storage)
- **URL**: http://localhost:9001
- **Login**: minioadmin / minioadmin123
- **View**: Browse uploaded packs in `reposynth-packs` bucket

### API Documentation
- **URL**: http://localhost:8000/docs
- **Interactive**: Try endpoints with Swagger UI
- **Schemas**: View request/response models

### Database
```powershell
# Connect to PostgreSQL
docker-compose exec postgres psql -U repouser -d reposynth

# List jobs
SELECT id, status, mode, created_at FROM jobs ORDER BY created_at DESC;

# View specific job
SELECT * FROM jobs WHERE id = 'YOUR_JOB_ID';

# Exit
\q
```

## ✨ Success Criteria

Your Week 9 implementation is working correctly when:

1. ✅ Frontend loads at http://localhost:3000
2. ✅ URL input accepts GitHub repository URLs
3. ✅ Token estimate appears within 1 second
4. ✅ Changing mode/features updates estimate
5. ✅ Submit button creates job successfully
6. ✅ Job status updates automatically every 3 seconds
7. ✅ Processing completes in ~30-90 seconds
8. ✅ Download button appears when completed
9. ✅ ZIP file downloads and contains pack files
10. ✅ Multiple jobs can be submitted concurrently

**Congratulations!** You've built a complete full-stack repository analysis system! 🎉
