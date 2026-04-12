param(
  [string]$BaseUrl = "http://127.0.0.1:7860"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/7] Testing /health" -ForegroundColor Cyan
$healthResp = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health"
if ($healthResp.status -ne "healthy") { throw "health response missing healthy status" }

Write-Host "[2/7] Testing /metadata" -ForegroundColor Cyan
$metadataResp = Invoke-RestMethod -Method Get -Uri "$BaseUrl/metadata"
if (-not $metadataResp.name) { throw "metadata response missing name" }

Write-Host "[3/7] Testing /reset" -ForegroundColor Cyan
$resetResp = Invoke-RestMethod -Method Post -Uri "$BaseUrl/reset" -ContentType "application/json" -Body "{}"
if (-not $resetResp.observation) { throw "reset response missing observation" }

Write-Host "[4/7] Testing /step" -ForegroundColor Cyan
$stepBody = @{
  action = @{
    operation = "noop"
  }
} | ConvertTo-Json -Depth 8
$stepResp = Invoke-RestMethod -Method Post -Uri "$BaseUrl/step" -ContentType "application/json" -Body $stepBody
if ($null -eq $stepResp.reward) { throw "step response missing reward" }

Write-Host "[5/7] Testing /state" -ForegroundColor Cyan
$stateResp = Invoke-RestMethod -Method Get -Uri "$BaseUrl/state"
if (-not $stateResp.task_id) { throw "state response missing task_id" }

Write-Host "[6/7] Testing /schema" -ForegroundColor Cyan
$schemaResp = Invoke-RestMethod -Method Get -Uri "$BaseUrl/schema"
if (-not $schemaResp.action) { throw "schema response missing action schema" }

Write-Host "[7/7] Testing /mcp" -ForegroundColor Cyan
$mcpResp = Invoke-RestMethod -Method Post -Uri "$BaseUrl/mcp" -ContentType "application/json" -Body "{}"
if ($mcpResp.jsonrpc -ne "2.0") { throw "mcp response missing jsonrpc version" }

Write-Host "Endpoint smoke test passed" -ForegroundColor Green
