param(
  [string]$BaseUrl = "http://127.0.0.1:7860"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Testing /reset" -ForegroundColor Cyan
$resetResp = Invoke-RestMethod -Method Post -Uri "$BaseUrl/reset" -ContentType "application/json" -Body "{}"
if (-not $resetResp.observation) { throw "reset response missing observation" }

Write-Host "[2/4] Testing /step" -ForegroundColor Cyan
$stepBody = @{
  operation = "noop"
  payload = @{}
} | ConvertTo-Json -Depth 5
$stepResp = Invoke-RestMethod -Method Post -Uri "$BaseUrl/step" -ContentType "application/json" -Body $stepBody
if ($null -eq $stepResp.reward) { throw "step response missing reward" }

Write-Host "[3/4] Testing /state" -ForegroundColor Cyan
$stateResp = Invoke-RestMethod -Method Get -Uri "$BaseUrl/state"
if (-not $stateResp.task_id) { throw "state response missing task_id" }

Write-Host "[4/4] Endpoint smoke test passed" -ForegroundColor Green
