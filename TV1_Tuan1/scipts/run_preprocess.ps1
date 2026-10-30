# Start (or resume) the Drive -> local preprocessing in the background. Safe to run any time: finished cases are skipped.
param(
    [string]$DataRoot = "G:\My Drive\CS406-SCB-TumorSeg\data_real",
    [string]$Scratch = "D:\scb_raw",
    [string]$Python = "D:\venv_scb\Scripts\python.exe",
    [int]$CopyWorkers = 16,
    [int]$Workers = 10
)
$project = Split-Path -Parent $PSScriptRoot
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'preprocess_via_local_copy' }
if ($running) { Write-Host "Already running (PID $($running[0].ProcessId)). Use status_preprocess.ps1 to watch it."; exit 0 }
if (-not (Test-Path $DataRoot)) { Write-Host "Cannot see $DataRoot - is Google Drive for desktop running?"; exit 1 }
New-Item -ItemType Directory -Force $Scratch | Out-Null
Remove-Item "$Scratch\STOP" -ErrorAction SilentlyContinue
$log = Join-Path (Split-Path $Python -Parent | Split-Path -Parent) "full.log"
$err = Join-Path (Split-Path $Python -Parent | Split-Path -Parent) "full.err"
Start-Process -FilePath $Python -WorkingDirectory $project -WindowStyle Hidden `
    -RedirectStandardOutput $log -RedirectStandardError $err `
    -ArgumentList @("scripts\preprocess_via_local_copy.py", "--data-root", "`"$DataRoot`"", "--scratch", $Scratch,
                    "--copy-workers", $CopyWorkers, "--workers", $Workers, "--max-inflight", 40)
Write-Host "Started. Log: $log"
