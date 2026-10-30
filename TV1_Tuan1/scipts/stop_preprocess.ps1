# Stop the preprocessing without losing finished work. Default: graceful (finishes cases already copied, then exits).
# Use -Force to kill immediately; only the few cases in flight are redone on the next run.
param(
    [string]$Scratch = "D:\scb_raw",
    [int]$WaitMinutes = 15,
    [switch]$Force
)
function Get-Jobs { Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'preprocess_via_local_copy' } }
if (-not (Get-Jobs)) { Write-Host "Not running."; exit 0 }
if (-not $Force) {
    New-Item -ItemType Directory -Force $Scratch | Out-Null
    New-Item -ItemType File -Force "$Scratch\STOP" | Out-Null
    Write-Host "STOP requested. Waiting up to $WaitMinutes min for in-flight cases to finish..."
    $deadline = (Get-Date).AddMinutes($WaitMinutes)
    while ((Get-Jobs) -and (Get-Date) -lt $deadline) { Start-Sleep 5 }
}
if (Get-Jobs) {
    Write-Host "Killing remaining processes."
    Get-Jobs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Write-Host "Stopped. Finished .npz files are kept; run_preprocess.ps1 resumes (it cleans half-done files itself)."
