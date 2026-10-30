# Show progress of the preprocessing.
param(
    [string]$DataRoot = "G:\My Drive\CS406-SCB-TumorSeg\data_real",
    [string]$LogDir = "D:\venv_scb"
)
$npz = (Get-ChildItem "$DataRoot\processed\npz_96x96x96" -Filter *.npz -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*.tmp.npz' } | Measure-Object).Count
$alive = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'preprocess_via_local_copy' } | Measure-Object).Count -gt 0
Write-Host ("Done: {0}/1480 ({1:P0}) | running: {2}" -f $npz, ($npz / 1480), $alive)
$last = Get-ChildItem "$DataRoot\processed\npz_96x96x96" -Filter *.npz -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($last) { Write-Host "Latest file: $($last.Name) at $($last.LastWriteTime)" }
if (Test-Path "$LogDir\full.log") { Write-Host "Log:"; Get-Content "$LogDir\full.log" -Tail 2 }
if ((Test-Path "$LogDir\full.err") -and (Get-Item "$LogDir\full.err").Length -gt 0) { Write-Host "Errors (tail):"; Get-Content "$LogDir\full.err" -Tail 3 }
