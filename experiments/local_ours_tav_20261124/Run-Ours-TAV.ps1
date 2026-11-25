param(
    [ValidateSet('Check','Train','Status','Evaluate','Export')][string]$Action='Check',
    [string]$Python='E:\UserData\Desktop\EMT-DLFR\.conda\python.exe',
    [string]$DataRoot='D:\MSA_Cloud_20261115\data',
    [string]$RunRoot=(Join-Path $PSScriptRoot 'outputs\formal')
)
$ErrorActionPreference='Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$entry=Join-Path $PSScriptRoot 'local_tav.py'
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
$verb=$Action.ToLowerInvariant()
& $Python -u $entry $verb --data-root $DataRoot --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "Ours TAV $Action failed. Read the error and logs; outputs are preserved." }
