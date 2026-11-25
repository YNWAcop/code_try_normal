param([ValidateSet('Now','AfterCurrent')][string]$Mode='AfterCurrent')
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$marker=Join-Path $root 'user_shutdown_pause.json'
$hold=Join-Path $root 'diagnostic_hold.flag'
function Read-State { Get-Content -LiteralPath (Join-Path $root 'queue_status.json') -Raw | ConvertFrom-Json }
function Process-ById($processId) { Get-CimInstance Win32_Process -Filter "ProcessId=$processId" }
if(Test-Path -LiteralPath $marker){throw 'Pause record already exists. Read it before repeating this command.'}
if(Test-Path -LiteralPath (Join-Path $root 'queue_error.json')){throw 'Queue already has an error. Inspect queue_error.json before pausing.'}
if(Test-Path -LiteralPath $hold){throw 'An existing diagnostic hold belongs to another task. Do not overwrite it.'}
$s=Read-State
if($s.stage -eq 'complete'){Write-Host 'Queue is already complete. No pause is needed.';exit 0}
$q=Process-ById $s.queue_pid
if(!$q -or $q.CommandLine -notmatch 'run_emt_queue\.py'){throw 'Cannot verify the live queue process.'}
$record=[ordered]@{mode=$Mode;job=$s.job;run=$s.run;epoch=$s.epoch;requested=(Get-Date).ToString('o');state='requested';scope='EMT/MISA repair queue only';resume_semantics='Completed jobs skipped; interrupted unlocked job restarted from scratch, not exact epoch resume'}
$record | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
Set-Content -LiteralPath $hold -Value 'User shutdown pause. Managed by Stop-Training.ps1 and Resume-Training.ps1.' -Encoding UTF8
if($Mode -eq 'AfterCurrent'){
 Write-Host 'Waiting for the current job AND its validation/test checks to finish. Do not shut down yet.'
 $last=''
 while($true){
  $s=Read-State
  $message="Stage=$($s.stage) Job=$($s.job) Completed=$($s.completed)/$($s.expected) Epoch=$($s.epoch)"
  if($message -ne $last){Write-Host $message;$last=$message}
  if(Test-Path -LiteralPath (Join-Path $root 'queue_error.json')){throw 'Queue failed while waiting. No safe-stop confirmation.'}
  if($s.stage -eq 'waiting_for_diagnostics' -or $s.stage -eq 'complete'){break}
  if(!(Process-ById $q.ProcessId)){throw 'Queue exited unexpectedly; inspect logs.'}
  Start-Sleep -Seconds 10
 }
}
$s=Read-State
if($s.job){$record.job=$s.job;$record.run=$s.run;$record.epoch=$s.epoch}
if($s.stage -eq 'complete'){
 $record.state='queue_complete';$record.stopped=(Get-Date).ToString('o')
 $record | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
 Remove-Item -LiteralPath $hold
 Write-Host 'Queue complete. Let finalize_tables.py finish before shutting down.'
 exit 0
}
# Stop the watcher only while it is waiting, never during table publication.
if(Test-Path -LiteralPath (Join-Path $root 'finalizer_status.json')){
 $f=Get-Content -LiteralPath (Join-Path $root 'finalizer_status.json') -Raw | ConvertFrom-Json
 if($f.stage -ne 'waiting_for_repairs'){throw 'Finalizer is not confirmed to be waiting. Inspect before stopping.'}
}
$watchers=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'baseline_repair_20261123[/\\]finalize_tables\.py' }
foreach($w in $watchers){Stop-Process -Id $w.ProcessId -ErrorAction Stop}
$q2=Process-ById $q.ProcessId
if($q2 -and $q2.CommandLine -match 'run_emt_queue\.py'){
 # Kill only the verified queue and its descendants, never all Python processes.
 & taskkill.exe /PID $q2.ProcessId /T /F
 if($LASTEXITCODE -ne 0){throw 'Queue process tree did not stop cleanly.'}
}
Start-Sleep -Seconds 2
if(Process-ById $q.ProcessId){throw 'Queue still exists. Do not shut down yet.'}
$record.state='stopped';$record.stopped=(Get-Date).ToString('o');$record.last_status=$s
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $marker -Encoding UTF8
Write-Host 'BASELINE QUEUE STOPPED. Finished jobs and original files are preserved.'
if($Mode -eq 'Now'){Write-Host 'On restart, the current unlocked job will be archived and trained again from epoch 1.'}
Write-Host 'Other Python jobs are outside this script. Check them before shutdown:'
Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId,CommandLine | Format-List

