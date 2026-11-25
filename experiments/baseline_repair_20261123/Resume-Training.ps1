param([switch]$RecoverAfterReboot,[switch]$CheckOnly)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$workspace=Split-Path $root -Parent
$python='E:\UserData\Desktop\EMT-DLFR\.conda\python.exe'
$runs=[IO.Path]::GetFullPath('D:\baseline_repair_runs_20261123')
$marker=Join-Path $root 'user_shutdown_pause.json'
$hold=Join-Path $root 'diagnostic_hold.flag'
if(!(Test-Path -LiteralPath $python)){throw "Python environment missing: $python"}
$active=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match '(run_emt_queue\.py|baseline_repair_20261123[/\\](adapter_emt_tfr|adapter_misa_almt|evaluate_repair|finalize_tables)\.py)' }
if($active){$active | Select-Object ProcessId,CommandLine | Format-List;throw 'A repair process is already active. Do not launch duplicates.'}
$recovered=$false
if(Test-Path -LiteralPath $marker){
 $pause=Get-Content -Encoding UTF8 -LiteralPath $marker -Raw | ConvertFrom-Json
}elseif($RecoverAfterReboot){
 $old=Get-Content -Encoding UTF8 -LiteralPath (Join-Path $root 'queue_status.json') -Raw | ConvertFrom-Json
 $boot=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime
 $oldTime=[DateTimeOffset]::FromUnixTimeSeconds([long]$old.updated)
 if($oldTime.UtcDateTime -ge $boot.ToUniversalTime()){throw 'Queue status is newer than boot; interruption needs manual inspection.'}
 if($old.stage -notin @('training','valid','test','waiting_for_diagnostics','complete')){throw 'Unrecognized interrupted stage.'}
 foreach($n in @('queue_error.json','finalizer_error.json','diagnostic_hold.flag')){
  if(Test-Path -LiteralPath (Join-Path $root $n)){throw "Existing error/hold requires inspection: $n"}
 }
 $pause=[pscustomobject]@{mode='Now';job=$old.job;run=$old.run;epoch=$old.epoch;state='stopped';requested=(Get-Date).ToString('o');recovered_without_pause_record=$true;last_boot=$boot.ToString('o');previous_queue_status=$old;scope='EMT/MISA repair queue only';resume_semantics='Completed jobs skipped; unfinished job archived and restarted, not exact epoch resume'}
 $recovered=$true
}else{
 throw 'No unused pause record. After a reboot, use -RecoverAfterReboot to verify and recover unfinished jobs.'
}
if($pause.state -notin @('stopped','queue_complete')){throw 'Pause was not confirmed. Inspect processes and pause record first.'}
$plan=Get-Content -Encoding UTF8 -LiteralPath (Join-Path $root 'repair_retrain_plan.json') -Raw | ConvertFrom-Json
# Check frozen training sources before moving any interrupted artifacts.
foreach($source in $plan.sources.PSObject.Properties){
 $sourcePath=$source.Name
 if(![IO.Path]::IsPathRooted($sourcePath)){$sourcePath=Join-Path $workspace $sourcePath}
 if((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $source.Value.ToLowerInvariant()){throw "Frozen training source changed: $sourcePath"}
}
$pending=@();$finished=0
foreach($job in $plan.jobs){
 $tag="$($job.method)_$($job.dataset)_$($job.condition)"
 $run=[IO.Path]::GetFullPath((Join-Path $runs $tag))
 if(!( $run.StartsWith($runs+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase))){throw 'Run path is outside the expected directory.'}
 $report=Join-Path $run 'evaluation\test_report.json'
 if(Test-Path -LiteralPath $report){
  if(!(Get-Content -Encoding UTF8 -LiteralPath $report -Raw | ConvertFrom-Json).passed){throw "Failed test report: $tag"}
  $finished++;continue
 }
 if((Test-Path -LiteralPath $run) -and !(Test-Path -LiteralPath (Join-Path $run 'validation_lock.json'))){
  if($tag -ne $pause.job -or $pause.mode -ne 'Now'){throw "Unexpected unfinished run: $tag. Manual inspection required."}
  $pending+=@{tag=$tag;path=$run}
 }
}
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
Write-Host "Verified completed jobs: $finished/$($plan.jobs.Count); interrupted unlocked jobs: $($pending.Count)."
if($CheckOnly){Write-Host 'Check only: no files moved and no training launched.';exit 0}
if($recovered){$pause | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $marker -Encoding UTF8}
foreach($item in $pending){
 $archive=[IO.Path]::GetFullPath((Join-Path $runs ('paused_archive\'+$item.tag+'_'+$stamp)))
 if(!$archive.StartsWith($runs+'\paused_archive\',[StringComparison]::OrdinalIgnoreCase)){throw 'Invalid archive destination.'}
 New-Item -ItemType Directory -Path (Split-Path $archive -Parent) -Force | Out-Null
 # Keep all weights/history. This is an interruption restart, not score-based selection.
 Move-Item -LiteralPath $item.path -Destination $archive
 foreach($stage in @('training','valid','test')){
  $oldlog=Join-Path $runs ($item.tag+'_'+$stage+'.log')
  if(Test-Path -LiteralPath $oldlog){Move-Item -LiteralPath $oldlog -Destination (Join-Path $archive ('previous_'+$stage+'.log'))}
 }
 Write-Host "Interrupted run preserved at $archive; this job will restart from epoch 1."
}
foreach($errorName in @('queue_error.json','finalizer_error.json')){
 $errorPath=Join-Path $root $errorName
 if(Test-Path -LiteralPath $errorPath){
  $err=Get-Content -Encoding UTF8 -LiteralPath $errorPath -Raw | ConvertFrom-Json
  $requested=[DateTimeOffset]::Parse($pause.requested).ToUnixTimeSeconds()
  if(!$err.time -or $err.time -lt $requested){throw "Unrelated earlier error: $errorName"}
  Move-Item -LiteralPath $errorPath -Destination (Join-Path $root ($errorName+'.paused_'+$stamp))
 }
}
if(Test-Path -LiteralPath $hold){
 if((Get-Content -Encoding UTF8 -LiteralPath $hold -Raw) -notmatch 'User shutdown pause'){throw 'Hold flag belongs to another task.'}
 Remove-Item -LiteralPath $hold
}
if($finished -eq $plan.jobs.Count){
 Write-Host 'All jobs finished. Running the existing finalizer only if not yet published.'
 if(!(Test-Path -LiteralPath (Join-Path $workspace 'outputs\baseline_repaired_tables_20261123\verification.json'))){
  Push-Location $workspace
  try { & $python (Join-Path $root 'finalize_tables.py'); if($LASTEXITCODE -ne 0){throw 'Finalization failed.'} } finally {Pop-Location}
 }
}else{
 $queue=Start-Process -FilePath $python -ArgumentList @('-u','baseline_repair_20261123/run_emt_queue.py') -WorkingDirectory $workspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root "resume_${stamp}.out.log") -RedirectStandardError (Join-Path $root "resume_${stamp}.err.log")
 # Wait for a fresh queue heartbeat before starting its watcher.
 $ready=$false
 for($i=0;$i -lt 60;$i++){
  Start-Sleep -Seconds 2;$queue.Refresh()
  if($queue.HasExited){throw 'Queue exited. Inspect the new resume logs.'}
  try {$s=Get-Content -Encoding UTF8 -LiteralPath (Join-Path $root 'queue_status.json') -Raw | ConvertFrom-Json;if($s.queue_pid -eq $queue.Id){$ready=$true;break}}catch{}
 }
 if(!$ready){throw 'No fresh heartbeat. Queue may still be starting; do not start a second copy.'}
 $watcher=Start-Process -FilePath $python -ArgumentList @('-u','baseline_repair_20261123/finalize_tables.py','--watch') -WorkingDirectory $workspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root "finalizer_${stamp}.out.log") -RedirectStandardError (Join-Path $root "finalizer_${stamp}.err.log")
 Write-Host "Running. Queue PID=$($queue.Id), watcher PID=$($watcher.Id). Completed jobs skipped: $finished."
}
Move-Item -LiteralPath $marker -Destination (Join-Path $root ('user_shutdown_pause.resumed_'+$stamp+'.json'))
