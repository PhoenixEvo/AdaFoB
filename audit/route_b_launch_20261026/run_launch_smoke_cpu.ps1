[CmdletBinding()]
param([string]$PythonPath='C:\Users\Admin\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe')
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'));Set-Location -LiteralPath $root
. (Join-Path $root 'audit/step9_cpu_recovery_20261024T133636Z/offline_cpu_launcher_utils_v3.ps1')
$env:CUDA_VISIBLE_DEVICES='';$env:PIP_NO_INDEX='1';$env:HF_HUB_OFFLINE='1';$env:OMP_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=(Join-Path $root 'audit/step9_cpu_recovery_20261024T133636Z/offline_guard')+';'+$root
foreach($n in @('PYTHONHOME','PYTHONNOUSERSITE','PYLAUNCHER_ALLOW_INSTALL')){[Environment]::SetEnvironmentVariable($n,$null,'Process')}
$lock=Get-Content (Join-Path $PSScriptRoot 'package_lock.json') -Raw|ConvertFrom-Json
foreach($r in $lock.files){if((Get-FileHash -LiteralPath (Join-Path $root $r.path) -Algorithm SHA256).Hash -ne $r.sha256){throw ('Source/manifest hash mismatch: '+$r.path)}}
$out=Join-Path $PSScriptRoot ('human_launch_smoke_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')+'_'+[Guid]::NewGuid().ToString('N').Substring(0,8));New-Item -ItemType Directory $out|Out-Null
$state=@{utc=[DateTime]::UtcNow.ToString('o');python=$PythonPath;status='NOT RUN';no_inference=$true;gpu_hidden=$true;failure=$null}
$code=1
try{
 if(!(Test-Path -LiteralPath $PythonPath -PathType Leaf)){throw 'Pass -PythonPath to an installed interpreter. No installation permitted.'}
 Write-Host 'CPU synthetic launch-wrapper smoke only: eight artificial cases, timing wrappers, cache caps, output writer. No real models/checkpoint loads, no GPU, no installs/network.'
 Write-Host "Logs and evidence: $out"
 $r=Invoke-OfflineProcess -Executable $PythonPath -Arguments @('-u','-m','experiments.route_b_preflight_v1.launch_pilot_v1','--out',(Join-Path $out 'checks')) -LogStem (Join-Path $out 'preflight') -TimeoutSeconds 120
 $state.exit_code=$r.exit_code;$state.status=$(if($r.exit_code -eq 0){'CPU CHECKS EXECUTED SUCCESSFULLY; review all evidence'}else{'FAIL or BLOCKED; inspect logs'})
 Get-Content (Join-Path $out 'preflight.stdout.log');Get-Content (Join-Path $out 'preflight.stderr.log')
 $code=$r.exit_code
}catch{$state.failure=$_.Exception.Message;Write-Host $state.failure}finally{
 $state.driver_exit=$code;Write-NewJson -Path (Join-Path $out 'driver_status.json') -Value $state
 Write-Host "SEND BACK: $out"
}
exit $code