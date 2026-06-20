@echo off
setlocal EnableExtensions

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'vaultq\.cli' -and $_.CommandLine -match '\bmcp\b' -and $_.CommandLine -match '\b7073\b' }; foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }; if ($procs) { Write-Host ('Stopped VaultQ MCP process(es): ' + (($procs | ForEach-Object ProcessId) -join ', ')) } else { Write-Host 'VaultQ MCP was not running.' }"
