@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "VQ_CONFIG_DIR=%ROOT%.vaultq"
set "VQ_ENV_FILE=%ROOT%.env"
if not defined VQ_EMBED_ENV_FILE set "VQ_EMBED_ENV_FILE=%ROOT%..\copyvector_mcp\copyvector_embed_pipeline\.env"
set "MCP_TRANSPORT=http"
set "MCP_HOST=127.0.0.1"
set "MCP_PORT=7073"
set "VQ_SELF_MAINTAIN_COLLECTION=second_brain"
set "VQ_SELF_MAINTAIN_INTERVAL_SECONDS=1800"
set "VQ_SELF_MAINTAIN_INITIAL_DELAY_SECONDS=60"
set "VQ_SELF_MAINTAIN_MAX_ACTIONS=20"
set "VQ_BACKGROUND_INDEX_COLLECTION=second_brain"
set "VQ_BACKGROUND_INDEX_POLL_SECONDS=300"
set "VQ_BACKGROUND_INDEX_INITIAL_DELAY_SECONDS=60"
set "VQ_BACKGROUND_NEW_FILE_DELAY_SECONDS=600"
set "VQ_BACKGROUND_CHANGED_INDEX_SECONDS=86400"
set "VQ_BACKGROUND_EMBED_LIMIT=100"
set "VQ_BACKGROUND_MAX_EMBED_BATCHES=1"

cd /d "%ROOT%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$conn = Get-NetTCPConnection -LocalAddress '%MCP_HOST%' -LocalPort %MCP_PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; " ^
  "if (-not $conn) { exit 1 }; " ^
  "$proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $conn.OwningProcess) -ErrorAction SilentlyContinue; " ^
  "if ($proc.CommandLine -match 'vaultq\.cli' -and $proc.CommandLine -match '\bmcp\b') { Write-Host 'VaultQ MCP already running on %MCP_HOST%:%MCP_PORT%'; exit 0 }; " ^
  "Write-Error ('Port %MCP_PORT% is already occupied by PID ' + $conn.OwningProcess + ': ' + $proc.CommandLine); exit 2"
if %ERRORLEVEL% EQU 0 exit /b 0
if %ERRORLEVEL% GEQ 2 exit /b %ERRORLEVEL%

"%PY%" -m vaultq.cli mcp --transport %MCP_TRANSPORT% --host %MCP_HOST% --port %MCP_PORT% --no-banner
