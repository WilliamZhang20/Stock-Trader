@echo off
REM Daily paper-trading runner for the best strategy (CVaR + adaptive universe).
REM Invoked by Task Scheduler at 9:31 AM Eastern on weekdays.
setlocal EnableExtensions

set "REPO=C:\Users\wzhan\programming\Trading"
set "LOGDIR=%REPO%\logs"
set "PYTHON=C:\Users\wzhan\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Stamp each run; keep last ~60 days of logs by name.
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmmss"') do set "STAMP=%%i"
set "LOG=%LOGDIR%\paper_%STAMP%.log"

cd /d "%REPO%" || (
  echo [%DATE% %TIME%] ERROR: cannot cd to %REPO% >> "%LOG%"
  exit /b 1
)

echo ============================================================ >> "%LOG%"
echo [%DATE% %TIME%] Starting CVaR adaptive paper rebalance >> "%LOG%"
echo ============================================================ >> "%LOG%"

REM Skip weekends (Task Scheduler may still fire if set daily).
powershell -NoProfile -Command "if ((Get-Date).DayOfWeek -in 'Saturday','Sunday') { exit 2 }"
if %ERRORLEVEL%==2 (
  echo [%DATE% %TIME%] Weekend — skipping. >> "%LOG%"
  exit /b 0
)

REM Ensure Alpaca keys are available. Prefer User env (persisted); fall back to
REM whatever is already in this process. Task Scheduler user tasks inherit User env.
if "%APCA_API_KEY_ID%"=="" (
  for /f "usebackq delims=" %%k in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('APCA_API_KEY_ID','User')"`) do set "APCA_API_KEY_ID=%%k"
)
if "%APCA_API_SECRET_KEY%"=="" (
  for /f "usebackq delims=" %%k in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('APCA_API_SECRET_KEY','User')"`) do set "APCA_API_SECRET_KEY=%%k"
)

if "%APCA_API_KEY_ID%"=="" (
  echo [%DATE% %TIME%] ERROR: APCA_API_KEY_ID not set in process or User env. >> "%LOG%"
  exit /b 1
)
if "%APCA_API_SECRET_KEY%"=="" (
  echo [%DATE% %TIME%] ERROR: APCA_API_SECRET_KEY not set in process or User env. >> "%LOG%"
  exit /b 1
)

set "PIP_USER=0"

"%PYTHON%" -u cvar_trader.py --paper --fast --universe-size 8 --universe-criterion sharpe --universe-allocation adaptive >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

echo [%DATE% %TIME%] Finished with exit code %RC% >> "%LOG%"
exit /b %RC%
