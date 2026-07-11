@echo off
REM ============================================================================
REM  launch.bat — start the Maverick trading cockpit.
REM
REM  What it does:
REM    1. Frees TCP :8062 if a stale server is holding it.
REM    2. Activates the `portfolio-manager` conda env (REQUIRED — that env has
REM       the pinned deps, incl. dash-ag-grid; see requirements.txt).
REM    3. Runs `python -m pm.app` from the repo root.
REM
REM  4. Once the server responds on :8062, a hidden watcher opens the default
REM     browser (it polls before opening, so a startup crash cannot masquerade
REM     as "site can't be reached"). The URL is also printed for manual open.
REM
REM  The window stays open on exit/error (pause at the end) so a Python
REM  traceback is readable instead of vanishing.
REM ============================================================================
setlocal

REM --- Free :8062 if a stale process is listening -----------------------------
echo Checking :8062...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr LISTENING ^| findstr :8062') do (
    echo   Found stale process %%p on :8062, killing...
    taskkill /F /PID %%p >nul 2>&1
)

echo === Maverick launching ===

REM --- Activate the portfolio-manager conda env -------------------------------
set CONDA_OK=0
for %%d in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\Miniconda3" "%USERPROFILE%\Anaconda3") do (
    if exist "%%~d\condabin\conda.bat" (
        call "%%~d\condabin\conda.bat" activate portfolio-manager 2>nul
        if not errorlevel 1 set CONDA_OK=1
    )
)
if "%CONDA_OK%"=="0" (
    where conda >nul 2>nul
    if not errorlevel 1 (
        call conda activate portfolio-manager 2>nul
        if not errorlevel 1 set CONDA_OK=1
    )
)
if "%CONDA_OK%"=="0" (
    echo ERROR: Could not activate conda env 'portfolio-manager'.
    echo Setup commands:
    echo     conda create -n portfolio-manager python=3.12 -y
    echo     conda activate portfolio-manager
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

python --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in env.
    pause
    exit /b 1
)

REM --- Run from the repo root so `pm` is importable ---------------------------
REM %~dp0 is this .bat's directory = the repo root (where pm/ lives).
cd /D "%~dp0"

echo.
echo   Starting server. The UI is reachable as soon as the server binds — the
echo   Bloomberg data loads *after* the page renders, with a spinner next to the
echo   Refresh BBG button. The browser opens automatically; if it does not, open:
echo.
echo       http://127.0.0.1:8062/
echo.

REM Background watcher: poll :8062 until it responds, THEN open the default
REM browser. The server now binds immediately (data loads post-render), so this
REM opens quickly on the skeleton UI; the spinner shows while BBG loads. Hidden
REM window; gives up after ~60s. Single line, no pipe — keeps quoting simple.
start "" powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 60;$i++){try{$r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:8062/'; Start-Process 'http://127.0.0.1:8062/'; break}catch{Start-Sleep -Seconds 1}}"

python -m pm.app

echo.
echo === Server stopped (or failed to start). Review any traceback above. ===
pause
endlocal
