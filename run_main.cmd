@echo off
setlocal
title Molding Inspection
cd /d "%~dp0"
if errorlevel 1 exit /b 1

set "UV_EXE=uv.exe"
where uv.exe >nul 2>nul
if not errorlevel 1 goto run

set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if exist "%UV_EXE%" goto run

echo uv.exe was not found. Install uv or add it to PATH.
pause
exit /b 1

:run
echo Starting inspection. Press Ctrl+C in this window to stop.
"%UV_EXE%" run python main.py %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" exit /b 0

echo.
echo Inspection stopped with exit code %EXIT_CODE%. See the message above.
pause
exit /b %EXIT_CODE%
