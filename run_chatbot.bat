@echo off
title ChatBot Launcher

echo Starting ChatBot Application...
echo.
echo This script will start both backend and frontend servers.
echo.

echo Step 1: Starting backend server...
cd /d %~dp0backend
start "Backend" cmd /c python -m uvicorn main:app --host 0.0.0.0 --port 8000

echo Step 2: Waiting 45 seconds for backend initialization...
timeout /t 45 /nobreak > nul

echo Step 3: Starting frontend server...
cd /d %~dp0frontend
start "Frontend" cmd /c npm start

cd /d %~dp0
echo.
echo Both servers have been started.
echo The web browser will open automatically to http://localhost:3000
echo.
echo To stop the application, close this window and the server windows.
echo.

pause 