@echo off
title ChatBot Frontend Server

echo Starting Frontend Server...
echo.
echo Note: Make sure the backend server is already running.
echo.

cd /d %~dp0frontend
echo Current directory: %CD%
echo.

echo Running: npm start
echo.
echo Frontend will be available at: http://localhost:3000
echo.

npm start

echo.
echo Frontend server stopped.
pause 