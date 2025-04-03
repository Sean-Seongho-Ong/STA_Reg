@echo off
title ChatBot Backend Server

echo Starting Backend Server...
echo.

cd /d %~dp0backend
echo Current directory: %CD%
echo.

echo Running: python -m uvicorn main:app --host 0.0.0.0 --port 8000
echo.
echo Server will be available at: http://localhost:8000
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000

echo.
echo Backend server stopped.
pause 