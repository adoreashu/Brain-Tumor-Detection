@echo off
echo ========================================================
echo        Brain Tumor Detection System - Launcher
echo ========================================================

echo Starting FastAPI Backend...
start "Brain Tumor Backend" cmd /k ".\venv\Scripts\activate && set PYTHONPATH=. && uvicorn app.backend.main:app --reload"

echo Starting React Frontend...
start "Brain Tumor Frontend" cmd /k "cd app\frontend && npm run dev"

echo.
echo All services are starting up!
echo - Frontend URL: http://localhost:3000
echo - Backend API: http://localhost:8000/docs
echo.
echo You can keep these terminal windows open to view logs.
pause
