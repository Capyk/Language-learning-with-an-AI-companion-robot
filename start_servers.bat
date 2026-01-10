@echo off
echo ===================================================
echo   Starting AI Companion Robot App (Fast Mode)
echo ===================================================

echo [1/2] Starting Backend (Port 8000)...
start "AI Backend" cmd /k "cd backend && (if exist .venv_new (call .venv_new\Scripts\activate) else if exist .venv (call .venv\Scripts\activate)) && python -m uvicorn app.main:app --reload"

echo [2/2] Starting Frontend (Port 5173)...
start "AI Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Success! Both servers are starting up.
echo - Backend: http://localhost:8000
echo - Frontend: http://localhost:5173
echo.
echo You can close this window now (servers will stay open).
pause
