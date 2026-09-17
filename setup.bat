@echo off
echo.
echo ======================================
echo   DeepTrace AI - Project Setup
echo ======================================
echo.

echo [1/3] Setting up Frontend...
cd frontend
call npm install
copy .env.example .env
echo   Done. Edit frontend\.env and add VITE_ANTHROPIC_API_KEY
cd ..

echo.
echo [2/3] Setting up Backend...
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt -q
copy .env.example .env
echo   Done. Edit backend\.env and add ANTHROPIC_API_KEY
cd ..

echo.
echo [3/3] Setting up ML Pipeline...
cd ml
pip install -r requirements.txt -q
mkdir data 2>nul
mkdir checkpoints 2>nul
cd ..

echo.
echo ======================================
echo   Setup Complete!
echo ======================================
echo.
echo  Next steps:
echo  1. Add your Anthropic API key to .env files
echo  2. cd frontend ^&^& npm run dev
echo  3. cd backend ^&^& uvicorn app.main:app --reload
echo.
echo  Frontend: http://localhost:5173
echo  Backend:  http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo.
pause
