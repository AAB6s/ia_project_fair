@echo off
setlocal
cd /d "%~dp0"
set LLM_BASE_URL=http://localhost:11434/v1
set LLM_API_KEY=ollama
set LLM_MODEL=llama3.1:8b
where ollama >nul 2>nul
if errorlevel 1 (
echo Ollama is not installed or not in PATH.
exit /b 1
)
ollama list >nul 2>nul
if errorlevel 1 (
echo Starting Ollama...
start "" /min ollama serve
timeout /t 5 /nobreak >nul
)
ollama list >nul 2>nul
if errorlevel 1 (
echo Could not connect to Ollama. Open the Ollama app and run this file again.
exit /b 1
)
ollama list | findstr /i "llama3.1:8b" >nul
if errorlevel 1 (
echo Pulling llama3.1:8b...
ollama pull llama3.1:8b
)
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
