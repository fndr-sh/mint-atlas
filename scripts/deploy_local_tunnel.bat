@echo off
REM Run Mint Atlas locally with a public URL (temporary demo deploy)
setlocal
cd /d "%~dp0.."

echo Starting Mint Atlas API on port 8000...
start "mint-atlas-api" /MIN cmd /c "mint-atlas serve --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

echo Starting public tunnel...
echo.
echo Your app will be available at the URL printed below.
echo Keep this window open. Ctrl+C to stop.
echo.
npx --yes localtunnel --port 8000

endlocal
