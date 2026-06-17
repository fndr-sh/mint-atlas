@echo off
REM Deploy Mint Atlas to Hugging Face Spaces (Docker)
REM Usage: scripts\deploy_hf.bat YOUR_HF_USERNAME

setlocal
if "%~1"=="" (
  echo Usage: scripts\deploy_hf.bat YOUR_HF_USERNAME
  exit /b 1
)

set HF_USER=%~1
set SPACE_NAME=mint-atlas
set SPACE_REPO=https://huggingface.co/spaces/%HF_USER%/%SPACE_NAME%

echo.
echo === Mint Atlas HF Spaces Deploy ===
echo Space: %SPACE_REPO%
echo.

where huggingface-cli >nul 2>&1
if errorlevel 1 (
  echo Installing huggingface_hub...
  pip install -q "huggingface_hub[cli]"
)

huggingface-cli whoami >nul 2>&1
if errorlevel 1 (
  echo Login required. Run: huggingface-cli login
  echo Get token at https://huggingface.co/settings/tokens
  exit /b 1
)

huggingface-cli repo create %SPACE_NAME% --type space --space-sdk docker -y 2>nul

if not exist .hf_deploy (
  git clone %SPACE_REPO% .hf_deploy
)

cd .hf_deploy
git pull origin main 2>nul

REM Sync project files into HF repo
cd ..
xcopy /E /Y /I mint_atlas .hf_deploy\mint_atlas >nul
xcopy /E /Y /I frontend .hf_deploy\frontend >nul
xcopy /E /Y /I tests .hf_deploy\tests >nul
xcopy /E /Y /I scripts .hf_deploy\scripts >nul
xcopy /E /Y /I docs .hf_deploy\docs >nul
xcopy /E /Y /I data .hf_deploy\data >nul
copy /Y Dockerfile .hf_deploy\ >nul
copy /Y pyproject.toml .hf_deploy\ >nul
copy /Y README.md .hf_deploy\ >nul
copy /Y LICENSE .hf_deploy\ >nul
copy /Y .dockerignore .hf_deploy\ >nul

cd .hf_deploy
git add -A
git commit -m "Deploy Mint Atlas" 2>nul
git push

echo.
echo Live at: https://huggingface.co/spaces/%HF_USER%/%SPACE_NAME%
endlocal
