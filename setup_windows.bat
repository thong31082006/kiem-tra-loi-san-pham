@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

set "APP_PYTHON=%~dp0..\.venv-ml\Scripts\python.exe"
if exist "%APP_PYTHON%" goto :install

set "BASE_PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%BASE_PYTHON%" (
    echo Python 3.13 is required. Install Python 3.13 before running setup.
    set "APP_EXIT_CODE=1"
    goto :finish
)
"%BASE_PYTHON%" -m venv "%~dp0..\.venv-ml"
if errorlevel 1 goto :failed

:install
"%APP_PYTHON%" -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto :failed
"%APP_PYTHON%" -m pip install --no-cache-dir -r requirements.txt
if errorlevel 1 goto :failed
set "APP_EXIT_CODE=0"
echo Setup complete. Run run_app.bat or press F5 in VS Code.
goto :finish

:failed
set "APP_EXIT_CODE=%ERRORLEVEL%"
echo Setup failed. Check the error above.

:finish
popd
pause
exit /b %APP_EXIT_CODE%
