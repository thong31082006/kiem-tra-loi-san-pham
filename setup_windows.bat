@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%APP_PYTHON%" goto :install

py -3.13 -c "import sys" >nul 2>&1
if errorlevel 1 goto :try_python
py -3.13 -m venv "%~dp0.venv"
if errorlevel 1 goto :failed
goto :install

:try_python
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>&1
if errorlevel 1 goto :missing_python
python -m venv "%~dp0.venv"
if errorlevel 1 goto :failed
goto :install

:missing_python
echo Python 3.13 is required. Install it with the Python launcher or add it to PATH.
set "APP_EXIT_CODE=1"
goto :finish

:install
"%APP_PYTHON%" -m pip install --no-cache-dir -r requirements.txt
if errorlevel 1 goto :failed
set "APP_EXIT_CODE=0"
echo Setup complete. Run run_app.bat.
goto :finish

:failed
set "APP_EXIT_CODE=%ERRORLEVEL%"
echo Setup failed. Check the error above.

:finish
popd
pause
exit /b %APP_EXIT_CODE%
