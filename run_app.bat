@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

set "APP_PYTHON=%~dp0..\.venv-ml\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
    echo Python environment not found. Run setup_windows.bat first.
    set "APP_EXIT_CODE=1"
    goto :finish
)

"%APP_PYTHON%" main.py
set "APP_EXIT_CODE=%ERRORLEVEL%"

:finish
popd
pause
exit /b %APP_EXIT_CODE%
