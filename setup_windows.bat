@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%APP_PYTHON%" goto :install

py -3.13 -c "import struct, sys; sys.exit(struct.calcsize('P') != 8)" >nul 2>&1
if errorlevel 1 goto :try_python
py -3.13 -m venv "%~dp0.venv"
if errorlevel 1 goto :failed
goto :install

:try_python
python -c "import struct, sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 13) and struct.calcsize('P') == 8))" >nul 2>&1
if errorlevel 1 (
    echo Install Python 3.13 64-bit from python.org and enable Add Python to PATH.
    echo Python 3.11 and 3.12 64-bit are also supported through the python command.
    set "APP_EXIT_CODE=1"
    goto :finish
)
python -m venv "%~dp0.venv"
if errorlevel 1 goto :failed

:install
"%APP_PYTHON%" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :failed
"%APP_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
set "APP_EXIT_CODE=0"
echo Setup complete. Run run_app.bat or press F5 in VS Code.
goto :finish

:failed
set "APP_EXIT_CODE=%ERRORLEVEL%"
echo Setup failed. Check the error above.

:finish
popd
if /i not "%~1"=="--no-pause" pause
exit /b %APP_EXIT_CODE%
