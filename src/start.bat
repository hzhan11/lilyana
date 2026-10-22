@echo off
setlocal
if defined PYTHON goto custom_python
if exist "%~dp0..\.venv\Scripts\python.exe" goto shared_python
if exist "%~dp0..\.venv\bin\python" goto shared_posix_python
where python >nul 2>nul
if not errorlevel 1 goto python
where py >nul 2>nul
if not errorlevel 1 goto py
echo Python 3 is required. Install it or set PYTHON to its executable. 1>&2
exit /b 1

:custom_python
"%PYTHON%" "%~dp0launcher.py" %*
exit /b %errorlevel%

:shared_python
"%~dp0..\.venv\Scripts\python.exe" "%~dp0launcher.py" %*
exit /b %errorlevel%

:shared_posix_python
"%~dp0..\.venv\bin\python" "%~dp0launcher.py" %*
exit /b %errorlevel%

:python
python "%~dp0launcher.py" %*
exit /b %errorlevel%

:py
py -3 "%~dp0launcher.py" %*
exit /b %errorlevel%
