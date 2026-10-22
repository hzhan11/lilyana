@echo off
call "%~dp0..\..\src\start.bat" --lesson 1 %*
exit /b %errorlevel%
