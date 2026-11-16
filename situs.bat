@echo off
chcp 65001 > nul 2>&1
setlocal

:: 1. If inside an active virtual environment, strictly prioritize that Python interpreter
if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        "%VIRTUAL_ENV%\Scripts\python.exe" -m situs %*
        goto :eof
    )
)

:: 2. Pre-built standalone binary
where situs.exe >nul 2>&1
if %ERRORLEVEL% equ 0 (
    situs.exe %*
    goto :eof
)

:: 3. Active PATH Python interpreter
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -m situs %*
    goto :eof
)

:: 4. Windows Python Launcher fallback
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py -3 -m situs %*
    goto :eof
)

where python3 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python3 -m situs %*
    goto :eof
)

echo [Situs Error] Python was not found on your system PATH.
echo Please install Python 3.10+ from https://python.org and ensure it is added to your PATH.
exit /b 1
