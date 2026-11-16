@echo off
chcp 65001 > nul 2>&1
setlocal

where situs.exe >nul 2>&1
if %ERRORLEVEL% equ 0 (
    situs.exe %*
    goto :eof
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py -3 -m situs %*
    goto :eof
)

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -m situs %*
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
