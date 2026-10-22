<#
.SYNOPSIS
    Situs PowerShell CLI Wrapper

.DESCRIPTION
    Convenience wrapper for launching Situs CLI commands in Windows PowerShell.
    Works seamlessly across any device with Python 3.10+.

.EXAMPLE
    .\situs.ps1 setup
    .\situs.ps1 coordinator --port 8000
    .\situs.ps1 join 172.20.164.173:8000
    .\situs.ps1 chat
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

# Force UTF-8 output encoding in PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# Try invoking the installed situs executable or fallback to python -m situs
$situsCmd = Get-Command "situs" -ErrorAction SilentlyContinue

if ($situsCmd) {
    & situs $CliArgs
} else {
    $pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command "python3" -ErrorAction SilentlyContinue
    }
    if ($pythonCmd) {
        & $pythonCmd.Source -m situs $CliArgs
    } else {
        Write-Error "Python was not found on your system PATH. Please install Python 3.10+ from python.org."
        exit 1
    }
}
