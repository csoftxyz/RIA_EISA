@echo off
REM ============================================================================
REM  C21 Unified-Flavour Verification Suite -- one-click launcher
REM  Opens the graphical control panel. Every program is a button with a
REM  plain-language explanation and a picture; press Run to execute it and
REM  watch the live output. "RUN ALL CHECKS" runs the master verifier
REM  (verify_prd.py, 70 assertions).
REM
REM  Project started: February 2025.
REM  Locked environment: Python 3.6.6 / numpy 1.19.5 / scipy 1.5.4 / mpl 3.3.4
REM ============================================================================
setlocal
cd /d "%~dp0"
set "PYTHONPATH="

REM Prefer the project's locked Python 3.6 interpreter if present.
set "PYEXE=D:\Python36\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo Launching C21 verification control panel...
"%PYEXE%" "08_gui\runall_gui.py"
if errorlevel 1 (
    echo.
    echo The GUI failed to start. Falling back to the command-line master verifier.
    "%PYEXE%" "04_verification_code\verify_prd.py"
)
endlocal
