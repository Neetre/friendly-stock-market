@ECHO OFF

CALL ./setup/setup_Windows.bat

CLS

REM Change directory to 'bin' and run the application
CD /d bin
IF ERRORLEVEL 1 (
    ECHO Failed to change directory to 'bin'
    PAUSE
    EXIT /B 1
)

python main.py --help
IF ERRORLEVEL 1 (
    ECHO Failed to run the application
    PAUSE
    EXIT /B 1
)
