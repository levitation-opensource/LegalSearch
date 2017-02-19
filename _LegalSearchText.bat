@echo off


title Search Legal Texts WITHOUT logging


if not defined iammaximized (
    set iammaximized=1
    start /max "" "%~0"
    exit
)


REM change screen dimensions
mode con: cols=170 lines=9999


REM change active dir to current location
%~d0
cd /d "%~dp0"


if not exist LegalSearchText.exe (
	echo.
	echo Please move this .bat file to the same folder as the .exe file!
	:loop_error
	set /p dummy=
	goto :loop_error
)


echo.
echo.
echo Please note: this program does NOT store your queries or results into a log file.
echo If you need logging, please run _LegalSearchText_with_logging.bat


:loop


echo.
echo.
set /p query=Enter search query or command: 
echo.


LegalSearchText.exe %query%


goto :loop
