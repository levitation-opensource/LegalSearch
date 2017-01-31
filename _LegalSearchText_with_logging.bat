@echo off


title Search Legal Texts with LOGGING


if not defined iammaximized (
    set iammaximized=1
    start /max "" "%~0"
    exit
)


REM change screen dimensions
mode con: cols=170 lines=9999


REM change active dir to current location
%~d0
cd /d %~dp0


if not exist LegalSearchText.exe (
	echo.
	echo Please move this .bat file to the same folder as the .exe file!
	:loop_error
	set /p dummy=
	goto loop_error
)


echo.
echo.
echo Please note: this program stores all your queries and results into a log file named:
echo LegalSearchTextLog.txt


:loop


echo.
echo.
set /p query=Enter search query or command: 
echo.


echo. >> LegalSearchTextLog.txt
echo. >> LegalSearchTextLog.txt
echo Query: %query% >> LegalSearchTextLog.txt
echo. >> LegalSearchTextLog.txt


FOR /F "tokens=*" %%I IN ('LegalSearchText.exe %query%') DO (
	echo %%I
	echo %%I >> LegalSearchTextLog.txt
)


goto loop
