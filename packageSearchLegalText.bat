


pyinstaller --noconfirm LegalSearchText.py > nul


copy /Y copyrights.txt ".\dist\LegalSearchText\" 
copy /Y corpus_references.txt ".\dist\LegalSearchText\" 
copy /Y development_instructions.txt ".\dist\LegalSearchText\" 
copy /Y _readme.txt ".\dist\LegalSearchText\" 
copy /Y _LegalSearchText.bat ".\dist\LegalSearchText\"
copy /Y _LegalSearchText_with_logging.bat ".\dist\LegalSearchText\"
robocopy "et-en" "dist\LegalSearchText\et-en" /S /E /XO > nul
robocopy "en-et_t" "dist\LegalSearchText\en-et_t" /S /E /XO > nul
robocopy "en-et_u" "dist\LegalSearchText\en-et_u" /S /E /XO > nul


del LegalSearchText_new.zip > nul 2>nul

"C:\Program Files (x86)\7-Zip\7z.exe" a -mx=9 -r -y LegalSearchText_new.zip ".\dist\LegalSearchText" > nul


pause

