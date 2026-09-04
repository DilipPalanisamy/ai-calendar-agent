@echo off
setlocal
title AI Calendar Assistant - Desktop Setup
color 0B

echo =======================================================
echo          AI Calendar Assistant - App Installer
echo =======================================================
echo.
echo Setting up AI Calendar Assistant on your laptop/PC...
echo.

set "APP_NAME=AI Calendar Assistant"
set "APP_URL=https://ai-calendar-agent-we11.onrender.com"
set "DESKTOP_DIR=%USERPROFILE%\Desktop"
set "SHORTCUT_PATH=%DESKTOP_DIR%\%APP_NAME%.url"

:: Create Desktop Internet Shortcut
(
echo [InternetShortcut]
echo URL=%APP_URL%
echo IconIndex=0
echo IconFile=C:\Windows\System32\shell32.dll,44
) > "%SHORTCUT_PATH%"

:: Also try creating rich PowerShell shortcut for Chrome / Edge App Mode
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell; " ^
  "$chromePath = 'C:\Program Files\Google\Chrome\Application\chrome.exe'; " ^
  "$edgePath = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'; " ^
  "$target = $null; $args = ''; " ^
  "if (Test-Path $chromePath) { $target = $chromePath; $args = '--app=' + '%APP_URL%'; } " ^
  "elseif (Test-Path $edgePath) { $target = $edgePath; $args = '--app=' + '%APP_URL%'; } " ^
  "else { $target = '%APP_URL%'; }; " ^
  "$s = $ws.CreateShortcut([System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), '%APP_NAME%.lnk')); " ^
  "$s.TargetPath = $target; " ^
  "if ($args -ne '') { $s.Arguments = $args; }; " ^
  "$s.Description = 'AI Calendar Assistant - Powered by Gemini'; " ^
  "$s.Save();" 2>nul

echo [SUCCESS] App Shortcut installed to your Desktop!
echo.
echo Launching %APP_NAME%...
start "" "%SHORTCUT_PATH%"

timeout /t 3 >nul
exit
