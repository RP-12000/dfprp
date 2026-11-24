@echo off
rem command-line build without opening Visual Studio: build.bat [Release^|Debug]
set CFG=%1
if "%CFG%"=="" set CFG=Release
"C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" "%~dp0dfprp.slnx" /m /nologo /v:minimal /p:Configuration=%CFG% /p:Platform=x64
