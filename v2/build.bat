@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /I C:\Users\kelvi\vcpkg\installed\x64-windows\include prp2.cpp /link C:\Users\kelvi\vcpkg\installed\x64-windows\lib\OpenCL.lib /out:prp2.exe
