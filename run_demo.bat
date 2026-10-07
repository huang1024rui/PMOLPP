@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
python demo.py %*
set "PMOLPP_EXIT_CODE=%errorlevel%"
popd
exit /b %PMOLPP_EXIT_CODE%
