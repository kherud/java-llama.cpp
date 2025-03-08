@echo off

mkdir build
cmake -Bbuild %*
cmake --build build --config RelWithDebInfo

if errorlevel 1 exit /b %ERRORLEVEL%