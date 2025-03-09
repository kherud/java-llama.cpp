@echo off

mkdir build
cmake -Bbuild %*
cmake --build build --config Release

if errorlevel 1 exit /b %ERRORLEVEL%