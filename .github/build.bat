@echo off

pushd ..
if not exist "build" (
    mkdir build
)
cd build
cmake .. %*
cmake --build . --config Release
popd

if errorlevel 1 exit /b %ERRORLEVEL%