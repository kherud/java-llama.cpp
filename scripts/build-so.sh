#!/bin/bash

# Script for compiling on amd64 and cross-compiling
# from amd64 to aarch64 and armhf

SCRIPT_DIR=$(dirname -- "$( readlink -f -- "$0"; )")
RESOURCES_DIR=linux-x86-64
cd ..
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cp llama.h ..
mkdir build
cd build
AARCH=$(dpkg --print-architecture)
case $AARCH in
  amd64)
    cmake -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF ..
    # Copy it to linux-amd64 in case different JNA versions check different locations
    mkdir -p ../../src/main/resources/linux-amd64
    cp libllama.so ../../src/main/resources/linux-amd64/
    ;;
  arm64)
    cmake -DCMAKE_TOOLCHAIN_FILE=$SCRIPT_DIR/aarch64-linux-gnu-toolchain.cmake -DBUILD_SHARED_LIBS=ON ..
    RESOURCES_DIR=linux-aarch64
    ;;
  armhf|armv7l)
    cmake -DCMAKE_TOOLCHAIN_FILE=$SCRIPT_DIR/arm-linux-gnueabihf-toolchain.cmake -DBUILD_SHARED_LIBS=ON ..
    RESOURCES_DIR=linux-arm
    ;;
esac

cmake --build . --config Release
cd ../..
mkdir -p src/main/resources/$RESOURCES_DIR
cp llama.cpp/build/libllama.so src/main/resources/$RESOURCES_DIR
rm -rf llama.cpp
cd scripts
