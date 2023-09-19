#!/bin/bash

pushd ..
mkdir -p build
cd build
cmake .. $@
cmake --build . --config Release
popd
