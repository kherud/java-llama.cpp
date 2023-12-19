#!/bin/bash

pushd ..
mkdir -p build
cd build
cmake .. $@ || (popd && exit 1)
cmake --build . --config Release || (popd && exit 1)
popd
