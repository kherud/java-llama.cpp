#!/bin/bash

mkdir build
cmake -Bbuild $@ || exit 1
cmake --build build --config Release || exit 1
