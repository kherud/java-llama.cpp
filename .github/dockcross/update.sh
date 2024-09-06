#!/bin/bash

# This script prints the commands to upgrade the docker cross compilation scripts
docker run --rm dockcross/manylinux2014-x64  > ./dockcross-manylinux2014-x64
docker run --rm dockcross/manylinux_2_28-x64  > ./dockcross-manylinux_2_28-x64
docker run --rm dockcross/manylinux2014-x86  > ./dockcross-manylinux2014-x86
docker run --rm dockcross/linux-arm64-lts    > ./dockcross-linux-arm64-lts
docker run --rm dockcross/android-arm        > ./dockcross-android-arm
docker run --rm dockcross/android-arm64      > ./dockcross-android-arm64
docker run --rm dockcross/android-x86        > ./dockcross-android-x86
docker run --rm dockcross/android-x86_64     > ./dockcross-android-x86_64
chmod +x ./dockcross-*
