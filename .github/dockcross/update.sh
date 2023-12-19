#!/bin/bash

# This script prints the commands to upgrade the docker cross compilation scripts
docker run --rm dockcross/manylinux2014-x64  > ./dockcross-manylinux2014-x64
docker run --rm dockcross/manylinux2014-x86  > ./dockcross-manylinux2014-x86
docker run --rm dockcross/linux-arm64-lts    > ./dockcross-linux-arm64-lts
chmod +x ./dockcross-*
