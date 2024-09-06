#!/bin/sh

# A Cuda 12.1 install script for RHEL8/Rocky8/Manylinux_2.28

sudo dnf install -y kernel-devel kernel-headers
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# We prefer CUDA 12.1 as it's compatible with 12.2+
sudo dnf install -y cuda-toolkit-12-1

exec .github/build.sh $@ -DGGML_CUDA=1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc