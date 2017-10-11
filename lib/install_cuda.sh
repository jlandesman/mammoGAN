#!/bin/bash

sudo apt-get install python3-pip python3-dev
sudo -H pip3 install --upgrade pip

sudo apt-get install gcc

wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb

sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get install cuda

echo "export CUDA_HOME=/usr/local/cuda-8.0" >> ~/.profile
echo "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64" >> ~/.profile

echo "export PATH=/usr/local/cuda-8.0/bin:$PATH" >> ~/.profile
source ~/.profile


