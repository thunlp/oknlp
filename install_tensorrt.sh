#!/bin/bash
echo 'USAGE: bash install_tensorrt.sh path_to_tensorrt.deb'

USE_PYTHON3=false
TRT_PATH=''

if [[ $# -gt 1 ]]; then
TRT_PATH=$1
fi

# pip install 'pycuda>=2017.1.1'

if [[ $USE_PYTHON3!=true ]]; then
pip3 install 'pycuda>=2017.1.1'
else
pip2 install 'pycuda>=2017.1.1'
fi

# Install TensorRT from the Debian package.
dpkg -i nv-tensorrt-repo-ubuntu1x04-cudax.x-trt5.x.x.x-rc-yyyymmdd_1-1_amd64.deb
# apt-key add /var/nv-tensorrt-repo-cudax.x-trt5.x.x.x-rc-yyyymmdd/7fa2af80.pub

dpkg -i $TRT_PATH
TRT_KEY=`ls /var/nv-tensorrt*/*.pub | grep 7fa2af80.pub`
apt-key add $TRT_KEY

# apt-get update
apt-get -y -q install tensorrt

if [[ $USE_PYTHON3!=true ]]; then

apt-get install -y -q python-libnvinfer-dev
# If using Python 3.x:
else

apt-get install -y -q python3-libnvinfer-dev

fi

echo '====================='
echo 'Installation finished'
echo '---> List of installed packages'
dpkg -l | grep TensorRT
