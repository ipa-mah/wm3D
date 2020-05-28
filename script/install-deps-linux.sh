#!/bin/sh

set -ev
# remove old eigen in system
sudo rm -rf /usr/include/eigen3
# download eigen3 and unpack it
cd /tmp
wget -O eigen3.zip http://bitbucket.org/eigen/eigen/get/3.3.7.zip
unzip -q eigen3.zip
ls -l eigen*
sudo mv /tmp/eigen-eigen-323c052e1731 /usr/include/eigen3
# check eigen version
cat /usr/include/eigen3/Eigen/src/Core/util/Macros.h | grep VERSION

# Install Open3D from source
git clone --recursive https://github.com/intel-isl/Open3D
cmake -DBUILD_EIGEN3=OFF -DBUILD_PNG=OFF -DBUILD_JSONCPP=OFF -DBUILD_FLANN=OFF -DGLIBCXX_USE_CXX11_ABI=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 ..
