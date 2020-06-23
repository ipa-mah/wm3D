
# Upgrade cmake from 3.10 to 3.16 for Open3D
#https://peshmerge.io/how-to-install-cmake-3-11-0-on-ubuntu-16-04/
## Install ros melodic
 
## Install eigen
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
#git clone --recursive https://github.com/intel-isl/Open3D
#cmake -DBUILD_EIGEN3=OFF -DBUILD_PNG=OFF -DBUILD_JSONCPP=OFF -DBUILD_FLANN=OFF -DGLIBCXX_USE_CXX11_ABI=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 ..

# Install OpenCV from source
#CUDA 10
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_PERF_TESTS=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" -DCUDA_ARCH_BIN=7.5 -DBUILD_opencv_cudacodec=OFF -D OPENCV_EXTRA_MODULES_PATH=/home/hah/Downloads/opencv_contrib/modules  -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D OPENCV_ENABLE_NONFREE=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D BUILD_opencv_python3=ON -D HAVE_opencv_python3=ON -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3.6 ..
#CUDA 8
#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_PERF_TESTS=OFF -#DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" -DCUDA_ARCH_BIN=6.1 -D OPENCV_EXTRA_MODULES_PATH=/../../opencv_contrib/modules -D HAVE_opencv_python3=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D #OPENCV_ENABLE_NONFREE=ON ..

# Problem with GL
# https://www.it-swarm.dev/es/compiling/el-destino-importado-qt5-gui-hace-referencia-al-archivo-usrlibx86-64-linux-gnulibegl.so-pero-este-archivo-no-existe./961362861/
#sudo rm /usr/lib/x86_64-linux-gnu/libEGL.so
#sudo ln /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so
#sudo rm /usr/lib/x86_64-linux-gnu/libGL.so
#sudo ln /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
