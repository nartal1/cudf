#########################################################################
# File Name: build-centos7.sh
# Author: test

set -e

WORKDIR=`pwd`

export GIT_COMMITTER_NAME="ci"
export GIT_COMMITTER_EMAIL="ci@nvidia.com"

cd /rapids/
git clone --recurse-submodules https://github.com/rapidsai/rmm.git
git clone --recurse-submodules https://github.com/rapidsai/custrings.git

gcc --version

export CUDACXX=/usr/local/cuda/bin/nvcc
export INSTALL_PREFIX=/usr/local/rapids

mkdir -p /rapids/rmm/build
cd /rapids/rmm/build
cmake .. -DCMAKE_CXX11_ABI=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
make -j4 install

mkdir -p /rapids/custrings/cpp/build
cd /rapids/custrings/cpp/build
cmake .. -DCMAKE_CXX11_ABI=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
make -j4 install

mkdir -p $WORKDIR/cpp/build
cd $WORKDIR/cpp/build
export RMM_ROOT=$INSTALL_PREFIX
export NVSTRINGS_ROOT=$INSTALL_PREFIX
cmake .. -DCMAKE_CXX11_ABI=OFF
make -j4 install DESTDIR=$INSTALL_PREFIX

cd $WORKDIR/java
mvn -P abiOff package

