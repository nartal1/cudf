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

mkdir -p /rapids/rmm/build
cd /rapids/rmm/build
cmake .. -DCMAKE_CXX11_ABI=OFF -DCMAKE_INSTALL_PREFIX=/usr/local/rapids/
make -j4 install

mkdir -p /rapids/custrings/cpp/build
cd /rapids/custrings/cpp/build
cmake .. -DCMAKE_CXX11_ABI=OFF -DCMAKE_INSTALL_PREFIX=/usr/local/rapids/
make -j4 install

mkdir -p $WORKDIR/cpp/build
cd $WORKDIR/cpp/build
cmake .. -DCMAKE_CXX11_ABI=OFF -DCMAKE_INSTALL_PREFIX=/usr/local/rapids/
make -j4 install

cd $WORKDIR/java
mvn package

