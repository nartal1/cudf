#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
#########################################
# cuDF GPU build and test script for CI #
#########################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Set versions of packages needed to be grabbed
export NVSTRINGS_VERSION=0.7.*
export RMM_VERSION=0.7.*

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install -c rapidsai/label/cuda${CUDA_REL} -c rapidsai-nightly/label/cuda${CUDA_REL} rmm=${RMM_VERSION} nvstrings=${NVSTRINGS_VERSION}

logger "Install Openjdk"
conda install -c anaconda openjdk

logger "Install maven"
conda install --no-deps -c conda-forge maven

logger "Check versions..."
python --version
$CC --version
$CXX --version
java -version
mvn -version
conda list

################################################################################
# BUILD - Build libcudf and cuDF from source
################################################################################

logger "Build libcudf..."
$WORKSPACE/build.sh clean libcudf cudf

################################################################################
# TEST - Run GoogleTest and py.tests for libcudf and cuDF
################################################################################

logger "Check GPU usage..."
nvidia-smi

logger "GoogleTest for libcudf..."
cd $WORKSPACE/cpp/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/" make -j${PARALLEL_LEVEL} test

# Temporarily install cupy for testing
logger "pip install cupy"
pip install cupy-cuda92

# Temporarily install feather for testing
logger "conda install feather-format"
conda install -c conda-forge -y feather-format

logger "Python py.test for cuDF..."
cd $WORKSPACE/python
py.test --cache-clear --junitxml=${WORKSPACE}/junit-cudf.xml -v --cov-config=.coveragerc --cov=cudf --cov-report=xml:${WORKSPACE}/cudf-coverage.xml --cov-report term

################################################################################
# BUILD and TEST libcudfjni
################################################################################

logger "Build cudfjni"
cd $WORKSPACE/java
mvn clean test
