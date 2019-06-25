#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

SOURCE_BRANCH=master

# Restrict uploads to master branch
if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ] && [ "${UPLOAD_CUDFJNI}" != 1 ]; then
  echo "Skipping upload"
  return 0
fi

#Uploads jars for CUDA 9.2 and CUDA 10.0
if [ "$UPLOAD_LIBCUDF" == "1" ]; then
    logger "Upload cudfjni"
    cd $WORKSPACE/java
    mvn -Dmaven.repo.local=$WORKSPACE/.m2 deploy -DskipTests
fi
