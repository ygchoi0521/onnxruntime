#!/bin/bash

# This script will run a full ORT build and use the python package built to generate ort format test files,
# and the exclude ops config file, which will be used in the build_minimal_ort_and_run_tests.sh

set -e

# Create an empty file to be used with build --include_ops_by_config, which will include no operators at all
echo -n > /home/onnxruntimedev/.test_data/include_no_operators.config

# Run a full build of ORT
# Since we need the ORT python package to generate the ORT format files and the include ops config files
# Will not run tests since those are covered by other CIs
python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir /build --cmake_generator Ninja \
    --config MinSizeRel \
    --skip_submodule_sync \
    --parallel \
    --android \
    --android_sdk_path /android_home \
    --android_ndk_path /android_home/ndk-bundle \
    --android_abi=arm64-v8a \
    --android_api=29 \
    --minimal_build \
    --build_shared_lib \
    --disable_ml_ops \
    --disable_exceptions \
    --test_binary_size \
    --include_ops_by_config /home/onnxruntimedev/.test_data/include_no_operators.config

# Install the ORT python wheel
python3 -m pip install --user mysql-connector-python

echo $BUILD_SOURCEVERSION
echo $BUILD_ID

cat /build/MinSizeRel/binary_size_data.txt

# Uninstall the ORT python wheel
python3 -m pip uninstall -y mysql-connector-python

# Clear the build
rm -rf /build/MinSizeRel
