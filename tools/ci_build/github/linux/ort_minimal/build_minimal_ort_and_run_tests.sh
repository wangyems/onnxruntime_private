#!/bin/bash

# This script will create a minimal build with the required operators for all ORT format models
# in the testdata directory. This includes E2E models generated by build_full_ort_and_create_ort_files.sh.
# The build will run the unit tests for the minimal build, followed by running onnx_test_runner
# for the E2E test cases.

set -e
set -x
export PATH=/opt/python/cp38-cp38/bin:$PATH
USAGE_TEXT="Usage:
  -b|--build-directory <build directory>
    Specifies the build directory. Required.
  -c|--reduced-ops-config <reduced Ops config file>
    Specifies the reduced Ops configuration file path. Required.
  [--enable-type-reduction]
    Builds with type reduction enabled.
  [--enable-custom-ops]
    Builds with custom op support enabled.
  [--skip-model-tests]
    Does not run the E2E model tests."

BUILD_DIR=
REDUCED_OPS_CONFIG_FILE=
ENABLE_TYPE_REDUCTION=
MINIMAL_BUILD_ARGS=
SKIP_MODEL_TESTS=

while [[ $# -gt 0 ]]
do
    OPTION_KEY="$1"
    case $OPTION_KEY in
        -b|--build-directory)
            BUILD_DIR="$2"
            shift
            shift
            ;;
        -c|--reduced-ops-config)
            REDUCED_OPS_CONFIG_FILE="$2"
            shift
            shift
            ;;
        --enable-type-reduction)
            ENABLE_TYPE_REDUCTION=1
            shift
            ;;
        --enable-custom-ops)
            MINIMAL_BUILD_ARGS="custom_ops"
            shift
            ;;
        --skip-model-tests)
            SKIP_MODEL_TESTS=1
            shift
            ;;
        *)
            echo "Invalid option: $1"
            echo "$USAGE_TEXT"
            exit 1
            ;;
    esac
done

if [[ -z "${BUILD_DIR}" || -z "${REDUCED_OPS_CONFIG_FILE}" ]]; then
    echo "Required option was not provided."
    echo "$USAGE_TEXT"
    exit 1
fi

# Perform a minimal build with required ops and run ORT minimal build UTs
python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir ${BUILD_DIR} --cmake_generator Ninja \
    --config Debug \
    --skip_submodule_sync \
    --build_shared_lib \
    --parallel --use_binskim_compliant_compile_flags \
    --minimal_build ${MINIMAL_BUILD_ARGS} \
    --disable_ml_ops \
    --include_ops_by_config ${REDUCED_OPS_CONFIG_FILE} \
    ${ENABLE_TYPE_REDUCTION:+"--enable_reduced_operator_type_support"}

if [[ -z "${SKIP_MODEL_TESTS}" ]]; then
    # Run the e2e model test cases
    ${BUILD_DIR}/Debug/onnx_test_runner /onnxruntime_src/onnxruntime/test/testdata/ort_minimal_e2e_test_data
fi

# Print binary size info
python3 /onnxruntime_src/tools/ci_build/github/linux/ort_minimal/check_build_binary_size.py \
    --arch "$(uname -m)" --os "$(uname -o)" --build_config "minimal-reduced" \
    ${BUILD_DIR}/Debug/libonnxruntime.so

echo "Binary size info:"
cat ${BUILD_DIR}/Debug/binary_size_data.txt
