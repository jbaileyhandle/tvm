#!/bin/bash
# Canonical wrapper for running Python against this TVM build.
#
# Sets the env vars TVM needs, then exec's the conda env's Python.
# Self-locates: TVM_HOME defaults to the parent of this script's directory,
# so the wrapper works wherever the TVM tree is cloned. Override TVM_HOME
# to point at a different TVM checkout if desired.
#
# Usage:
#   ./run_tvm.sh script.py [args...]
#   ./run_tvm.sh -c "import tvm; print(tvm.__version__)"
#   rocprof --stats ./run_tvm.sh script.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export TVM_HOME="${TVM_HOME:-$(dirname "$SCRIPT_DIR")}"
export ROCM_PATH=/opt/rocm-5.2.3
export HIP_PATH=$ROCM_PATH/hip
export PYTHONPATH=$TVM_HOME/python
export LD_LIBRARY_PATH=$TVM_HOME/build:$ROCM_PATH/llvm/lib/:$ROCM_PATH/lib:$ROCM_PATH/hip/lib
export LIBRARY_PATH=$ROCM_PATH/llvm/lib/
export PATH=$ROCM_PATH/llvm/bin:$ROCM_PATH/bin:$ROCM_PATH/opencl/bin:$PATH
export CPATH=$ROCM_PATH/llvm/include/

exec /home/jbaile/miniconda3/envs/tvm_54/bin/python3 "$@"
