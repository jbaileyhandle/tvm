# TVM for ROCm gfx906 (Radeon VII)

## What this is

A working TVM build for GPU scheduling research on AMD Radeon VII (gfx906).
Based on `mvermeulen/tvm` branch `rocm-5.4-test` (commit `fe1e8ef43`), which
is apache/tvm `main` at Jan 16, 2023 plus 4 AMD engineer patches (bitcode
fixes, dense/matmul fix, debug check, logging).

## Patches on this branch

Two additional patches beyond `rocm-5.4-test`:

1. **`python/tvm/contrib/rocm.py`** — Changed bitcode path lookup to read
   `ROCM_PATH` env var instead of hardcoded `/opt/rocm/`. Added `import os`.

2. **`python/setup.py`** — Fixed `SameFileError` when the cmake build has
   already copied `.so` files and config directories into `python/tvm/`.

## Build environment

- **ROCm:** 5.2.3 (LLVM 14). Must use this version for both build and runtime.
  ROCm 5.7.1 (LLVM 17) is API-incompatible with this TVM commit.
- **GPU:** gfx906 (AMD Radeon VII)
- **Host:** Ubuntu with ROCm 5.2.3 installed at `/opt/rocm-5.2.3/`
- **Kernel driver:** ROCm 5.7.1 kernel driver works (backward compatible)
- **Python:** 3.10 via conda env `tvm_54`

## Build instructions

```bash
# Clone
git clone --branch rocm-5.2.3-gfx906 --single-branch --recursive \
    https://github.com/jbaileyhandle/tvm.git tvm_54_src
cd tvm_54_src

# Configure
mkdir -p build
cat > build/config.cmake << 'EOF'
set(USE_ROCM ON)
set(USE_LLVM /opt/rocm-5.2.3/llvm/bin/llvm-config)
set(USE_ROCBLAS OFF)
set(USE_MIOPEN OFF)
EOF

# Build
cd build
CPATH="" cmake .. -DCMAKE_BUILD_TYPE=Release
CPATH="" make -j$(nproc)
```

## Python dependencies (conda env `tvm_54`)

```bash
conda create -n tvm_54 python=3.10 -y
conda activate tvm_54
pip install "numpy<2" decorator scipy attrs psutil cloudpickle \
    tornado typing_extensions "setuptools>=65,<70"
```

## Running

Use the wrapper script (sets up all ROCm 5.2.3 env vars):
```bash
./run_tvm_54.sh script.py
./run_tvm_54.sh -c "import tvm; print(tvm.__version__)"
rocprof --stats ./run_tvm_54.sh my_model.py
```

Or manually:
```bash
export TVM_HOME=/path/to/tvm_54_src
export ROCM_PATH=/opt/rocm-5.2.3
export HIP_PATH=$ROCM_PATH/hip
export PYTHONPATH=$TVM_HOME/python
export LD_LIBRARY_PATH=$TVM_HOME/build:$ROCM_PATH/llvm/lib/:$ROCM_PATH/lib:$ROCM_PATH/hip/lib
export LIBRARY_PATH=$ROCM_PATH/llvm/lib/
export PATH=$ROCM_PATH/llvm/bin:$ROCM_PATH/bin:$ROCM_PATH/opencl/bin:$PATH
export CPATH=$ROCM_PATH/llvm/include/
python3 script.py
```

## Important notes for scheduling research

- **USE_ROCBLAS and USE_MIOPEN must be OFF.** When ON, TVM routes GEMM/conv to
  pre-compiled library kernels that bypass our custom LLVM scheduler entirely.
  With both OFF, TVM generates its own kernels compiled through the linked LLVM.

- **Do NOT auto-tune.** Use `opt_level=3` with fallback schedules (the default).
  Auto-tuning would pick different TVM-level schedules per scheduler variant,
  confounding the comparison. We want to hold the TVM schedule constant and
  only vary the LLVM instruction scheduling via `misched.txt`.

- **Scheduler singleton:** `MachineInstrSchedulerConfig::GetConfig()` reads
  `misched.txt` once per process. Each scheduler variant requires a separate
  process invocation.

- **TVM compiles internally via linked LLVM — no .hip source is produced.**
  `export_library()` produces `.o` object files. To use a custom scheduler,
  TVM must be rebuilt against a custom LLVM that has `MachineInstrSchedulerConfig`.

## Verified test results (2026-03-31, gfx906)

| Test | Size | opt_level | Runs | Result |
|------|------|-----------|------|--------|
| Element-wise relu | 1024x1024 | 3 | 10 | PASS (max_err=0.00) |
| Matmul (dense) | 1024x1024 | 3 | 10 | PASS (max_err~2e-4) |

Tested both in Docker (`mevermeulen/rocm-tvm:5.4.2`) and natively with this
build against ROCm 5.2.3 LLVM 14.

## Why ROCm 5.2.3 and not 5.7.1

ROCm 5.7.1 ships LLVM 17. This TVM commit cannot compile against LLVM 17 due
to removed headers (`PassManagerBuilder.h`), moved headers (`Triple.h`), and
changed APIs (`AnalysisManager`, `CodeGenFileType`). Porting all these fixes
from upstream TVM also brings in the code changes that cause output data
corruption at large tensor sizes (>= 1024x1024). ROCm 5.2.3 (LLVM 14) is the
newest installed version that is API-compatible with this TVM commit.

The ROCm 5.7.1 kernel driver is backward compatible with 5.2.3 userspace
libraries, so GPU execution works despite the version mismatch.
