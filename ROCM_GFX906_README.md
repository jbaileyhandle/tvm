# TVM for ROCm gfx906 — Custom Scheduler Research Setup

A working TVM build paired with custom LLVM toolchains, used for evaluating
custom GPU instruction-scheduling strategies on AMD Radeon VII (gfx906) via
real ML workloads compiled by TVM. This document is the entry point for the
overall setup: where things live, how the pieces fit together, how to build
and run, and what's been validated so far.

For a much shorter project-status summary including in-progress decisions,
see `~/gpu2/apps/pytorch_to_hip_tests/path1_tvm/STATUS.md`. For the deep
historical investigation including failed paths, see that directory's
`../investigation_guide.md`.

---

## Project goal

PyTorch model → TVM → LLVM IR → AMDGPU ISA → gfx906 → rocprof

…where the LLVM IR → AMDGPU ISA step is done by a **custom LLVM** that reads
`misched.txt` from CWD to select an experimental `MachineInstrSchedulerConfig`.
The custom scheduler can then be A/B compared across real ML kernels.

## Layout — where the pieces live

| Path | What |
|------|------|
| `~/gpu2/tvm_repos/tvm-amd-rocm54-fork/` | **(this dir)** TVM source + built libraries. Forked from `mvermeulen/tvm` branch `rocm-5.4-test` (commit `fe1e8ef43`); our branch is `rocm-5.2.3-gfx906`. Remote: `git@github.com:jbaileyhandle/tvm.git`. |
| `~/gpu2/tvm_repos/tvm-amd-rocm54-fork/build/` | Compiled `libtvm.so` + `libtvm_runtime.so`, plus `config.cmake` that pins `USE_LLVM=/opt/rocm-5.2.3/llvm/bin/llvm-config`. |
| `~/gpu2/llvm_repos/llvm-project/` | Custom LLVM 17 fork. Branch `jbailey_handle_import_gpu_on_gpu_features`. Has `MachineInstrSchedulerConfig`. Remote: `jbaileyhandle/llvm-project`. |
| `~/gpu2/llvm_repos/gpu_on_gpu_llvm-project/` | Custom LLVM 15 fork. Branch `jbaile_tmp_two_pass_debug`. Older snapshot of the scheduler work; same major LLVM as the known-good Docker reference. |
| `/opt/rocm-5.2.3/` | ROCm 5.2.3 userspace (LLVM 14, HIP runtime, device bitcode). What TVM is built and run against. |
| `/opt/rocm-5.7.1/` | ROCm 5.7.1 kernel driver (active). Backward compatible with 5.2.3 userspace. Not used at the userspace level. |
| `~/miniconda3/envs/tvm_54/` | Python 3.10 conda env that the wrapper script exec's. Has TVM's Python deps (numpy<2, scipy, decorator, etc.). |
| `~/gpu2/apps/pytorch_to_hip_tests/path1_tvm/` | **R&D area.** Shell wrapper, original v0.14.0 experiments, baseline rocprof scripts, the split-codegen experiment (`backend_experiment/`). |
| `~/gpu2/apps/gpu2_benchmarks/TVM/` (future) | Where TVM benchmarks integrated with the `gpu2_benchmarks` Make-based envelope will go. |

## Quick start

To run a Python script that imports TVM and exercises the GPU:

```bash
cd ~/gpu2/apps/pytorch_to_hip_tests/path1_tvm
./run_tvm_54.sh some_script.py             # ordinary run
rocprof --stats ./run_tvm_54.sh some_script.py   # with profiling
```

To verify TVM imports correctly:
```bash
./run_tvm_54.sh -c "import tvm; print(tvm.__version__)"   # → 0.11.dev0
```

The wrapper sets `TVM_HOME`, `ROCM_PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH`,
`PATH`, etc., then `exec`'s the right Python interpreter. See *Wrapper script*
below for details.

## Build configuration

This is what was used to produce the current `build/libtvm.so` (Mar 31, 2026):

```cmake
# build/config.cmake
set(USE_ROCM ON)
set(USE_LLVM /opt/rocm-5.2.3/llvm/bin/llvm-config)
set(USE_ROCBLAS OFF)
set(USE_MIOPEN OFF)
```

Why these choices:

- **`USE_LLVM` → LLVM 14 from ROCm 5.2.3**. LLVM is statically linked into
  `libtvm.so` at build time. We picked 5.2.3's LLVM 14 because the
  `rocm-5.4-test` source can't compile against LLVM 17 (removed/moved headers,
  changed APIs) and 5.2.3's LLVM 14 is the newest installed LLVM with which it
  builds cleanly. The "14" stays inside `libtvm.so` forever; everything else
  about ROCm (bitcode, runtime libs, lld) is env-var driven, not baked in.

- **`USE_ROCBLAS=OFF`, `USE_MIOPEN=OFF`**. When ON, TVM routes GEMM/conv ops
  to pre-compiled AMD library kernels that completely bypass our LLVM. With
  both OFF, every kernel TVM emits flows through the linked LLVM (and, when
  the split-codegen path is used, through our custom LLVM after that). This
  is essential for scheduling research.

Build steps (from scratch):
```bash
git clone --branch rocm-5.2.3-gfx906 --single-branch --recursive \
    git@github.com:jbaileyhandle/tvm.git tvm-amd-rocm54-fork
cd tvm-amd-rocm54-fork
mkdir -p build && cd build
cat > config.cmake << 'EOF'
set(USE_ROCM ON)
set(USE_LLVM /opt/rocm-5.2.3/llvm/bin/llvm-config)
set(USE_ROCBLAS OFF)
set(USE_MIOPEN OFF)
EOF
CPATH="" cmake .. -DCMAKE_BUILD_TYPE=Release
CPATH="" make -j$(nproc)
```

**Critical:** `CPATH=""` during build. The system has
`CPATH=/opt/rocm-5.7.1/llvm/include/` set globally, which gives ROCm 5.7.1's
LLVM 17 headers higher priority than any `-I` flag. This silently poisons
the build (LLVM 17 headers paired with LLVM 14 libs at link time → build
failures or runtime segfaults).

## Patches on this branch

Two patches beyond `rocm-5.4-test`'s commit `fe1e8ef43`:

1. **`python/tvm/contrib/rocm.py`** — Changed device-bitcode path lookup to
   read `ROCM_PATH` env var instead of hardcoded `/opt/rocm/`. Lets the
   wrapper script point at a non-default ROCm install.
2. **`python/setup.py`** — Fixed `SameFileError` when the cmake build had
   already copied `.so` files and config dirs into `python/tvm/`.

Both packaged in commit `20b811f1f`.

## Wrapper script (`run_tvm_54.sh`)

Currently at `~/gpu2/apps/pytorch_to_hip_tests/path1_tvm/run_tvm_54.sh`.
Its job: set the eight env vars TVM needs, then `exec` Python from the conda
env.

```bash
export TVM_HOME="${TVM_HOME:-$HOME/gpu2/tvm_repos/tvm-amd-rocm54-fork}"
export ROCM_PATH=/opt/rocm-5.2.3
export HIP_PATH=$ROCM_PATH/hip
export PYTHONPATH=$TVM_HOME/python
export LD_LIBRARY_PATH=$TVM_HOME/build:$ROCM_PATH/llvm/lib/:$ROCM_PATH/lib:$ROCM_PATH/hip/lib
export LIBRARY_PATH=$ROCM_PATH/llvm/lib/
export PATH=$ROCM_PATH/llvm/bin:$ROCM_PATH/bin:$ROCM_PATH/opencl/bin:$PATH
export CPATH=$ROCM_PATH/llvm/include/
exec /home/jbaile/miniconda3/envs/tvm_54/bin/python3 "$@"
```

`TVM_HOME` is the only env-var-overridable one — useful for testing a
different TVM checkout without touching the script. ROCm/Python/etc. are
currently hardcoded.

(Planned cleanup: move the canonical wrapper inside this directory, e.g.
`scripts/run_tvm.sh`, and turn the `path1_tvm/` copy into a one-line
forwarder. Not done yet.)

## Conda environment (`tvm_54`)

```bash
conda create -n tvm_54 python=3.10 -y
conda activate tvm_54
pip install "numpy<2" decorator scipy attrs psutil cloudpickle \
    tornado typing_extensions "setuptools>=65,<70"
```

Lives at `~/miniconda3/envs/tvm_54/`. The wrapper points at this env's
Python by absolute path.

The "_54" in the env name is a vestigial label (from the `rocm-5.4-test`
upstream branch lineage); doesn't reflect what we actually link or run
against. Renaming would require recreating the env.

## Custom LLVM toolchains

Both forks have full toolchains (`clang`, `llc`, `opt`, `llvm-link`,
`ld.lld`, etc.) with `MachineInstrSchedulerConfig` patches. Either reads
`misched.txt` from CWD to select a scheduler variant.

| Fork | Major version | Branch (HEAD) | Notes |
|---|---|---|---|
| `~/gpu2/llvm_repos/llvm-project/` | **17** | `jbailey_handle_import_gpu_on_gpu_features` | Active development branch. Most recent scheduler-telemetry work (`HierarchicalScheduler`, etc.). |
| `~/gpu2/llvm_repos/gpu_on_gpu_llvm-project/` | **15** | `jbaile_tmp_two_pass_debug` | Older snapshot. Same major LLVM as the verified-correct `mevermeulen/rocm-tvm:5.4.2` Docker config; lower compatibility risk for ROCm 5.2.3 device bitcode. |

The LLVM-17 fork was originally suspected of being the corruption source
(see *Why ROCm 5.2.3, not 5.7.1*). Today's split-codegen experiment
exonerated its AMDGPU backend — see *Pipeline: split-codegen* below.

## Pipeline: split-codegen (TVM → IR → Custom LLVM → Run)

The current production pipeline. Decouples "TVM emits LLVM IR" from
"LLVM IR is lowered to AMDGPU ISA," so the IR-emission half uses the
known-good stock LLVM 14, and the lowering half uses a custom LLVM that
runs the experimental scheduler.

```
   ┌──────────────────────────────────────────┐
   │  TVM (linked LLVM 14)                    │
   │  PyTorch → Relay → TIR → CodeGenAMDGPU   │
   │  → LLVM IR (linked w/ ROCm 5.2.3 bitcode)│
   └──────────────┬───────────────────────────┘
                  │ kernel.ll (textual IR)
                  ▼
   ┌──────────────────────────────────────────┐
   │  Custom LLVM (15 or 17)                  │
   │  clang -x ir → AMDGPU backend → .o       │
   │   (MachineInstrSchedulerConfig reads     │
   │    misched.txt here)                     │
   │  ld.lld --no-undefined -shared → HSACO   │
   └──────────────┬───────────────────────────┘
                  │ kernel.hsaco
                  ▼
   ┌──────────────────────────────────────────┐
   │  TVM runtime (libtvm_runtime.so)         │
   │  graph executor → hipModuleLoadData      │
   │  → hipModuleLaunchKernel → gfx906        │
   └──────────────────────────────────────────┘
```

Reachable from Python: the textual IR is stored on the compiled module and
extractable via `lib.get_lib().imported_modules[0].get_source("llvm")`. The
host-side substitution swaps just the HSACO bytes (via overriding
`tvm_callback_rocm_link`) so the graph executor stays in control of
kernel-launch orchestration.

The hook point in TVM source: `src/target/llvm/codegen_amdgpu.cc` —
`BuildAMDGPU` calls `module->print(dest_ll, ...)` (line 290) to capture the
textual IR, and `addPassesToEmitFile(..., CGFT_ObjectFile)` (lines 298-310)
is where the AMDGPU backend (including the scheduler) runs in-process. The
split-codegen approach replaces that in-process call with a subprocess
invocation of custom clang on the dumped `.ll`.

Working harness lives at:
`~/gpu2/apps/pytorch_to_hip_tests/path1_tvm/backend_experiment/`
(see that directory's `README.md` for reproduce commands).

## Verified results

### Baseline (TVM internal LLVM 14, no custom LLVM in loop)
`~/gpu2/apps/pytorch_to_hip_tests/path1_tvm/baseline_profile/`, May 11 2026:

| Kernel | Calls | Avg ns | Notes |
|---|---|---|---|
| `tvmgen_default_fused_add_kernel0` | 11 | 4698 | 512² opt_level=0 |
| `tvmgen_default_fused_nn_relu_kernel0` | 11 | 3767 | 512² opt_level=0 |

Correctness PASS. Numbers within a few percent of the original v0.14.0
native build's Mar 12 numbers — confirms the new TVM build + harness are
equivalent on the small known-good case.

### Earlier native test (Mar 31 2026)
| Test | Size | opt_level | Result |
|---|---|---|---|
| relu(x+1) | 1024² | 3 | PASS (max_err=0.00) |
| matmul (dense) | 1024² | 3 | PASS (max_err≈2e-4) |

### Split-codegen — three-way LLVM comparison (May 11 2026)
TVM (internal LLVM 14) emits `.ll`, each LLVM toolchain compiles it:

| Variant | LLVM | Small (512² opt=0) | Large (2048² opt=3) |
|---|---|---|---|
| stock_14 | 14 (ROCm 5.2.3) | PASS | PASS |
| gpu_on_gpu_15 | 15 (custom fork) | PASS | PASS |
| custom_17 | 17 (custom fork) | PASS | **PASS** |

**All three correct at the workload size that previously corrupted in the
v0.14.0 native build.** LLVM 17's AMDGPU backend is not the corruption
source for this workload. The previous corruption (v0.14.0 native, 5.7
Docker) must have been in TVM-side codegen and/or ROCm 5.7.1 bitcode — both
out of the loop in the current split-codegen pipeline.

## Why ROCm 5.2.3, not 5.7.1

Historically the project tried building TVM directly against ROCm 5.7.1
(LLVM 17). That hit two problems:
1. The `rocm-5.4-test`-vintage TVM source can't compile against LLVM 17
   (removed `PassManagerBuilder.h`, moved `Triple.h`, changed
   `AnalysisManager` / `CodeGenFileType` APIs).
2. Even after porting fixes (the `mevermeulen/rocm-tvm:5.7` Docker image
   does this), kernels corrupted at ≥1024² with opt_level=3
   (non-deterministic zero rows, garbage values, plus a "skip Verify()"
   patch in TVM's `codegen_amdgpu.cc` that's a smoking gun for invalid IR
   being silently accepted).

ROCm 5.2.3 (LLVM 14) is the newest installed LLVM the source compiles
against cleanly. The 5.7.1 *kernel driver* is in use (backward compatible
with 5.2.3 userspace), so GPU execution works.

Decision: do **not** switch the userspace to 5.7.1. Reasons:
- ROCm 5.7.1 device bitcode was emitted by LLVM 17; TVM's internal LLVM 14
  likely can't read it (bitcode format is not strongly forward-compatible
  across major versions). Would fail at `relay.build()` time.
- HIP/HSA runtime ABI compat across 5.2.3 → 5.7.1 is a known unknown.
- The current 5.2.3 setup works for all known correctness tests and is the
  reference point for the bisect that ID'd LLVM 17 (specifically: the
  earlier-believed correlation). No concrete need for 5.7.1.

## Important notes for scheduling research

- **Hold the TVM-level schedule constant.** Use `opt_level=3` with fallback
  schedules (the default for TVM). Don't run AutoTVM / MetaSchedule — it
  picks different TVM-level schedules per scheduler variant, confounding
  the LLVM-side comparison.

- **`MachineInstrSchedulerConfig::GetConfig()` is a singleton** — reads
  `misched.txt` once per process. Each scheduler variant requires a
  separate clang/llc subprocess invocation. The split-codegen pipeline
  spawns one clang per build, so this works naturally — each `make build`
  with a different `misched.txt` gets fresh scheduler state.

- **Where `misched.txt` is read from**: the CWD of the clang subprocess.
  In the split-codegen flow, drop `misched.txt` next to the `.ll` file
  before invoking custom clang on it.

- **`USE_ROCBLAS` / `USE_MIOPEN` must stay OFF**. Repeat from build
  config because it's easy to forget: with these ON, the kernels you care
  about route to pre-compiled AMD library code and your scheduler never
  runs on them.

## Pointers to related docs

- `~/gpu2/apps/pytorch_to_hip_tests/path1_tvm/STATUS.md` — project status
  with active configuration, baseline numbers, split-codegen strategy,
  open decisions.
- `~/gpu2/apps/pytorch_to_hip_tests/investigation_guide.md` — historical
  investigation across all three explored paths (TVM / Torch-MLIR+IREE /
  AITemplate), Docker-vs-native bisect details.
- `~/gpu2/apps/pytorch_to_hip_tests/path1_tvm/backend_experiment/README.md`
  — split-codegen experiment specifics: scripts, methodology, results.
- `~/gpu2/apps/gpu2_benchmarks/BENCHMARK_SETUP_STANDARDS.md` — conventions
  for the future `gpu2_benchmarks/TVM/` integration.
