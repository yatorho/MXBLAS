# MXBLAS

## Overview

**MXBLAS** is a high-performance library for **Micro-scaled General Matrix Multiplication (MX-GEMM)**.  
It leverages 8-bit micro-scaling formats (**MX-Format**) to accelerate deep learning workloads on modern GPUs.

The MX-Format supports diverse scaling patterns and granularities, and MXBLAS efficiently handles all MX-Format variations within a unified framework.

## Highlights

✅ Unified MX-Format support under a single framework  
✅ Adaptive runtime kernel generation and auto-tuning  
✅ Compute–store co-optimization to minimize overhead  
✅ High performance on **NVIDIA Hopper GPUs**

---

## Usage


### Unified MX-GEMM API

```python
import torch
import mxblas

M, N, K = 8192, 8192, 8192
SM, SN, SK = 8192, 8192, 8192  # Per-Tensor Scaling Pattern
out_quant = True  # Enable quantization in the output
QM, QN = 1, 16

left = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
right = torch.randn(N, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e5m2fn)
left_scales = torch.randn((M // SM, K // SK), dtype=torch.bfloat16, device='cuda')
right_scales = torch.randn((N // SN, K // SK), dtype=torch.bfloat16, device='cuda')

out_value, out_scales = mxblas.mx_gemm_kernel(
    left,
    right.T,
    left_scales,
    right_scales.T,
    out_quant,
    torch.Size([QM, QN])
)
```



### Debugging & Profiling Flags

Users can control the library’s debugging and profiling behavior via environment variables defined below:

| Variable Name                 | Description                                                                                  |
|-------------------------------|----------------------------------------------------------------------------------------------|
| `MXBLAS_JIT_DEBUG`            | Enables JIT debugging mode and prints detailed information about the JIT compilation process. |
| `MXBLAS_NVCC_COMPILE`         | Path to the NVCC compiler. Defaults to the system’s NVCC if unset.                           |
| `MXBLAS_CACHE_DIR`            | Directory where JIT-compiled kernels are cached. Defaults to `~/.mxblas/`.                  |
| `MXBLAS_PTXAS_VERBOSE`        | Enables verbose mode for PTXAS during compilation, printing PTX assembly details.          |
| `MXBLAS_JIT_PRINT_NVCC_COMMAND` | Prints the NVCC command used during JIT compilation.                                         |
| `MXBLAS_PRINT_AUTO_TUNE`      | Prints details about the auto-tuning process, including kernel profiling and selection.    |
| `MXBLAS_BUILD_FAILURE_INFO_PATH` | Specifies a file path to log build failures for debugging.                                  |
| `MXBLAS_PRINT_MATCHING`       | Prints details about multi-template matching during JIT compilation.                       |

You can set these environment variables when running your script:
```bash
MXBLAS_JIT_DEBUG=1 python your_script.py
MXBLAS_PRINT_AUTO_TUNE=1 python your_script.py
```

---

## Installation

### Hardware Requirements

- **GPU:** NVIDIA Hopper architecture (Compute Capability = 9.0)  
  Required hardware features: Tensor Memory Accelerator (TMA), Warp-Group Matrix Multiply Accumulate (WGMMA), and Memory Barrier (MBarrier) instructions.

### Software Requirements

- GCC/G++ ≥ 11
- CUDA ≥ 12.1
- Python ≥ 3.12
- PyTorch ≥ 2.1 (with Hopper support)

### Setting Up the Environment

1. Create a Python environment:
```bash
conda create -n mxblas python=3.12
conda activate mxblas
```

2. Install dependencies:
```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. Install MXBLAS in editable mode:
```bash
pip install -e .
```

4. (Optional) Run the test suite:
```bash
python tests/test_jit.py
python tests/test_mxgemm.py
```

---

## AE Reproduction

This section details the steps to reproduce the main results (Figure 9) from the MXBLAS paper:  
**"MXBLAS: Accelerating 8-bit Deep Learning with a Unified Micro-Scaled GEMM Library."**


### Pip Install Required Packages

```bash
pip install fbgemm_gpu==1.0.0 --index-url https://download.pytorch.org/whl/cu124
pip install triton==3.1.0
pip install sgl_kernel==0.0.5
pip install --no-build-isolation transformer_engine[pytorch]==1.13.0
```

### Submodule Preparation

#### Set Environment Variables

Export the MXBLAS root directory:
```bash
export MXBLAS_ROOT=$(pwd)
```

#### Clone and Checkout Submodules

You will need specific versions of the following third-party repositories:

| Project       | Commit Hash                                |
|---------------|--------------------------------------------|
| **DeepGEMM**  | `a6d97a1c1b48a7a9d7994d0e155ee6f11d0a3f07` |
| **CUTLASS**   | `e9627ce55b42fd2599f58cd4396da9380954def0` |
| **COAT**      | `efcd56e223ef3e37eb42a10cff14183fb612e6d0` |

Check out each repository to the corresponding commit hash.
```bash
git submodule update --init --recursive

cd $MXBLAS_ROOT/third_party/DeepGEMM
git checkout a6d97a1c1b48a7a9d7994d0e155ee6f11d0a3f07

cd $MXBLAS_ROOT/bench/cutlass
git checkout e9627ce55b42fd2599f58cd4396da9380954def0

cd $MXBLAS_ROOT/third_party/coat
git checkout efcd56e223ef3e37eb42a10cff14183fb612e6d0
```



#### Build Third-Party Dependencies

1. DeepGEMM
```bash
cd $MXBLAS_ROOT/third_party/DeepGEMM
git submodule init
git submodule update
python setup.py develop
```

2. CUTLASS
```bash
cd $MXBLAS_ROOT/bench
./make_cutlass.sh
```

3. COAT
```bash
cd $MXBLAS_ROOT/third_party/coat
pip install -e .
```

### Run Evaluation

Run all benchmarks:
```bash
cd $MXBLAS_ROOT/bench
python bench_all.py
```

Benchmark results will be saved in `bench/bench_all.csv`.

⚠️ Note: The benchmarking process can take **several hours**, depending on your hardware configuration. Please be patient.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## Contact

For questions, bug reports, or contributions, please open an issue or contact the [authors](weihuwang@whu.edu.cn).

---

