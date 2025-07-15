# 🚀 MXBLAS: Accelerating 8-bit Deep Learning with a Unified Micro-Scaled GEMM Library

## 🌟 Overview

**MXBLAS** is a high-performance library for **Micro-scaled General Matrix Multiplication (MX-GEMM)**.  
It leverages 8-bit micro-scaling formats (**MX-Format**) to accelerate deep learning workloads on modern GPUs.

The MX-Format supports diverse scaling patterns and granularities, and MXBLAS efficiently handles all MX-Format variations within a unified framework.

## ✨ Highlights

✅ Unified MX-Format support under a single framework  
✅ Adaptive runtime kernel generation and auto-tuning  
✅ Compute–store co-optimization to minimize overhead  
✅ High performance on **NVIDIA Hopper GPUs**

---

## 🧪 Usage


### 🔗 Unified MX-GEMM API

The `mxblas.mx_gemm_kernel` function from MXBLAS provides a unified and flexible API for performing MX-GEMM operations on tensors in MX-Formats, supporting various scaling patterns and output quantization options.

Below is a basic usage example for **Per-Tensor x Per-Tensor (TT) scaled input** and **16-sized group quantization output**:

```python
import torch
import mxblas

M, N, K = 8192, 8192, 8192
SM, SN, SK = 8192, 8192, 8192  # Per-Tensor Scaling Pattern
out_quant = True  # Enable quantization in the output
QM, QN = 1, 16  # Group-wise quantization: 1 row, 16 columns per group

# Generate random input tensors in FP8 format (E4M3FN)
left = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
right = torch.randn(N, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)

# Prepare scale tensors for both operands
left_scales = torch.randn((M // SM, K // SK), dtype=torch.bfloat16, device='cuda')
right_scales = torch.randn((N // SN, K // SK), dtype=torch.bfloat16, device='cuda')

# Register all kernel templates before use (only needs to be called once)
mxblas.register_all_kernels()  # Register all templates

# Perform MX-GEMM
out_value, out_scales = mxblas.mx_gemm_kernel(
    left,
    right.T, # Transpose right for correct shape
    left_scales,
    right_scales.T, # Transpose scales to match inputs
    out_quant,
    torch.Size([QM, QN])
)
```

---

#### 📖 API Details

```python
def mx_gemm_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
    output_quant: bool = False,
    quant_size: Optional[torch.Size] = None,
    out_dtype: Optional[torch.dtype] = None,
    out_transpose: bool = False,
    out_scale_transpose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**Arguments:**

- `left`, `right`: Input tensors in FP8 (e.g., `torch.float8_e4m3fn`), representing the left and right operands for MX-GEMM.
- `left_scale`, `right_scale`: Scale tensors (FP16/FP32) associated with each operand. 
- `output_quant`: If `True`, quantize the output tensor. 
- `quant_size`: (Optional) Quantization group size for the output. (e.g. `torch.Size([1, 32])`). 
- `out_dtype`: (Optional) Desired output tensor dtype. Defaults to left's dtype if `output_quant=True`, otherwise `torch.bfloat16`. 
- `out_transpose`, `out_scale_transpose`: Whether to transpose the output tensor and/or the output scale tensor. 

**Returns:** 

- A tuple `(output, output_scale)`: 
    - `output`: The result tensor. 
    - `output_scale`: Corresponding scale tensor. If `output_quant=False`, the scale tensor is empty.

You can adjust quantization patterns, scaling strategies, and output data types according to your application needs, making `mx_gemm_kernel` a powerful drop-in solution for high-performance, quantization-aware GEMM on GPUs.

---

### 🐞 Debugging & Profiling Flags

Users can control the library’s debugging and profiling behavior via environment variables defined below:

| Variable Name                 | Description                                                                                  |
|-------------------------------|----------------------------------------------------------------------------------------------|
| `MXBLAS_JIT_DEBUG`            | Enables JIT debugging mode and prints detailed information about the JIT compilation process. |
| `MXBLAS_NVCC_COMPILE`         | Path to the NVCC compiler. Defaults to the system's NVCC if unset.                           |
| `MXBLAS_CACHE_DIR`            | Directory where JIT-compiled kernels are cached. Defaults to `~/.mxblas/`.                  |
| `MXBLAS_PTXAS_VERBOSE`        | Enables verbose mode for PTXAS during compilation, printing PTX assembly details.          |
| `MXBLAS_JIT_PRINT_NVCC_COMMAND` | Prints the NVCC command used during JIT compilation.                                         |
| `MXBLAS_PRINT_AUTO_TUNE`      | Prints details about the auto-tuning process, including kernel profiling and selection.    |
| `MXBLAS_BUILD_FAILURE_INFO_PATH` | Specifies a file path to log build failures for debugging.                                  |
| `MXBLAS_PRINT_MATCHING`       | Prints details about multi-template matching during JIT compilation.                       |

You can set these environment variables separately or in combination when running your script:

```bash
MXBLAS_JIT_DEBUG=1 python your_script.py
MXBLAS_PRINT_AUTO_TUNE=1 MXBLAS_BUILD_FAILURE_INFO_PATH=log.txt python your_script.py
```

---

## 🛠️ Installation

### 📋 Hardware Requirements

- **GPU:** NVIDIA Hopper architecture (Compute Capability = 9.0)  
  Required hardware features: Tensor Memory Accelerator (TMA), Warp-Group Matrix Multiply Accumulate (WGMMA), and Memory Barrier (MBarrier) instructions.

---

### 📋 Software Requirements

- GCC/G++ ≥ 11
- CUDA ≥ 12.1
- Python ≥ 3.12
- PyTorch ≥ 2.1 (with Hopper support)

### 📦 Setting Up the Environment

#### 1️⃣ Create Python environment:

```bash
conda create -n mxblas python=3.12
conda activate mxblas
```

#### 2️⃣ Install dependencies:

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 3️⃣ Clone the MXBLAS repository:

```bash
git clone https://github.com/yatorho/MXBLAS.git
cd MXBLAS
```

#### 4️⃣ Install MXBLAS:

```bash
pip install -e .
```

#### 5️⃣ (Optional) Run tests

This project provides two simple test scripts to validate functionality and performance.

**i). Run the JIT test**

```bash
python tests/test_jit.py
```

If everything works as expected, you should see output similar to:

```bash
Building ...
Running ...
Hello, MXBLAS!
JIT test passed
```

**ii). Run the MX-GEMM performance/correctness test**

```bash
python tests/test_mxgemm.py
```

Expected output:

```bash
difference rate: 0.0715%
TFLOPS: 987.2140 | Time: 1.1138 seconds
MXBLAS test completed successfully.
```

You can also pass different flags to `test_mxgemm.py` to control the matrix dimensions and scaling pattern. For example:

```bash
python tests/test_mxgemm.py -m=8192 -n=8192 -k=8192 -sm=1 -sn=1 -sk=8192 -quant -qn=16
```

This runs a Per-Channel Scaling Pattern test with:

- `-m`, `-n`, `-k`: specify the dimensions of the matrices.
- `-sm`, `-sn`, `-sk`: specify the scaling granularities along the M, N, and K dimensions.
- `-quant`: enables quantization in the output.
- `-qn`: specifies the group size for quantization.

Feel free to adjust these parameters to experiment with different configurations and observe their impact on performance and accuracy.


---

## 🔄 AE Reproduction

This section details the steps to reproduce the main results (Figure 9) from the MXBLAS paper:  
**"MXBLAS: Accelerating 8-bit Deep Learning with a Unified Micro-Scaled GEMM Library"**.


### 📥 Install Required Packages

```bash
pip install fbgemm_gpu==1.0.0 --index-url https://download.pytorch.org/whl/cu124
pip install triton==3.1.0  # Triton backend support
pip install sgl_kernel==0.0.5  # SGL kernel support
pip install --no-build-isolation transformer_engine[pytorch]==1.13.0  # Transformer Engine support
pip install einops==0.8.1  # Transformer Engine dependency
```

---

### 🪄 Submodule Preparation

#### Set Environment Variables

Export the MXBLAS root directory:

```bash
export MXBLAS_ROOT=$(pwd)
```

#### Clone and Checkout Submodules

Please clone the following third-party repositories and check out each repository to its corresponding commit hash.

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

**1). DeepGEMM**

```bash
cd $MXBLAS_ROOT/third_party/DeepGEMM
python setup.py develop
```

**2). CUTLASS**

```bash
cd $MXBLAS_ROOT/bench
./make_cutlass.sh
```

**3). COAT**

```bash
cd $MXBLAS_ROOT/third_party/coat
pip install -e .
```

### 🚀 Run Evaluation

You can run all benchmarks by executing the following commands:

```bash
cd $MXBLAS_ROOT/bench
python bench_all.py
```

The script also supports customizing the benchmark configuration via optional arguments. For example:

```bash
python bench_all.py --models=llama-3-70B,opt-66B --Ms=2048,4096,8192 --scaling_pattern=TT,BB
```

where the arguments mean:

- `--models`: a comma-separated list of model names to benchmark, e.g. `llama-3-70B,opt-66B`.
- `--Ms`: a comma-separated list of M dimensions to benchmark, e.g. `1024,2048,4096,8192`.
- `--scaling_pattern`: a comma-separated list of scaling patterns in the format `TT`, `BB`, `GB`, `CC`.

If no arguments are specified, the default configuration used in Figure 9 of the paper will be applied.


Benchmark results will be appended to the file `bench/bench_all.csv`.

Note: The benchmarking process can take **several hours**, depending on your hardware configuration. Please be patient.

---

## 📜 License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## 📧 Contact

For questions, bug reports, or contributions, please open an issue or contact the [authors](weihuwang@whu.edu.cn).

---
