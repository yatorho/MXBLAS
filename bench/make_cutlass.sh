#!/bin/bash
cd $MXBLAS_ROOT/third_party/cutlass

mkdir build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS="90a"

## mkdir dir for binary executables
bin=$MXBLAS_ROOT/bench/bin
mkdir -p $bin

# build fp8 TT gemm kernel
cd examples/54_hopper_fp8_warp_specialized_gemm
make
cp ./54_hopper_fp8_warp_specialized_gemm $bin/cutlass_fp8_gemm

# build fp8 GB/BB gemm kernel
cd ../67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling
make 
cp ./67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling $bin/cutlass_fp8_row_block_gemm
cp ./67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling $bin/cutlass_fp8_block_wise_gemm
