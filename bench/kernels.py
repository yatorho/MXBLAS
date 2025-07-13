from math import nan

import torch
from shapes_def import SP as SP
from shapes_def import Method, method_to_kernel_tag

from mxblas import mx_gemm_kernel, register_all_kernels
from mxblas.utils import GPU_bench

register_all_kernels()


def mx_gemm_nt(
    sp: SP,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    quant_output: bool,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    quant_size: int = 16,
):

    result = mx_gemm_kernel(
        lhs,
        rhs.T,
        scale_a,
        scale_b.T,
        quant_output,
        torch.Size((1, quant_size)),
    )

    return result[0] if not quant_output else result


def coat_gemm_nt(
    sp: SP,
    a_fp8,
    b_fp8,
    quant_output,
    a_scale,
    b_scale,
    quant_size,
):
    assert sp == SP.TT

    from coat.activation.real_quantization import fp8_linear_forward  # type: ignore

    return fp8_linear_forward(a_fp8, a_scale, b_fp8, b_scale, quant_output, quant_size)


def sgl_gemm_nt(
    sp: SP,
    a_fp8,
    b_fp8,
    quant_output,
    a_scale,
    b_scale,
    quant_size,
):
    import sgl_kernel

    if sp == SP.CC:
        return sgl_kernel.fp8_scaled_mm(
            a_fp8, b_fp8.T, a_scale, b_scale.T, out_dtype=torch.bfloat16
        )
    elif sp == SP.GB:
        return sgl_kernel.fp8_blockwise_scaled_mm(
            a_fp8, b_fp8.T, a_scale, b_scale.T, out_dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"Unknown scaling pattern: {sp}")


def deepgemm_gemm_nt(
    sp: SP,
    a_fp8,
    b_fp8,
    quant_output,
    a_scale,
    b_scale,
    quant_size,
):
    import deep_gemm

    assert sp == SP.GB

    result = torch.empty(
        (a_fp8.size(0), b_fp8.size(0)), dtype=torch.bfloat16, device="cuda"
    )
    deep_gemm.gemm_fp8_fp8_bf16_nt((a_fp8, a_scale), (b_fp8, b_scale), result)
    return result


def fbgemm_gemm_nt(
    sp: SP,
    a_fp8,
    b_fp8,
    quant_output,
    a_scale,
    b_scale,
    quant_size,
):
    import fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm as fb_fp8

    if sp == SP.BB:
        q_m = a_fp8.size(0) // a_scale.size(0)
        q_n = b_fp8.size(0) // b_scale.size(0)
        q_k = b_fp8.size(1) // b_scale.size(1)
        return fb_fp8.matmul_fp8_block(
            a_fp8, b_fp8, a_scale, b_scale, q_m, q_n, q_k, fp8_fast_accum=True
        )
    elif sp == SP.CC:
        return fb_fp8.matmul_fp8_row(
            a_fp8, b_fp8, a_scale, b_scale, tma_persistent=False
        )
    else:
        raise ValueError(f"Unknown scaling pattern: {sp}")


def cutlass_gemm_bench(sp: SP, m, n, k):
    if sp == SP.TT:
        exe_file = "./bin/cutlass_fp8_gemm"
    elif sp == SP.BB:
        exe_file = "./bin/cutlass_fp8_block_wise_gemm"
    elif sp == SP.GB:
        exe_file = "./bin/cutlass_fp8_row_block_gemm"
    else:
        raise ValueError(f"Unknown scaling pattern: {sp}")

    import subprocess

    cmds = f"{exe_file} --m={m} --n={n} --k={k} --device_scale=true --iterations=16 --save_amax=false"

    """ Output format:
    Disposition: Skipped
    Problem Size: 8192x8192x8192x1
    Rasterization: Heuristic with a maximum CTA swizzle of 16
    Avg runtime: 1.88009 ms
    GFLOPS: 584819
    """

    result = subprocess.run(cmds, shell=True, capture_output=True, text=True)

    diff_rate = nan
    time = nan
    tflops = nan
    bandwidth = nan
    for line in result.stdout.splitlines():
        if "Avg runtime" in line:
            time = float(line.split(": ")[1].split(" ")[0])
        if "GFLOPS" in line:
            gflops = float(line.split(": ")[1].split(" ")[0])
            tflops = gflops / 1000

    if time != nan:
        bandwidth = (m * k + k * n + m * n * 2) / (time * 1e6)

    return diff_rate, time, tflops, bandwidth


def te_gemm_bench(sp: SP, m, n, k):
    assert sp == SP.TT

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
    )

    my_linear = te.Linear(k, n, bias=True).to(torch.bfloat16).cuda()
    inp = torch.rand((m, k), dtype=torch.bfloat16).cuda()

    def func():
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out_fp8: torch.Tensor = my_linear(inp)

    try:
        time = GPU_bench(
            func, iters=8, warmup=9, kernel_name=method_to_kernel_tag(Method.TE_CUBLAS)
        )
        tflops = 2 * m * n * k / (time * 1e9)
        bandwidth = (m * k + k * n + m * n * 2) / (time * 1e6)
    except Exception as e:
        time = nan
        tflops = nan
        bandwidth = nan

    return nan, time, tflops, bandwidth
