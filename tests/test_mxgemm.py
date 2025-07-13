import argparse

import torch

import mxblas
import mxblas.utils as utils


def check_is_ef(m, n, k, sm, sn, sk):
    bm_max = 128
    bn_max = 256

    is_ef = True
    is_ef &= k == sk
    is_ef &= sm >= bm_max and sm % bm_max == 0
    is_ef &= sn >= bn_max and sn % bn_max == 0

    return is_ef


def run(M, N, K, SM, SN, SK, quant=False, QN=16, repeats=8, warmup=8):
    if not check_is_ef(M, N, K, SM, SN, SK):
        raise ValueError(
            f"Invalid EF configuration: M={M}, N={N}, K={K}, SM={SM}, SN={SN}, SK={SK}"
        )

    a_bf16 = utils.gaussian_with_channel_outlier(
        (M, K), dim=0, outlier_ratio=0.003, o_std=10
    )
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16).cuda()
    ref_out = torch.mm(a_bf16, b_bf16.t())
    utils.check_nan_inf(ref_out)

    a_fp8, a_scales = utils.block_quant(a_bf16, (SM, SK))
    b_fp8, b_scales = utils.block_quant(b_bf16, (SN, SK))

    def kernel():
        result = mxblas.mx_gemm_kernel(
            a_fp8, b_fp8.T, a_scales, b_scales.T, quant, torch.Size((1, QN))
        )

        if not quant:
            result = result[0]
        return result

    result = kernel()
    out_bf16 = (
        result
        if not quant
        else utils.block_dequant(result[0], result[1], (1, QN), torch.bfloat16)
    )

    print(f"difference rate: {utils.calc_diff(out_bf16, ref_out) * 100.:.4f}%")

    time = utils.GPU_bench(kernel, iters=repeats, warmup=warmup)
    tflops = 2 * M * N * K / (time * 1e9)
    print(f"TFLOPS: {tflops:.4f} | Time: {time:.4f} seconds")


if __name__ == "__main__":
    mxblas.register_all_kernels()

    parser = argparse.ArgumentParser(description="Run MXBLAS EF test")
    parser.add_argument("--M", type=int, default=8192, help="Matrix M dimension")
    parser.add_argument("--N", type=int, default=8192, help="Matrix N dimension")
    parser.add_argument("--K", type=int, default=8192, help="Matrix K dimension")
    parser.add_argument("--SM", type=int, default=8192, help="SM dimension")
    parser.add_argument("--SN", type=int, default=8192, help="SN dimension")
    parser.add_argument("--SK", type=int, default=8192, help="SK dimension")
    parser.add_argument("--quant", action="store_true", help="Enable quantization")
    parser.add_argument("--QN", type=int, default=16, help="Quantization size")
    parser.add_argument(
        "--repeats", type=int, default=8, help="Number of repeats for benchmarking"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Number of warmup iterations for benchmarking",
    )

    args = parser.parse_args()
    run(
        M=args.M,
        N=args.N,
        K=args.K,
        SM=args.SM,
        SN=args.SN,
        SK=args.SK,
        quant=args.quant,
        QN=args.QN,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    print("MXBLAS test completed successfully.")
