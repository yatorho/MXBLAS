import math
import multiprocessing as mp

import shapes_def as sd
import torch
from shapes_def import SP, Method, method_to_kernel_tag

from mxblas.utils import (
    GPU_bench,
    block_dequant,
    block_quant,
    calc_diff,
    check_nan_inf,
    guassion_with_channel_outlier,
)


def gen_problem(M, N, K, SM, SN, SK):
    a_bf16 = guassion_with_channel_outlier((M, K), dim=0, outlier_ratio=0.003, o_std=10)
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16).cuda()

    ref_out = torch.mm(a_bf16, b_bf16.t())
    check_nan_inf(ref_out)

    a_fp8, a_scale = block_quant(a_bf16, (SM, SK))
    b_fp8, b_scale = block_quant(b_bf16, (SN, SK))
    # a_scale_t = a_scale.T.contiguous().transpose(0, 1)

    return a_fp8, b_fp8, a_scale, b_scale, ref_out


def time_tflops_bandwidth(func, M, N, K, kernel_tag):
    time = GPU_bench(func, iters=repeats, warmup=warmup, kernel_name=kernel_tag)
    tflops = 2 * M * N * K / (time * 1e9)
    bandwidth = (M * K + K * N + M * N * 2) / (time * 1e6)

    return time, tflops, bandwidth


def dequant_or_not(result, quant_output, quant_size):
    if quant_output:
        assert isinstance(result, tuple)
        fp8, scale = result
        result = block_dequant(fp8, scale, (1, quant_size), torch.bfloat16)
    assert isinstance(result, torch.Tensor)
    return result


def bench_kernel(
    method,
    sp,
    ref_out,
    shape,
    *args,
    **kwargs,
):
    import kernels

    if method == Method.TE_CUBLAS:
        diff_rate, time, tflops, bandwidth = kernels.te_gemm_bench(sp, *shape)
    elif method == Method.CUTLASS:
        diff_rate, time, tflops, bandwidth = kernels.cutlass_gemm_bench(sp, *shape)
    else:
        if method == Method.COAT:

            def func():
                return kernels.coat_gemm_nt(sp, *args, **kwargs)

        elif method == Method.SG_LANG:

            def func():
                return kernels.sgl_gemm_nt(sp, *args, **kwargs)

        elif method == Method.DEEP_GEMM:

            def func():
                result = kernels.deepgemm_gemm_nt(sp, *args, **kwargs)
                return result

        elif method == Method.FB_GEMM:

            def func():
                result = kernels.fbgemm_gemm_nt(sp, *args, **kwargs)
                return result

        else:
            raise ValueError(f"Unknown method: {method}")

        diff_rate = (
            calc_diff(dequant_or_not(func(), args[2], args[-1]), ref_out) * 100.0
        )
        time, tflops, bandwidth = time_tflops_bandwidth(
            func, shape[0], shape[1], shape[2], method_to_kernel_tag(method)
        )

    return diff_rate, time, tflops, bandwidth


def baseline_worker(
    model, M, N, K, SM, SN, SK, quant_output, quant_size, sp, method, output_file
):
    prefix = f"{model},{M},{N},{K},{SM},{SN},{SK},{quant_output},{quant_size},{sp},"

    ### Generate the problem
    a_fp8, b_fp8, a_scale, b_scale, ref_out = gen_problem(M, N, K, SM, SN, SK)
    a_scale_t = a_scale.T.contiguous().transpose(0, 1)
    ### ==========

    try:
        diff_rate, time, tflops, bandwidth = bench_kernel(
            method,
            sp,
            ref_out,
            (M, N, K),
            a_fp8,
            b_fp8,
            quant_output,
            a_scale if sp != SP.GB else a_scale_t,
            b_scale,
            quant_size,
        )
    except Exception as e:
        diff_rate, time, tflops, bandwidth = math.nan, math.nan, math.nan, math.nan
        print(f"Error in {method}: {e}")
        import traceback

        traceback.print_exc()

    print(
        f"        {method}: diff_rate: {diff_rate:.4f}%, time: {time:.4f}ms, TFLOPS: {tflops:.2f}, Bandwidth: {bandwidth:.2f}GB/s"
    )
    with open(output_file, "a") as f:
        f.write(f"{prefix}{method},{time:.4f},{tflops:.2f},{bandwidth:.2f}\n")


def mx_worker(
    model, M, N, K, SM, SN, SK, quant_output, quant_size, sp, method, output_file
):
    import kernels

    prefix = f"{model},{M},{N},{K},{SM},{SN},{SK},{quant_output},{quant_size},{sp},"

    ### Generate the problem
    a_fp8, b_fp8, a_scale, b_scale, ref_out = gen_problem(M, N, K, SM, SN, SK)
    a_scale_t = a_scale.T.contiguous().transpose(0, 1)
    ### ==========

    ### MXBLAS:
    def func():
        result = kernels.mx_gemm_nt(
            sp,
            a_fp8,
            b_fp8,
            quant_output,
            a_scale if sp != SP.GB else a_scale_t,
            b_scale,
            quant_size,
        )
        return result

    try:
        diff_rate = (
            calc_diff(dequant_or_not(func(), quant_output, quant_size), ref_out) * 100.0
        )
        time, tflops, bandwidth = time_tflops_bandwidth(
            func, M, N, K, method_to_kernel_tag(Method.MXBLAS)
        )
    except Exception as e:
        diff_rate, time, tflops, bandwidth = math.nan, math.nan, math.nan, math.nan
        print(f"Error in MXBLAS: {e}")
        import traceback

        traceback.print_exc()

    print(
        f"        MXBLAS: diff_rate: {diff_rate:.4f}%, time: {time:.4f}ms, TFLOPS: {tflops:.2f}, Bandwidth: {bandwidth:.2f}GB/s"
    )
    with open(output_file, "a") as f:
        f.write(f"{prefix}MXBLAS,{time:.4f},{tflops:.2f},{bandwidth:.2f}\n")
    ### ==========


warmup = 8
repeats = 8

csv_head = "Model,M,N,K,SM,SN,SK,Q,QN,SP,Method,Time(ms),TFLOPS,Bandwidth(GB/s)\n"


# def main()

if __name__ == "__main__":

    def _comma_separated_list(s):
        return s.split(",")

    import argparse

    parser = argparse.ArgumentParser(description="Benchmarking script")

    def parse_models(s):
        models = _comma_separated_list(s)
        if model in models:
            if model not in sd.MODEL_NAMES:
                raise ValueError(f"Model {model} is not supported.")
        return models

    def parse_shapes(s):
        return map(int, _comma_separated_list(s))

    def parse_scaling_patterns(s):
        sp_str_list = _comma_separated_list(s)
        return [SP(sp_str) for sp_str in sp_str_list]

    # models = ["llama", "gpt3_6.7B"]
    models = [
        "llama-3-70B",
        "gpt-neox-20B",
        "palm-8B",
        "palm-62B",
        "t5-3B",
        "flan-t5-small",
        "flan-t5-xl",
        "bert-large",
        "roberta-base",
        "electra-base",
        "albert-large",
        "xlnet-large",
        "bart-base",
        "bart-large",
        "deberta-v2-xxlarge",
        "bloom-7.1B",
        "opt-30B",
        "opt-66B",
        "chatglm3-6B",
        "qwen-72B",
        "mistral-7B",
        "mixtral-8x7B",
        "gemini-1.5-pro",
        "claude-3-opus",
        "yi-6B",
        "yi-34B",
        "deepseek-7B",
        "deepseek-67B",
        "phi-3",
        "stablelm-7B",
        "stablelm-12B",
        "falcon-180B",
        "mpt-7B",
        "codegen2-7B",
        "codegen2-16B",
        "wizardcoder-15B",
        "wizardlm-13B",
        "openchat-3.5",
        "zephyr-7B",
        "baichuan-13B",
        "aquila-34B",
        "xverse-65B",
        "skywork-53B",
        "orion-70B",
        "minimax-13B",
        "deepseek-coder-33B-instruct",
    ]
    Ms = [128, 1024, 4096, 8192, 16384]
    scaling_patterns = [SP.TT, SP.GB, SP.BB, SP.CC]
    quant_size = 16

    parser.add_argument(
        "--models",
        type=parse_models,
        default=models,
    )
    parser.add_argument(
        "--Ms",
        type=parse_shapes,
        default=Ms,
    )
    parser.add_argument(
        "--scaling_pattern",
        type=parse_scaling_patterns,
        default=scaling_patterns,
    )
    parser.add_argument(
        "--quant_output",
        action="store_true",
    )
    parser.add_argument(
        "--quant_size",
        type=int,
        default=quant_size,
    )
    parser.add_argument("--output_file", type=str, default="bench_all.csv")

    args = parser.parse_args()

    models = args.models
    Ms = args.Ms
    scaling_patterns = args.scaling_pattern
    quant_output = args.quant_output
    quant_size = args.quant_size
    output_file = args.output_file

    mp.set_start_method("spawn")
    _ = torch.randn(1).cuda()  # for occupy GPU

    with open(output_file, "a") as f:
        f.write(csv_head)

    print(f"Running all models: {models}")
    for model in models:
        for shape in sd.gemm_shapes(Ms)[model]:
            for sp in scaling_patterns:
                M, N, K, SM, SN, SK = sd.gen_problem_size(*shape, sp)
                methods = sd.sp_to_method(sp)

                print(
                    f"Running {model=}, {M=}, {N=}, {K=}, {SM=}, {SN=}, {SK=}, Q={quant_output}, QN={quant_size}, sp={sp}, Method={methods}"
                )
                prefix = f"{model},{M},{N},{K},{SM},{SN},{SK},{quant_output},{quant_size},{sp},"

                ### MXBLAS:
                p = mp.Process(
                    target=mx_worker,
                    args=(
                        model,
                        M,
                        N,
                        K,
                        SM,
                        SN,
                        SK,
                        quant_output,
                        quant_size,
                        sp,
                        Method.MXBLAS,
                        output_file,
                    ),
                )
                p.start()
                p.join()
                if p.exitcode != 0:
                    (
                        diff_rate,
                        time,
                        tflops,
                        bandwidth,
                    ) = (math.nan, math.nan, math.nan, math.nan)
                    print(
                        f"        MXBLAS(ERROR): diff_rate: {diff_rate:.4f}%, time: {time:.4f}ms, TFLOPS: {tflops:.2f}, Bandwidth: {bandwidth:.2f}GB/s"
                    )
                    with open(output_file, "a") as f:
                        f.write(
                            f"{prefix}MXBLAS,{time:.4f},{tflops:.2f},{bandwidth:.2f}\n"
                        )
                ### ==========

                ### Baseline:
                for method in methods:
                    p = mp.Process(
                        target=baseline_worker,
                        args=(
                            model,
                            M,
                            N,
                            K,
                            SM,
                            SN,
                            SK,
                            quant_output,
                            quant_size,
                            sp,
                            method,
                            output_file,
                        ),
                    )
                    p.start()
                    p.join()
                    if p.exitcode != 0:
                        (
                            diff_rate,
                            time,
                            tflops,
                            bandwidth,
                        ) = (math.nan, math.nan, math.nan, math.nan)
                        print(
                            f"        {method}(ERROR): diff_rate: {diff_rate:.4f}%, time: {time:.4f}ms, TFLOPS: {tflops:.2f}, Bandwidth: {bandwidth:.2f}GB/s"
                        )
                        with open(output_file, "a") as f:
                            f.write(
                                f"{prefix}{method},{time:.4f},{tflops:.2f},{bandwidth:.2f}\n"
                            )
                    ### ==========
