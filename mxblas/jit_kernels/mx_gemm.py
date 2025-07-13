from functools import lru_cache
from itertools import product
from typing import Optional, Tuple, Union, cast

import torch

from mxblas.gemm.descriptor import (
    Layout,
    MatrixLayouts,
    MXGEMMDescriptor,
    OperatorsShape,
    QuantizationShape,
    ScalarDType,
    ScalesShape,
)
from mxblas.gemm.kernel_pool import RuntimeParameter, pool
from mxblas.gemm.keys import CTASwizzleDim, SMemSwizzleBits

supported_fp8_dtypes = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)

supported_fp16_dtypes = (
    torch.float16,
    torch.bfloat16,
)

supported_fp32_dtypes = (torch.float32,)

supported_high_precision_dtypes = supported_fp16_dtypes + supported_fp32_dtypes


torch_dtype_to_scalar_dtype = {
    torch.float8_e4m3fn: ScalarDType.FP8_E4M3,
    torch.float8_e5m2: ScalarDType.FP8_E5M2,
    torch.float16: ScalarDType.FP16,
    torch.bfloat16: ScalarDType.BF16,
    torch.float32: ScalarDType.FP32,
}

scalar_dtype_to_torch_dtype = {
    ScalarDType.FP8_E4M3: torch.float8_e4m3fn,
    ScalarDType.FP8_E5M2: torch.float8_e5m2,
    ScalarDType.FP16: torch.float16,
    ScalarDType.BF16: torch.bfloat16,
    ScalarDType.FP32: torch.float32,
}


def get_layout(tensor_or_contiguous: Union[torch.Tensor, bool]) -> Layout:
    if isinstance(tensor_or_contiguous, torch.Tensor):
        return (
            Layout.ROW_MAJOR
            if tensor_or_contiguous.is_contiguous()
            else Layout.COLUMN_MAJOR
        )
    elif isinstance(tensor_or_contiguous, bool):
        return Layout.ROW_MAJOR if tensor_or_contiguous else Layout.COLUMN_MAJOR
    else:
        raise TypeError(
            f"Expected tensor_or_contiguous to be a torch.Tensor or bool, got {type(tensor_or_contiguous)}."
        )


@lru_cache(maxsize=None)
def get_meta_info(
    left_dtype: torch.dtype,
    right_dtype: torch.dtype,
    left_scale_dtype: torch.dtype,
    right_scale_dtype: torch.dtype,
    output_quant: bool,
    quant_size: Optional[torch.Size],
    out_dtype: Optional[torch.dtype],
    left_size: torch.Size,
    right_size: torch.Size,
    left_scale_size: torch.Size,
    right_scale_size: torch.Size,
    left_contiguous: bool,
    right_contiguous: bool,
    left_scale_contiguous: bool,
    right_scale_contiguous: bool,
    out_transpose: bool,
    out_scale_transpose: bool,
) -> MXGEMMDescriptor:

    ### Dtype checks
    if not (left_dtype == right_dtype and left_dtype in supported_fp8_dtypes):
        raise ValueError(
            f"Unsupported input dtypes {left_dtype} and {right_dtype}. Supported dtypes are {supported_fp8_dtypes}."
        )
    if not (
        left_scale_dtype == right_scale_dtype
        and left_scale_dtype in supported_high_precision_dtypes
    ):
        raise ValueError(
            f"Unsupported scale dtypes {left_scale_dtype} and {right_scale_dtype}. Supported dtypes are {supported_high_precision_dtypes}."
        )

    input_torch_dtype = left_dtype
    scale_torch_dtype = left_scale_dtype

    if output_quant:
        if out_dtype is None:
            out_dtype = input_torch_dtype
        else:
            if out_dtype not in supported_fp8_dtypes:
                raise ValueError(
                    f"Unsupported output dtype {out_dtype}. Supported dtypes are {supported_fp8_dtypes}."
                )
    else:
        if out_dtype is None:
            out_dtype = torch.bfloat16
        else:
            if out_dtype not in supported_fp16_dtypes:
                raise ValueError(
                    f"Unsupported output dtype {out_dtype}. Supported dtypes are {supported_fp16_dtypes}."
                )

    out_torch_dtype = out_dtype

    ### Shape checks
    if len(left_size) != 2 or len(right_size) != 2 or left_size[1] != right_size[0]:
        raise ValueError(
            f"Left tensor shape {left_size} and right tensor shape {right_size} are incompatible for matrix multiplication."
        )
    m, n, k = left_size[0], right_size[1], left_size[1]

    if (
        len(left_scale_size) != 2
        or len(right_scale_size) != 2
        or left_scale_size[1] != right_scale_size[0]
    ):
        raise ValueError(
            f"Left scale shape {left_scale_size} and right scale shape {right_scale_size} are incompatible."
        )
    if (
        (m % left_scale_size[0] != 0)
        or (n % right_scale_size[1] != 0)
        or (k % left_scale_size[1] != 0)
    ):
        raise ValueError(
            f"Left scale shape {left_scale_size} and right scale shape {right_scale_size} do not divide m ({m}) and n ({n})."
        )

    sm, sn, sk = (
        m // left_scale_size[0],
        n // right_scale_size[1],
        k // left_scale_size[1],
    )

    if output_quant:
        if quant_size is None:
            raise ValueError("quant_size must be provided when output_quant is True.")
        if len(quant_size) != 2 or m % quant_size[0] != 0 or n % quant_size[1] != 0:
            raise ValueError(
                f"Invalid quant_size {quant_size}. It must be a tuple of two integers that divide m ({m}) and n ({n})."
            )

    else:
        if quant_size is None:
            quant_size = torch.Size((0, 0))

    qm, qn = quant_size

    ### Layout checks:
    left_layout = get_layout(left_contiguous)
    right_layout = get_layout(right_contiguous)
    left_scale_layout = get_layout(left_scale_contiguous)
    right_scale_layout = get_layout(right_scale_contiguous)
    out_layout = Layout.ROW_MAJOR if not out_transpose else Layout.COLUMN_MAJOR
    out_scale_layout = (
        Layout.ROW_MAJOR if not out_scale_transpose else Layout.COLUMN_MAJOR
    )

    ### Build MXGEMMDescriptor
    input_dtype = torch_dtype_to_scalar_dtype[input_torch_dtype]
    output_dtype = torch_dtype_to_scalar_dtype[out_torch_dtype]
    scales_dtype = torch_dtype_to_scalar_dtype[scale_torch_dtype]
    operators_shape = OperatorsShape(m=m, n=n, k=k)
    scales_shape = ScalesShape(m=sm, n=sn, k=sk)
    quantization_shape = QuantizationShape(output_quant, m=qm, n=qn)
    layouts = MatrixLayouts(
        a_layout=left_layout,
        b_layout=right_layout,
        c_layout=out_layout,
        as_layout=left_scale_layout,
        bs_layout=right_scale_layout,
        cs_layout=out_scale_layout,
    )

    return MXGEMMDescriptor(
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        scales_dtype=scales_dtype,
        operators_shape=operators_shape,
        scales_shape=scales_shape,
        quantization_shape=quantization_shape,
        layouts=layouts,
    )


@lru_cache(maxsize=None)
def build_tunning_space():
    K_BM_list = [64, 128]
    K_BK_list = [128]
    K_Num_Stages_list = [3, 4, 5, 6, 7, 8, 9, 10]
    # K_Num_Stages_list = [3]
    K_Num_Threads_list = [256, 384]
    K_SMs_list = [108, 114]
    K_Cluster_M_list = [1, 2]
    K_Cluster_N_list = [1, 2]
    K_WGMMA_M_list = [64]
    K_WGMMA_K_list = [32]
    K_CTA_Swizzle_Lead_Dim_list = [CTASwizzleDim.ROW_LEAD, CTASwizzleDim.COL_LEAD]
    K_CTA_Swizzle_Lead_Size_list = [4, 8, 12, 16]
    K_AB_SMem_Swizzle_list = [SMemSwizzleBits.B128]

    K_Num_TMA_Math_Registers_list = [(24, 240), (40, 232)]
    bn_store_swizzle_list = [
        (256, SMemSwizzleBits.B128),
        (224, SMemSwizzleBits.B64),
        (192, SMemSwizzleBits.B128),
        (160, SMemSwizzleBits.B64),
        (128, SMemSwizzleBits.B128),
        (112, SMemSwizzleBits.DISABLE),
        (96, SMemSwizzleBits.DISABLE),
        (80, SMemSwizzleBits.DISABLE),
        (64, SMemSwizzleBits.DISABLE),
        (48, SMemSwizzleBits.DISABLE),
        (32, SMemSwizzleBits.DISABLE),
    ]

    param_product = product(
        K_BM_list,
        K_BK_list,
        K_Num_Stages_list,
        K_Num_Threads_list,
        K_SMs_list,
        K_Cluster_M_list,
        K_Cluster_N_list,
        K_WGMMA_M_list,
        K_WGMMA_K_list,
        K_CTA_Swizzle_Lead_Dim_list,
        K_CTA_Swizzle_Lead_Size_list,
        K_AB_SMem_Swizzle_list,
        K_Num_TMA_Math_Registers_list,
        bn_store_swizzle_list,
    )

    space = []
    for params in param_product:
        (
            bm,
            bk,
            num_stages,
            num_threads,
            sm,
            cluster_m,
            cluster_n,
            wgmma_m,
            wgmma_k,
            cta_swizzle_lead_dim,
            cta_swizzle_lead_size,
            ab_smem_swizzle,
            num_tma_math_regs,
            bn_store_swizzle,
        ) = params

        num_tma_regs, num_math_regs = cast(Tuple[int, int], num_tma_math_regs)
        bn, store_swizzle = cast(Tuple[int, SMemSwizzleBits], bn_store_swizzle)

        space.append(
            {
                "K_BM": bm,
                "K_BK": bk,
                "K_Num_Stages": num_stages,
                "K_Num_Threads": num_threads,
                "K_Num_SMs": sm,
                "K_Cluster_M": cluster_m,
                "K_Cluster_N": cluster_n,
                "K_WGMMA_M": wgmma_m,
                "K_WGMMA_K": wgmma_k,
                "K_WGMMA_N": bn,
                "K_Num_TMA_Registers": num_tma_regs,
                "K_Num_Math_Registers": num_math_regs,
                "K_CTA_Swizzle_Lead_Dim": cta_swizzle_lead_dim,
                "K_CTA_Swizzle_Lead_Size": cta_swizzle_lead_size,
                "K_BN": bn,
                "K_AB_SMem_Swizzle": ab_smem_swizzle,
                "K_C_SMem_Swizzle": store_swizzle,
            }
        )

    return space


def build_kernel_parameters(
    desc: MXGEMMDescriptor,
    left: torch.Tensor,
    right: torch.Tensor,
    left_scale: torch.Tensor,
    right_scale: torch.Tensor,
):
    output = torch.empty(
        desc.os.m,
        desc.os.n,
        dtype=scalar_dtype_to_torch_dtype[desc.output_dtype],
        device=left.device,
    )
    if desc.layouts.c_layout == Layout.COLUMN_MAJOR:
        output = output.t()

    if desc.qs.enable:
        output_scale = torch.empty(
            desc.os.m // desc.qs.m,
            desc.os.n // desc.qs.n,
            dtype=scalar_dtype_to_torch_dtype[desc.scales_dtype],
            device=left.device,
        )
    else:
        output_scale = torch.empty(
            [],
            dtype=scalar_dtype_to_torch_dtype[desc.scales_dtype],
            device=left.device,
        )

    if desc.layouts.cs_layout == Layout.COLUMN_MAJOR:
        output_scale = output_scale.t()

    parameters = {
        RuntimeParameter.A_PTR: left,
        RuntimeParameter.B_PTR: right,
        RuntimeParameter.C_PTR: output,
        RuntimeParameter.AS_PTR: left_scale,
        RuntimeParameter.BS_PTR: right_scale,
        RuntimeParameter.CS_PTR: output_scale,
        RuntimeParameter.CUDA_STREAM: torch.cuda.current_stream(),
    }

    return parameters


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
    """
    Perform matrix multiplication with quantization support.

    Args:
        left (torch.Tensor): Left operand of the matrix multiplication.
        right (torch.Tensor): Right operand of the matrix multiplication.
        left_scale (torch.Tensor): Scale for the left operand.
        right_scale (torch.Tensor): Scale for the right operand.
        output_quant (bool): Whether to quantize the output.
        quant_size (int): Size of the quantization.
        out_dtype (Optional[torch.dtype]): Desired output data type. If `output_quant` is True, defaults to left's dtype;
            otherwise, defaults to `torch.bf16`.
        out_transpose (bool): Whether to transpose the output tensor.
        out_scale_transpose (bool): Whether to transpose the output scale tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Resulting tensor and scale tensor.

    """

    mx_desc = get_meta_info(
        left.dtype,
        right.dtype,
        left_scale.dtype,
        right_scale.dtype,
        output_quant,
        quant_size,
        out_dtype,
        left.shape,
        right.shape,
        left_scale.shape,
        right_scale.shape,
        left.is_contiguous(),
        right.is_contiguous(),
        left_scale.is_contiguous(),
        right_scale.is_contiguous(),
        out_transpose,
        out_scale_transpose,
    )
    kernel_params = build_kernel_parameters(
        mx_desc,
        left,
        right,
        left_scale,
        right_scale,
    )
    tuning_space = build_tunning_space()

    runtime = pool.generate_kernel(
        desc=mx_desc,
        parameters=kernel_params,
        space=tuning_space,
    )

    kernel_args = (
        kernel_params[RuntimeParameter.A_PTR],
        kernel_params[RuntimeParameter.B_PTR],
        kernel_params[RuntimeParameter.C_PTR],
        kernel_params[RuntimeParameter.AS_PTR],
        kernel_params[RuntimeParameter.BS_PTR],
        kernel_params[RuntimeParameter.CS_PTR],
        kernel_params[RuntimeParameter.CUDA_STREAM],
    )

    runtime(*kernel_args)

    return kernel_params[RuntimeParameter.C_PTR], kernel_params[RuntimeParameter.CS_PTR]


class _RegisterState:
    def __init__(self):
        self.registered = False
        self.registered_modules = set()

    def is_registered(self) -> bool:
        return self.registered

    def register(self, module_name: Optional[str] = None):
        if module_name is not None:
            if module_name in self.registered_modules:
                return
            self.registered_modules.add(module_name)

        if not self.registered:
            self.registered = True


register_state = _RegisterState()


def register_all_kernels():
    """
    Register all MX-GEMM template."""

    import mxblas.gemm.naive_templates as naive_templates

    register_state.register(naive_templates.__name__)
