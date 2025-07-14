from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ScalarDType(Enum):
    FP8_E4M3 = "FP8_E4M3"
    FP8_E5M2 = "FP8_E5M2"
    FP32 = "FP32"
    BF16 = "BF16"
    FP16 = "FP16"


scalar_type_bytes = {
    ScalarDType.FP8_E4M3: 1,
    ScalarDType.FP8_E5M2: 1,
    ScalarDType.FP32: 4,
    ScalarDType.BF16: 2,
    ScalarDType.FP16: 2,
}


class Layout(Enum):
    ROW_MAJOR = "ROW_MAJOR"
    COLUMN_MAJOR = "COLUMN_MAJOR"



value_dtype_to_cpp_type =  {
    ScalarDType.FP8_E4M3: "__nv_fp8_e4m3",
    ScalarDType.FP8_E5M2: "__nv_fp8_e5m2",
    ScalarDType.FP32: "float",
    ScalarDType.BF16: "__nv_bfloat16",
    ScalarDType.FP16: "half",
}


@dataclass(frozen=True)
class OperatorsShape:
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class ScalesShape:
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class QuantizationShape:
    enable: bool
    m: int = 0
    n: int = 0


class ScalingPattern(Enum):
    BB = "BxB"
    GB = "GxB"
    CC = "CxC"
    TT = "TxT"
    TC = "TxC"
    CT = "CxT"
    GG = "GxG"
    BG = "BxG"


@dataclass(frozen=True)
class MatrixLayouts:
    a_layout: Layout
    b_layout: Layout
    c_layout: Layout
    as_layout: Optional[Layout] = None
    bs_layout: Optional[Layout] = None
    cs_layout: Optional[Layout] = None


class MXGEMMDescriptor:
    def __init__(
        self,
        input_dtype: ScalarDType,
        output_dtype: ScalarDType,
        scales_dtype: ScalarDType,
        operators_shape: OperatorsShape,
        scales_shape: ScalesShape,
        quantization_shape: QuantizationShape,
        layouts: MatrixLayouts,
    ):
        assert input_dtype in [ScalarDType.FP8_E4M3, ScalarDType.FP8_E5M2]

        quant_enable = quantization_shape.enable
        if quant_enable:
            assert output_dtype in [
                ScalarDType.FP8_E4M3,
                ScalarDType.FP8_E5M2,
            ]
        else:
            assert output_dtype in [
                ScalarDType.FP16,
                ScalarDType.BF16,
            ]

        assert scales_dtype in [ScalarDType.FP32, ScalarDType.FP16, ScalarDType.BF16]

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.scales_dtype = scales_dtype

        m = operators_shape.m
        n = operators_shape.n
        k = operators_shape.k
        sm = scales_shape.m
        sn = scales_shape.n
        sk = scales_shape.k

        assert m % sm == 0, f"m ({m}) must be divisible by sm ({sm})"
        assert n % sn == 0, f"n ({n}) must be divisible by sn ({sn})"
        assert k % sk == 0, f"k ({k}) must be divisible by sk ({sk})"

        self.os = operators_shape
        self.ss = scales_shape

        qm = quantization_shape.m
        qn = quantization_shape.n
        if quant_enable:
            assert qm is not None and m % qm == 0, f"m ({m}) must be divisible by qm ({qm})"
            assert qn is not None and n % qn == 0, f"n ({n}) must be divisible by qn ({qn})"

        self.qs = quantization_shape

        self.layouts = layouts

    def __str__(self):
        return (
            f"MXGEMMDescriptor(\n"
            f"  input_dtype={self.input_dtype.name},\n"
            f"  output_dtype={self.output_dtype.name},\n"
            f"  scales_dtype={self.scales_dtype.name},\n"
            f"  operators_shape={self.os},\n"
            f"  scales_shape={self.ss},\n"
            f"  quantization_shape={self.qs},\n"
            f"  layouts={self.layouts}\n"
            f")"
        )

    def __repr__(self):
        return (
            f"MXGEMMDescriptor("
            f"input_dtype=ValueDType.{self.input_dtype.name}, "
            f"output_dtype=ValueDType.{self.output_dtype.name}, "
            f"scales_dtype=ValueDType.{self.scales_dtype.name}, "
            f"operators_shape={repr(self.os)}, "
            f"scales_shape={repr(self.ss)}, "
            f"quantization_shape={repr(self.qs)}, "
            f"layouts={repr(self.layouts)}"
            f")"
        )

    def __eq__(self, other):
        if not isinstance(other, MXGEMMDescriptor):
            return NotImplemented
        return (
            self.input_dtype == other.input_dtype
            and self.output_dtype == other.output_dtype
            and self.scales_dtype == other.scales_dtype
            and self.os == other.os
            and self.ss == other.ss
            and self.qs == other.qs
            and self.layouts == other.layouts
        )

    def __hash__(self):
        return hash(
            (
                self.input_dtype,
                self.output_dtype,
                self.scales_dtype,
                self.os,
                self.ss,
                self.qs,
                self.layouts
            )
        )

    @property
    def scaling_pattern(self) -> ScalingPattern:
        m = self.os.m
        n = self.os.n
        k = self.os.k

        sm = self.ss.m
        sn = self.ss.n
        sk = self.ss.k

        if (1 < sm <= m) and (1 < sn <= n) and (1 < sk <= k):
            return ScalingPattern.BB
        elif (sm == 1) and (1 < sn <= n) and (1 < sk <= k):
            return ScalingPattern.GB
        elif (sm == 1) and (sm == 1) and (sk == k):
            return ScalingPattern.CC
        elif (sm == m) and (sn == n) and (sk == k):
            return ScalingPattern.TT
        elif (sm == m) and (sn == 1) and (sk == k):
            return ScalingPattern.TC
        elif (sm == 1) and (sn == n) and (sk == k):
            return ScalingPattern.CT
        elif (sm == 1) and (sn == 1) and (1 < sk <= k):
            return ScalingPattern.GG
        elif (1 < sm <= m) and (sn == 1) and (1 < sk <= k):
            return ScalingPattern.BG
        else:
            raise ValueError(
                f"Unsupported scaling pattern for operators_shape={self.os} and scales_shape={self.ss}"
            )


global_keys = (
    "num_threads",
    "cluster_m",
    "cluster_n",
    "block_m",
    "block_n",
    "block_k",
    "num_stages",
    "num_sms",
    "wgmma_m",
    "wgmma_n",
    "wgmma_k",
    "ab_smem_swizzle",
    "c_smem_swizzle",
    "num_tma_registers",
    "num_math_registers",
    "cta_swizzle_lead_size",
    "cta_swizzle_lead_dim",
)
