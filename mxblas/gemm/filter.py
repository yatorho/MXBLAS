import functools
from abc import ABC, abstractmethod

from .keys import (
    DIVISIBLE_BY,
    EQ,
    GE,
    GT,
    K_BK,
    K_BM,
    K_BN,
    K_K,
    K_M,
    K_N,
    K_SK,
    K_SM,
    K_SN,
    LE,
    LT,
    Condition,
    IfExpr,
    K_C_SMem_Swizzle,
    K_Cluster_M,
    K_Cluster_N,
    K_Num_SMs,
    K_Num_Stages,
    K_Num_Threads,
    K_Quant,
    SMemSwizzleBits,
)


def ceil_div(x, y):
    return (x + y - 1) // y


class Filter(ABC):

    @abstractmethod
    def __call__(self) -> Condition:
        raise NotImplementedError("Subclasses must implement __call__ method")


@functools.lru_cache(maxsize=None)
def get_smem_size():
    return 227 * 1024


@functools.lru_cache(maxsize=None)
def naive_gemm_smem_size(K_Num_Stages):
    express = (K_BM * K_BK + K_BN * K_BK) * K_Num_Stages + 8 * K_Num_Stages * 2

    express += IfExpr(EQ(K_Quant, True), K_BM * K_BN, K_BM * K_BN * 2)

    return express


class BasicShapeFilter(Filter):

    def __call__(self):
        return DIVISIBLE_BY(K_M, K_SM).DIVISIBLE_BY(K_N, K_SN).DIVISIBLE_BY(K_K, K_SK)


class SMemSizeFilter(Filter):
    """
    Filter to check if the shared memory size for naive GEMM is within the limits.
    """

    def __call__(self):

        num_stages_threshold = IfExpr(LE(K_BK, 128), 4.1, 2.1)
        # num_stages_threshold = 3.1 if (bn <= 128) else 2.1

        return LT(naive_gemm_smem_size(K_Num_Stages), get_smem_size()).GT(
            naive_gemm_smem_size(K_Num_Stages + num_stages_threshold),
            get_smem_size(),
        )


class NumConsumerX64EqualBMFilter(Filter):
    """ """

    def __call__(self):
        K_Num_Consumer = (K_Num_Threads // 128) - 1
        return (
            EQ(K_Num_Threads % 128, 0)
            .GE(K_Num_Threads, 256)
            .EQ(K_BM, K_Num_Consumer * 64)
        )


class ClusterFilter(Filter):
    """
    Filter to check if the cluster configuration is valid.
    """

    def __call__(self):
        return (
            EQ(K_Num_SMs % (K_Cluster_M * K_Cluster_N), 0)
            .EQ(ceil_div(K_M, K_BM) % K_Cluster_M, 0)
            .EQ(ceil_div(K_N, K_BN) % K_Cluster_N, 0)
        )


class BMNKTileFilter(Filter):

    def __call__(self):
        cond = Condition()

        large_tiling_cond = GT(K_M * K_N, K_Num_SMs * K_BM * 128)
        cond.If(large_tiling_cond).GE(K_BN, 128).Else().LE(K_BN, 128).build()

        mxn = K_M * K_N
        large_blocks_threshold = 2048
        small_blocks_threshold = K_Num_SMs / 2.2

        for factor, value in [
            (128, 128),
            (160, 160),
            (192, 192),
        ]:
            cond.If(GE(mxn, large_blocks_threshold * K_BM * factor)).GE(
                K_BN, value
            ).build()

        for factor, value in [
            (128, 128),
            (96, 96),
            (80, 80),
            (64, 64),
        ]:
            cond.If(LE(mxn, small_blocks_threshold * K_BM * factor)).LE(
                K_BN, value
            ).build()

        cond.If(GT(mxn, K_Num_SMs * 128 * 256)).EQ(K_BM, 128).build()

        return cond


def CTypeBytes():
    return IfExpr(EQ(K_Quant, True), 1, 2)


class CSMemSwizzleBNFilter(Filter):
    """
    Filter to check if the quantization and storing swizzling stride for N-dimension is valid.
    """

    def __call__(self):
        return (
            Condition()
            .If(EQ(K_C_SMem_Swizzle, SMemSwizzleBits.B128))
            .DIVISIBLE_BY(K_BN, 128 / CTypeBytes())
            .build()
            .If(EQ(K_C_SMem_Swizzle, SMemSwizzleBits.B64))
            .DIVISIBLE_BY(K_BN, 64 / CTypeBytes())
            .build()
            .If(EQ(K_C_SMem_Swizzle, SMemSwizzleBits.B32))
            .DIVISIBLE_BY(K_BN, 32 / CTypeBytes())
            .build()
        )
