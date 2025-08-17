import re
from typing import Collection, Iterable, List

from .descriptor import Layout, global_keys, value_dtype_to_cpp_type
from .keys import (
    K_BK,
    K_BM,
    K_BN,
    K_K,
    K_M,
    K_N,
    K_QM,
    K_QN,
    K_SK,
    K_SM,
    K_SN,
    K_WGMMA_K,
    K_WGMMA_M,
    K_WGMMA_N,
    K_A_Layout,
    K_AB_Scale_Type,
    K_AB_SMem_Swizzle,
    K_AB_Type,
    K_B_Layout,
    K_C_Layout,
    K_C_Scale_Type,
    K_C_SMem_Swizzle,
    K_C_Type,
    K_Cluster_M,
    K_Cluster_N,
    K_CTA_Swizzle_Lead_Dim,
    K_CTA_Swizzle_Lead_Size,
    K_Num_Math_Registers,
    K_Num_SMs,
    K_Num_Stages,
    K_Num_Threads,
    K_Num_TMA_Registers,
    Key_T,
    key_type_to_cpp_Ttype,
)


class F:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs):
        pattern = re.compile(r"\$\$(\w+)\$\$")

        def repl(m):
            key = m.group(1)
            return str(kwargs.get(key, m.group(0)))  # 不存在的 key 保留原样

        return pattern.sub(repl, self.template)

    def __str__(self):
        return self.template


class CodeGenerator:

    def register_keys(self, with_wrapper: bool = True) -> tuple[tuple[str, ...], str]:
        code = self.generate()
        keys = re.findall(r"\$\((.*?)\)", code)
        for key in keys:
            if key not in global_keys:
                raise ValueError(f"Key '{key}' is not registered in global_keys.")

        if with_wrapper:
            keys = (f"$({key})" for key in keys)

        return tuple(keys), code

    def parse_key(self, kvs: dict[str, str]) -> str:
        keys, template = self.register_keys(with_wrapper=True)
        for key in keys:
            unwrapped_key = key[2:-1]  # Remove the $() wrapper
            assert (
                unwrapped_key in kvs
            ), f"Key '{unwrapped_key}' is not found in the provided dictionary."

            # replace all the key in template with the value from kvs
            template = template.replace(key, kvs[unwrapped_key])

        return template

    def generate(self) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")


class BaseSignatureGenerator(CodeGenerator):

    def parse_arguments(self):

        return """

const {ab_scale_t} * __restrict__ a_scales,
const {ab_scale_t} * __restrict__ b_scales,
{c_scale_t} * __restrict__ c_scales,

const __grid_constant__ CUtensorMap tensor_map_a,
const __grid_constant__ CUtensorMap tensor_map_b,
const __grid_constant__ CUtensorMap tensor_map_c """.format(
            ab_scale_t=K_AB_Scale_Type,
            c_scale_t=K_C_Scale_Type,
        )

    def generate(self) -> str:
        return f"""
__global__ __launch_bounds__({K_Num_Threads}) void __cluster_dims__(
{K_Cluster_M}*{K_Cluster_N}, 1, 1) mx_gemm_kernel({self.parse_arguments()}) """


class IncludeGenerator(CodeGenerator):

    def __init__(self, mxblas_include_root: str = "."):
        self.mxblas_root = mxblas_include_root

    def generate(self):
        return """

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cassert>

#include \"{mxblas_root}/mxblas/com_utils.cuh\"
#include \"{mxblas_root}/mxblas/ptx_wrapper.cuh\"
#include \"{mxblas_root}/mxblas/swizzle.cuh\"
#include \"{mxblas_root}/mxblas/type_traits.cuh\"
#include \"{mxblas_root}/mxblas/wgmma.cuh\"\n""".format(
            mxblas_root=self.mxblas_root
        )


class PredefinedCodeGenerator(CodeGenerator):

    def generate(self) -> str:
        return r"""

template <int32_t BlockMajorSize, int32_t BlockMinorSize, SMemSwizzleBits swizzle_bits, typename T>
HOST CUtensorMap create_3d_tensor_map(const T *gmem_ptr, int global_height, int global_width) {
  using SwizzleInfo = TMASwizzleInfo<swizzle_bits>;

  constexpr int32_t alignment = SwizzleInfo::alignment();
  constexpr int32_t swizzle_stride = SwizzleInfo::template swizzle_stride_or<T, BlockMinorSize>();
  constexpr int32_t cols_per_phase = SwizzleInfo::cols_per_phase;
  static_assert(BlockMinorSize >= swizzle_stride && BlockMinorSize % swizzle_stride == 0);
  static_assert(BlockMajorSize >= cols_per_phase && BlockMajorSize % cols_per_phase == 0);

  constexpr int32_t rank = 3;
  CUtensorMap tma_map;

  void *gmem_address = (void *)gmem_ptr;
  CHECK_ALIGNMENT(gmem_address, alignment);
  // HOST_ASSERT(global_width % swizzle_stride == 0);
  // HOST_ASSERT(global_height % BlockMajorSize == 0);

  uint64_t gmem_prob_shape[5]
      = {swizzle_stride, (uint64_t)global_height, (uint64_t)(global_width / swizzle_stride), 1, 1};
  uint64_t gmem_prob_stride[5]
      = {sizeof(T), sizeof(T) * global_width, swizzle_stride * sizeof(T), 0, 0};
  uint32_t smem_box_shape[5]
      = {swizzle_stride, uint32_t(BlockMajorSize), uint32_t(BlockMinorSize / swizzle_stride), 1, 1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

  constexpr auto cu_swizzle_pattern = to_CUtensorMapSwizzle<swizzle_bits>();
  constexpr auto cu_data_type = to_CUtensorMapDataType<T>();

  assert(gmem_prob_shape[0] >= (uint64_t(1)));
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[1] >= (uint64_t(1)));
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[2] >= (uint64_t(1)));
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[3] >= (uint64_t(1)));
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[4] >= (uint64_t(1)));
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32));

  assert(gmem_prob_stride[0] == sizeof(T));
  assert(gmem_prob_stride[1] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[1] & 0b1111) == 0);
  assert(gmem_prob_stride[2] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[2] & 0b1111) == 0);
  assert(gmem_prob_stride[3] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[3] & 0b1111) == 0);
  assert(gmem_prob_stride[4] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[4] & 0b1111) == 0);

  assert(smem_box_shape[0] >= (uint32_t(1)));       // Size must be min 1
  assert(smem_box_shape[0] <= (uint32_t(1) << 8));  // Size must be max 2^8 = 256
  assert(smem_box_shape[1] >= (uint32_t(1)));       // Size must be min 1
  assert(smem_box_shape[1] <= (uint32_t(1) << 8));  // Size must be max 2^8 = 256
  assert(smem_box_shape[2] >= (uint32_t(1)));       // Size must be min 1
  assert(smem_box_shape[2] <= (uint32_t(1) << 8));  // Size must be max 2^8 = 256
  assert(smem_box_shape[3] >= (uint32_t(1)));       // Size must be min 1
  assert(smem_box_shape[3] <= (uint32_t(1) << 8));  // Size must be max 2^8 = 256
  assert(smem_box_shape[4] >= (uint32_t(1)));       // Size must be min 1
  assert(smem_box_shape[4] <= (uint32_t(1) << 8));  // Size must be max 2^8 = 256

  assert(smem_box_stride[0] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[0] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[1] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[1] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[2] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[2] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[3] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[3] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[4] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[4] <= (uint32_t(8)));  // Stride must be max 2^3 = 8

  CUresult result = cuTensorMapEncodeTiled(&tma_map,
                                           cu_data_type,
                                           rank,
                                           gmem_address,
                                           gmem_prob_shape,
                                           gmem_prob_stride + 1,
                                           smem_box_shape,
                                           smem_box_stride,
                                           CU_TENSOR_MAP_INTERLEAVE_NONE,
                                           cu_swizzle_pattern,
                                           CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // HOST_ASSERT(result == CUDA_SUCCESS);
  if (result != CUDA_SUCCESS) {
    printf("address: %p", gmem_address);
    CHECK_ALIGNMENT(gmem_address, alignment);
    std::cerr << "Error: " << result << std::endl;
    exit(1);
  }

  return tma_map;
}


template <typename T>
struct IsSupportedFloat16Type {
  static constexpr bool value = false;
};

template <>
struct IsSupportedFloat16Type<half> {
  using type = half;
  static constexpr bool value = true;
};

template <>
struct IsSupportedFloat16Type<__nv_bfloat16> {
  using type = __nv_bfloat16;
  static constexpr bool value = true;
};


template <typename OT, size_t... Indices, typename... Ts, typename RegT>
DEVICE void vectorized_cast_impl(RegT *out, std::index_sequence<Indices...>, Ts... inputs) {
  constexpr size_t max_index = sizeof...(Ts);
  static_assert((... && (Indices < max_index)), "Index out of range");
  ((*(reinterpret_cast<OT *>(out) + Indices) = static_cast<OT>(inputs)), ...);
}

// X[Index[i]] = X[i]
template <typename OT, size_t... Indices, typename... Ts, typename RegT>
DEVICE void vectorized_cast_map(RegT (&out)[sizeof...(Indices) * sizeof(OT) / sizeof(RegT)],
                                Ts... inputs) {
  static_assert(sizeof...(Indices) == sizeof...(inputs));
  vectorized_cast_impl<OT>(out, std::index_sequence<Indices...>{}, inputs...);
}

template <typename OT, typename... Ts, typename RegT>
DEVICE void vectorized_cast(RegT (&out)[sizeof...(Ts) * sizeof(OT) / sizeof(RegT)], Ts... inputs) {
  vectorized_cast_impl<OT>(out, std::index_sequence_for<Ts...>{}, inputs...);
}


enum class ReduceOp { MAX, AMAX, RAMAX, SUM };

enum class ReduceScope { WARP, HALF_WARP, QUARTER_WARP, EIGHTH_WARP, SIXTEENTH_WARP };

template <ReduceScope scope>
struct ReduceTraits {};

template <>
struct ReduceTraits<ReduceScope::WARP> {
  static constexpr int32_t offset_base = 16;
};

template <>
struct ReduceTraits<ReduceScope::HALF_WARP> {
  static constexpr int32_t offset_base = 8;
};

template <>
struct ReduceTraits<ReduceScope::QUARTER_WARP> {
  static constexpr int32_t offset_base = 4;
};

template <>
struct ReduceTraits<ReduceScope::EIGHTH_WARP> {
  static constexpr int32_t offset_base = 2;
};

template <>
struct ReduceTraits<ReduceScope::SIXTEENTH_WARP> {
  static constexpr int32_t offset_base = 1;
};

template <ReduceOp op>
struct ReduceFunc {};

template <>
struct ReduceFunc<ReduceOp::MAX> {
  template <typename T>
  static DEVICE T apply(T a, T b) {
    return a > b ? a : b;
    // return max(a, b);
  }
};

template <typename T>
DEVICE T abs(T a) {
  return a > (T)0 ? a : -a;
  // return abs(a);
}

template <>
struct ReduceFunc<ReduceOp::AMAX> {
  template <typename T>
  static DEVICE T apply(T a, T b) {
    // return abs(a) > abs(b) ? abs(a) : abs(b);
    return ReduceFunc<ReduceOp::MAX>::apply(abs(a), abs(b));
  }
};

template <>
struct ReduceFunc<ReduceOp::RAMAX> {
  template <typename T>
  static DEVICE T apply(T a, T b) {
    return ReduceFunc<ReduceOp::MAX>::apply(a, abs(b));
  }
};

template <>
struct ReduceFunc<ReduceOp::SUM> {
  template <typename T>
  static DEVICE T apply(T a, T b) {
    return a + b;
  }
};

template <ReduceOp op, ReduceScope scope, ReduceOp compare_op = op, typename T, typename... Args>
DEVICE T reduce(T value, Args... compare) {
  static_assert((std::is_same_v<T, Args> && ...), "All arguments must be the same type.");
  constexpr int32_t offset_base = ReduceTraits<scope>::offset_base;
  constexpr uint32_t mask = 0xFFFFFFFF;

  ((value = ReduceFunc<compare_op>::apply(value, compare)), ...);
#pragma unroll
  for (int32_t offset = offset_base; offset > 0; offset >>= 1) {
    value = ReduceFunc<op>::apply(value, __shfl_xor_sync(mask, value, offset));
  }
  return value;
}

template <typename T, int32_t pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template <typename T, int32_t pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template <typename T, int32_t pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size);

  static constexpr int32_t size = pack_size;

  PackType<T, pack_size> storage;
  T elem[pack_size];
};

template <typename T, typename U>
struct t22u2 {};

template <>
struct t22u2<float, __nv_fp8_e4m3> {
  using t_type = const float2;
  using u_type = __nv_fp8x2_storage_t;

  static DEVICE u_type convert(t_type a) {
    return __nv_cvt_float2_to_fp8x2(a, __NV_SATFINITE, __NV_E4M3);
  }
};

template <>
struct t22u2<__nv_bfloat16, __nv_fp8_e4m3> {
  using t_type = const __nv_bfloat162;
  using u_type = __nv_fp8x2_storage_t;

  static DEVICE u_type convert(t_type a) {
    // __nv_bfloat162_raw x{*cast_ptr<uint16_t>(&(a.x)),
    // *cast_ptr<uint16_t>(&(a.y))};
    __nv_bfloat162_raw x(a);
    return __nv_cvt_bfloat16raw2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
  }
};



enum class Layout {
  ROW_MAJOR,
  COLUMN_MAJOR,
};


template <int32_t D1,
          int32_t D2,
          Layout layout,
          int32_t BlockMajorSize,
          int32_t BlockMinorSize,
          SMemSwizzleBits swizzle_bits,
          typename T>
HOST CUtensorMap create_mx_gemm_tensor_map(const T *ptr) {
  if constexpr (layout == Layout::ROW_MAJOR) {
    return create_3d_tensor_map<BlockMajorSize, BlockMinorSize, swizzle_bits>(ptr, D1, D2);
  } else {
    return create_3d_tensor_map<BlockMinorSize, BlockMajorSize, swizzle_bits>(ptr, D2, D1);
  }
}


template <int32_t NUM_SM, int32_t M, int32_t N, int32_t BM, int32_t BN, int32_t GROUP_N_SIZE>
struct Scheduler {
  int32_t current_iter = -1;
  static constexpr int32_t total_blocks_m = CEIL_DIV(M, BM);
  static constexpr int32_t total_blocks_n = CEIL_DIV(N, BN);
  static constexpr int32_t num_blocks = total_blocks_m * total_blocks_n;

  const int32_t rank;

  DEVICE Scheduler(int32_t rank) : rank(rank) {}

  template <int32_t LEAD_DIM = 0>
  DEVICE bool next(int32_t &block_m, int32_t &block_n) {
    const auto next_block_idx = (++current_iter) * NUM_SM + rank;

    if (next_block_idx >= num_blocks) return false;

    constexpr auto num_blocks_per_group = total_blocks_m * GROUP_N_SIZE;
    auto group_idx = next_block_idx / num_blocks_per_group;
    auto first_n_block_idx = group_idx * GROUP_N_SIZE;
    auto num_n_blocks_in_group = min(GROUP_N_SIZE, total_blocks_n - first_n_block_idx);
    auto in_group_idx = next_block_idx % num_blocks_per_group;

    if constexpr (LEAD_DIM == 0) {
      block_m = in_group_idx / num_n_blocks_in_group;
      block_n = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
    } else {
      block_n = in_group_idx / num_n_blocks_in_group;
      block_m = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
    }

    return true;
  }
};

template <int32_t NUM_SM,
          int32_t M,
          int32_t N,
          int32_t BM,
          int32_t BN,
          int32_t LEAD_SIZE,
          int32_t LEAD_DIM>
auto DEVICE create_scheduler(int32_t rank) {
  if constexpr (LEAD_DIM == 0) {
    return Scheduler<NUM_SM, M, N, BM, BN, LEAD_SIZE>(rank);
  } else {
    return Scheduler<NUM_SM, N, M, BN, BM, LEAD_SIZE>(rank);
  }
}

"""


class SharedMemoryStructGenerator(CodeGenerator):
    
    def generate(self) -> str:
        return r"""
template <int32_t BM,
          int32_t BN,
          int32_t BK,
          int32_t QSIZE,
          typename AType,
          typename BType,
          typename CType>
struct SMem {
  /// GEMM need
  alignas(1024) AType A[BM * BK * QSIZE];
  alignas(1024) BType B[BK * BN * QSIZE];
  alignas(1024) CType C[BN * BM];
  alignas(16) alignas(8) uint64_t full[QSIZE], empty[QSIZE];
  /// *********

  static constexpr size_t A_SMEM_SIZE = BM * BK * sizeof(AType);
  static constexpr size_t B_SMEM_SIZE = BK * BN * sizeof(BType);
  static constexpr size_t C_SMEM_SIZE = BM * BN * sizeof(CType);
  static_assert(A_SMEM_SIZE % 1024 == 0);
  static_assert(B_SMEM_SIZE % 1024 == 0);
  static_assert(C_SMEM_SIZE % 1024 == 0);
};
"""


class ProducerGenerator(CodeGenerator):
    def generate(self):
        return r"""
    warpgroup_reg_dealloc<NUM_TMA_REGISTERS>();

    if (tid == 0) {
      int p = 1;
      int qidx = 0;
      uint32_t col_mask = 0;

      for (int i = 0; i < CLUSTER_M; ++i) {
        col_mask |= (1 << (i * CLUSTER_N));
      }
      int num_block_m, num_block_n;

      while (schedule.template next<LEAD_DIM>(num_block_m, num_block_n)) {
        num_block_n = num_block_n * CLUSTER_N + rank_n;
        num_block_m = num_block_m * CLUSTER_M + rank_m;

        for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
          if (qidx == QSIZE) {
            qidx = 0;
            p ^= 1;
          }
          wait(&empty[qidx], p);

          expect_bytes(&full[qidx], (BK * BN + BK * BM) * sizeof(ABType));

          if constexpr (CLUSTER_N > 1) {
            uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
            if (rank_n == 0) {
              load_async_multicast<load_stride>(&sA[qidx * BK * BM],
                                                &tensor_map_a,
                                                &full[qidx],
                                                block_k_iter * BK,
                                                num_block_m * BM,
                                                mask);
            }
          } else {
            load_async<load_stride>(&sA[qidx * BK * BM],
                                    &tensor_map_a,
                                    &full[qidx],
                                    block_k_iter * BK,
                                    num_block_m * BM);
          }

          if constexpr (CLUSTER_M > 1) {
            if (rank_m == 0) {
              load_async_multicast<load_stride>(&sB[qidx * BK * BN],
                                                &tensor_map_b,
                                                &full[qidx],
                                                block_k_iter * BK,
                                                num_block_n * BN,
                                                col_mask << rank_n);
            }
          } else {
            load_async<load_stride>(&sB[qidx * BK * BN],
                                    &tensor_map_b,
                                    &full[qidx],
                                    block_k_iter * BK,
                                    num_block_n * BN);
          }
        }
      }

      // make sure distributed shared barriers deconstructed safely
      if constexpr (CLUSTERS > 1) {
#pragma unroll
        for (int i = 0; i < QSIZE; ++i, ++qidx) {
          if (qidx == QSIZE) {
            qidx = 0;
            p ^= 1;
          }
          wait(&empty[qidx], p);
        }
      }
    }

"""


from enum import Enum


class PromotionPhaseType(Enum):
    MAIN_LOOP = "main-loop"
    EPILOGUE = "epilogue"


class PromotionOperationType(Enum):
    FULL = "full-broadcast"
    PARTIAL = "partial-broadcast"


from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateType:
    phase: PromotionPhaseType
    operation: PromotionOperationType


class ConsumerGenerator(CodeGenerator):
    def __init__(self, c_quant: bool, c_layout: Layout, template_type: TemplateType):
        self.c_quant = c_quant
        self.c_layout = c_layout
        self.template_type = template_type

    def assign_promotion_registers(self):
        phase = self.template_type.phase
        operation = self.template_type.operation
        if phase == PromotionPhaseType.MAIN_LOOP:

            def gen_promote_regs_code():
                if operation == PromotionOperationType.FULL:
                    code = r"""
            /// static_assert(SM % BM == 0);
            /// static_assert(SN % BN == 0);  // TODO
            /// static_assert(SK % BK == 0);

            constexpr int32_t scale_stride = K / SK;

            ABScaleType a_scale
                = __ldg(&a_scales[(num_block_m * BM / SM) * scale_stride + block_k_iter * BK / SK]);
            ABScaleType b_scale
                = __ldg(&b_scales[(num_block_n * BN / SN) * scale_stride + block_k_iter * BK / SK]);

            // ABScaleType a_scale = 1;
            // ABScaleType b_scale = 1;

            // float ab_scale = static_cast<float>(a_scale) * static_cast<float>(b_scale);
            float ab_scale
                = __shfl_sync(0xFFFFFFFF, static_cast<float>(a_scale) * static_cast<float>(b_scale), 0);

    #pragma unroll
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; m_it++) {
    #pragma unroll
            for (int w = 0; w < WGMMA_N; w += 16) {
    #define D(i) d[m_it][w / 16][i]
    #define ACC(i) acc[m_it][w / 16][i]
                D(0) += ACC(0) * ab_scale;
                D(1) += ACC(1) * ab_scale;
                D(4) += ACC(4) * ab_scale;
                D(5) += ACC(5) * ab_scale;
                D(2) += ACC(2) * ab_scale;
                D(3) += ACC(3) * ab_scale;
                D(6) += ACC(6) * ab_scale;
                D(7) += ACC(7) * ab_scale;
    #undef ACC
    #undef D
            }
            }
    """
                elif operation == PromotionOperationType.PARTIAL:
                    code = r"""
            /// static_assert(BM % SM == 0);
            // // static_assert(BN % SN == 0);  // TODO  // TODO
            /// static_assert(SK % BK == 0);
            /// static_assert(K % SK == 0);
            constexpr int32_t scale_stride = K / SK;

            constexpr int32_t NumAScalesIn16x16Tile = 2;
            constexpr int32_t NumBScalesIn16x16Tile = 4;

    #pragma unroll
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; m_it++) {
            int32_t s_row_0
                = num_block_m * BM + wg_idx * B_WG_M + m_it * WGMMA_M + warp * 16 + lane / 4;
            int32_t s_row_1 = s_row_0 + 8;

            ABScaleType a_scale_regs[NumAScalesIn16x16Tile];

            a_scale_regs[0] = __ldg(&a_scales[s_row_0 / SM * scale_stride + block_k_iter * BK / SK]);
            // a_scale_regs[1] = __ldg(&a_scales[s_row_1 / SM * scale_stride + block_k_iter * BK / SK]);

    #pragma unroll
            for (int w = 0; w < WGMMA_N; w += 16) {
                int32_t s_col = num_block_n * BN + w + 2 * (lane % 4);
                ABScaleType b_scale_regs[NumBScalesIn16x16Tile];

                b_scale_regs[0] = __ldg(&b_scales[(s_col + 0) / SN * scale_stride + block_k_iter * BK / SK]);
                // b_scale_regs[1] = __ldg(&b_scales[(s_col + 1) / SN * scale_stride + block_k_iter * BK / SK]);
                // b_scale_regs[2] = __ldg(&b_scales[(s_col + 8) / SN * scale_stride + block_k_iter * BK / SK]);
                // b_scale_regs[3] = __ldg(&b_scales[(s_col + 9) / SN * scale_stride + block_k_iter * BK / SK]);

    #define D(i) d[m_it][w / 16][i]
    #define ACC(i) acc[m_it][w / 16][i]
                D(0) += ACC(0) * a_scale_regs[0] * b_scale_regs[0];
                D(1) += ACC(1) * a_scale_regs[0] * b_scale_regs[1];
                D(2) += ACC(2) * a_scale_regs[1] * b_scale_regs[0];
                D(3) += ACC(3) * a_scale_regs[1] * b_scale_regs[1];
                D(4) += ACC(4) * a_scale_regs[0] * b_scale_regs[2];
                D(5) += ACC(5) * a_scale_regs[0] * b_scale_regs[3];
                D(6) += ACC(6) * a_scale_regs[1] * b_scale_regs[2];
                D(7) += ACC(7) * a_scale_regs[1] * b_scale_regs[3];
    #undef ACC
    #undef D
            }
            }

    """
                else:
                    assert (
                        False
                    ), f"You should not reach here, unsupported operation type: {operation} in {phase} phase."

                return code

            code = F(
                r"""
      float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8] = {0};
      float acc[B_WG_M / WGMMA_M][WGMMA_N / 16][8];

      for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
        if (qidx == QSIZE) {
          qidx = 0;
          p ^= 1;
        };
        wait(&full[qidx], p);
        warpgroup_arrive();

#pragma unroll
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
          ABType *wgmma_sA
              = sA + qidx * BK * BM + load_stride * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
          ABType *wgmma_sB = sB + qidx * BK * BN;

          {
            WGMMA::template wgmma<0, 1, 1, ab_swizzle>(acc[m_it], &wgmma_sA[0], &wgmma_sB[0]);
#pragma unroll
            for (int k_it = 1; k_it < load_stride / WGMMA_K; ++k_it) {
              WGMMA::template wgmma<1, 1, 1, ab_swizzle>(acc[m_it],
                                                         &wgmma_sA[k_it * WGMMA_K],
                                                         &wgmma_sB[k_it * WGMMA_K]);
            }
            wgmma_sA += load_stride * BM;
            wgmma_sB += load_stride * BN;
          }

#pragma unroll
          for (int bk = load_stride; bk < BK; bk += load_stride) {
#pragma unroll
            for (int k_it = 0; k_it < load_stride / WGMMA_K; ++k_it) {
              WGMMA::template wgmma<1, 1, 1, ab_swizzle>(acc[m_it],
                                                         &wgmma_sA[k_it * WGMMA_K],
                                                         &wgmma_sB[k_it * WGMMA_K]);
            }
            wgmma_sA += load_stride * BM;
            wgmma_sB += load_stride * BN;
          }
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        if (tid < CLUSTERS) arrive_cluster(&empty[qidx], tid);
                            
        $$promote_regs_code$$

      }
"""
            ).format(promote_regs_code=gen_promote_regs_code())

            scaled_regs = True

        elif phase == PromotionPhaseType.EPILOGUE:
            code = r"""
        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];

        {
          if (qidx == QSIZE) {
            qidx = 0;
            p ^= 1;
          };
          wait(&full[qidx], p);
          warpgroup_arrive();

  #pragma unroll
          for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
            // ABType *wgmma_sA = sA + qidx * BK * BM + 128 * (m_it + wg_idx *
            // B_WG_M / WGMMA_M) * WGMMA_M;
            ABType *wgmma_sA = sA + qidx * BK * BM + load_stride * wg_idx * B_WG_M
                                + load_stride * m_it * WGMMA_M;  // mysterious code speed up 10%
            ABType *wgmma_sB = sB + qidx * BK * BN;
            {
              WGMMA::template wgmma<0, 1, 1, ab_swizzle>(d[m_it], &wgmma_sA[0], &wgmma_sB[0]);
  #pragma unroll
              for (int k_it = 1; k_it < load_stride / WGMMA_K; ++k_it) {
                WGMMA::template wgmma<1, 1, 1, ab_swizzle>(d[m_it],
                                                          &wgmma_sA[k_it * WGMMA_K],
                                                          &wgmma_sB[k_it * WGMMA_K]);
              }
              wgmma_sA += load_stride * BM;
              wgmma_sB += load_stride * BN;
            }
  #pragma unroll
            for (int bk = load_stride; bk < BK; bk += load_stride) {
  #pragma unroll
              for (int k_it = 0; k_it < load_stride / WGMMA_K; ++k_it) {
                WGMMA::template wgmma<1, 1, 1, ab_swizzle>(d[m_it],
                                                          &wgmma_sA[k_it * WGMMA_K],
                                                          &wgmma_sB[k_it * WGMMA_K]);
              }
              wgmma_sA += load_stride * BM;
              wgmma_sB += load_stride * BN;
            }
          }
          warpgroup_commit_batch();
          warpgroup_wait<0>();
          if (tid < CLUSTERS) arrive_cluster(&empty[qidx], tid);
          ++qidx;
        }

        for (int block_k_iter = 1; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
          if (qidx == QSIZE) {
            qidx = 0;
            p ^= 1;
          };
          wait(&full[qidx], p);
          warpgroup_arrive();
  #pragma unroll
          for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
            ABType *wgmma_sA
                = sA + qidx * BK * BM + load_stride * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
            ABType *wgmma_sB = sB + qidx * BK * BN;
  #pragma unroll
            for (int bk = 0; bk < BK; bk += load_stride) {
  #pragma unroll
              for (int k_it = 0; k_it < load_stride / WGMMA_K; ++k_it) {
                WGMMA::template wgmma<1, 1, 1, ab_swizzle>(d[m_it],
                                                          &wgmma_sA[k_it * WGMMA_K],
                                                          &wgmma_sB[k_it * WGMMA_K]);
              }
              wgmma_sA += load_stride * BM;
              wgmma_sB += load_stride * BN;
            }
          }
          warpgroup_commit_batch();
          warpgroup_wait<0>();
          if (tid < CLUSTERS) arrive_cluster(&empty[qidx], tid);
        }

        static_assert(SK == K);
  """

            if operation == PromotionOperationType.FULL:
                code += r"""

        /// static_assert(SM % BM == 0);
        /// static_assert(SN % BN == 0);

        ABScaleType a_scale = __ldg(&a_scales[num_block_m * BM / SM]);
        ABScaleType b_scale = __ldg(&b_scales[num_block_n * BN / SN]);

        // float ab_scale = static_cast<float>(a_scale) * static_cast<float>(b_scale);
        float ab_scale
            = __shfl_sync(0xFFFFFFFF, static_cast<float>(a_scale) * static_cast<float>(b_scale), 0);

  """
                scaled_regs = False
            elif operation == PromotionOperationType.PARTIAL:
                code += r"""
      /// static_assert(BM % SM == 0);
      // static_assert(BN % SN == 0);  // TODO

      constexpr int32_t NumAScalesIn16x16Tile = 2;
      constexpr int32_t NumBScalesIn16x16Tile = 4;

      for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
        int32_t s_row1 = num_block_m * BM + wg_idx * B_WG_M + m_it * WGMMA_M + warp * 16 + lane / 4;
        int32_t s_row2 = s_row1 + 8;


        ABScaleType a_scale[NumAScalesIn16x16Tile];
        a_scale[0] = __ldg(&a_scales[s_row1]);
        a_scale[1] = __ldg(&a_scales[s_row2]);

        for (int w = 0; w < BN; w += 16) {
          int32_t s_col = num_block_n * BN + w + 2 * (lane % 4);

          ABScaleType b_scale[NumBScalesIn16x16Tile];

          b_scale[0] = __ldg(&b_scales[s_col]);
          b_scale[1] = __ldg(&b_scales[s_col + 1]);
          b_scale[2] = __ldg(&b_scales[s_col + 8]);
          b_scale[3] = __ldg(&b_scales[s_col + 9]);

          d[m_it][w / 16][0] *= a_scale[0] * b_scale[0];
          d[m_it][w / 16][1] *= a_scale[0] * b_scale[1];
          d[m_it][w / 16][2] *= a_scale[1] * b_scale[0];
          d[m_it][w / 16][3] *= a_scale[1] * b_scale[1];
          d[m_it][w / 16][4] *= a_scale[0] * b_scale[2];
          d[m_it][w / 16][5] *= a_scale[0] * b_scale[3];
          d[m_it][w / 16][6] *= a_scale[1] * b_scale[2];
          d[m_it][w / 16][7] *= a_scale[1] * b_scale[3];
        }
      }
"""
                scaled_regs = True
            else:
                assert (
                    False
                ), f"You should not reach here, unsupported operation type: {operation} in {phase} phase."

        else:
            raise ValueError(f"Unsupported template type: {self.template_type}. ")

        return code, scaled_regs

    def storing_to_smem_code(
        self,
        wgmma_regs_name: str,
        scaled_wgmma_regs: bool,
        need_quantization: bool,
        c_layout: Layout,
    ):
        if c_layout == Layout.COLUMN_MAJOR:
            raise NotImplementedError(
                "Column major layout is not supported for storing to shared memory."
            )

        if not need_quantization:
            code = F(
                r"""
        static_assert(IsSupportedFloat16Type<CType>::value, "CType must be a supported float16 type.");
          // CType *block_sC = sC + wg_idx * B_WG_M * BN;
          CType *block_sC = sC;
#pragma unroll
          for (int n_it = 0; n_it < WGMMA_N / store_stride; n_it++) {
#pragma unroll
            for (int w = n_it * store_stride; w < (n_it + 1) * store_stride; w += 16) {
#pragma unroll
              for (int m_it = 0; m_it < B_WG_M / WGMMA_M; m_it++) {
                auto addr = &block_sC[swizzle::apply(
                    (wg_idx * B_WG_M + m_it * WGMMA_M + warp * 16 + lane % 16) * store_stride
                    + w % store_stride + 8 * (lane / 16))];

                uint32_t reg[4];
#define D(i) d[m_it][w / 16][i] $$reg_scale_name$$
                vectorized_cast<CType>(reg, D(0), D(1), D(2), D(3), D(4), D(5), D(6), D(7));
#undef D
                stmatrix<false>(addr, reg[0], reg[1], reg[2], reg[3]);
              }
            }
            block_sC += store_stride * BM;
          }
"""
            ).format(reg_scale_name=" * ab_scale" if not scaled_wgmma_regs else "")
        else:
            code = F(
                r"""
          constexpr int32_t c_q_m = QM;
          constexpr int32_t c_q_n = QN;

          static_assert(c_q_m == 1);
          static_assert(c_q_n % 16 == 0);  // only support per-token quantization yet
          static_assert(store_stride % c_q_n == 0);
          static_assert(c_q_n == 16 || c_q_n == 32 || c_q_n == 64 || c_q_n == 128);

          CType *block_sC = sC;

          constexpr int32_t scale_m = M / c_q_m;
          constexpr int32_t scale_bm = BM / c_q_m;
          constexpr int32_t scale_n = N / c_q_n;
          constexpr int32_t scale_bn = BN / c_q_n;
          CScaleType *scale_base
              = c_scales + num_block_n * scale_bn + num_block_m * scale_bm * scale_n;

          constexpr int32_t block_size = BM / c_q_m * BN / c_q_n;
          static_assert(store_stride % c_q_n == 0);
          constexpr int32_t q_store_stride = store_stride / c_q_n;

          static_assert(B_WG_M % c_q_m == 0);
          constexpr int32_t scale_bwgm = B_WG_M / c_q_m;
          static_assert(WGMMA_M % c_q_m == 0);
          constexpr int32_t scale_wgmma_m = WGMMA_M / c_q_m;
          constexpr int32_t warp_m = 16 / c_q_m;

#pragma unroll
          for (int n_it = 0; n_it < WGMMA_N / store_stride; n_it++) {
#pragma unroll
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; m_it++) {
              const int32_t row = lane / 4;
              const int32_t col = (lane % 4) * 2;
              const int32_t q_row = row / c_q_m;

#pragma unroll
              for (int s_it = n_it * store_stride, s_it_q = 0; s_it < (n_it + 1) * store_stride;
                   s_it += c_q_n, s_it_q++) {
                float amax0 = 0;
                float amax1 = 0;

                constexpr float compare_base = SCALE_MIN_THRES;

                //// find amax and add threshold
                if constexpr (c_q_n == 128) {
#define D(i)                                                                           \
  d[m_it][s_it / 16][i], d[m_it][s_it / 16 + 1][i], d[m_it][s_it / 16 + 2][i],         \
      d[m_it][s_it / 16 + 3][i], d[m_it][s_it / 16 + 4][i], d[m_it][s_it / 16 + 5][i], \
      d[m_it][s_it / 16 + 6][i], d[m_it][s_it / 16 + 7][i]
                  amax0 = reduce<ReduceOp::MAX, ReduceScope::EIGHTH_WARP, ReduceOp::RAMAX>(
                      compare_base,
                      D(0),
                      D(1),
                      D(4),
                      D(5));
                  amax1 = reduce<ReduceOp::MAX, ReduceScope::EIGHTH_WARP, ReduceOp::RAMAX>(
                      compare_base,
                      D(2),
                      D(3),
                      D(6),
                      D(7));
#undef D
                } else if constexpr (c_q_n == 64) {
#define D(i)                                                                   \
  d[m_it][s_it / 16][i], d[m_it][s_it / 16 + 1][i], d[m_it][s_it / 16 + 2][i], \
      d[m_it][s_it / 16 + 3][i]
                  amax0 = reduce<ReduceOp::MAX, ReduceScope::EIGHTH_WARP, ReduceOp::RAMAX>(
                      compare_base,
                      D(0),
                      D(1),
                      D(4),
                      D(5));
                  amax1 = reduce<ReduceOp::MAX, ReduceScope::EIGHTH_WARP, ReduceOp::RAMAX>(
                      compare_base,
                      D(2),
                      D(3),
                      D(6),
                      D(7));
#undef D
                } else if constexpr (c_q_n == 32) {
#define D(i) d[m_it][s_it / 16][i], d[m_it][s_it / 16 + 1][i]
                  amax0 = reduce<ReduceOp::MAX, ReduceScope::EIGHTH_WARP, ReduceOp::RAMAX>(
                      compare_base,
                      D(0),
                      D(1),
                      D(4),
                      D(5));
                  amax1 = reduce<ReduceOp::MAX, ReduceScope::EIGHTH_WARP, ReduceOp::RAMAX>(
                      compare_base,
                      D(2),
                      D(3),
                      D(6),
                      D(7));
#undef D
                } else if constexpr (c_q_n == 16) {
#define D(i) d[m_it][s_it / 16][i]
                  amax0 = reduce<ReduceOp::MAX, ReduceScope::EIGHTH_WARP, ReduceOp::RAMAX>(
                      compare_base,
                      D(0),
                      D(1),
                      D(4),
                      D(5));
                  amax1 = reduce<ReduceOp::MAX, ReduceScope::EIGHTH_WARP, ReduceOp::RAMAX>(
                      compare_base,
                      D(2),
                      D(3),
                      D(6),
                      D(7));
#undef D
                }

                constexpr float max_inv = FloatTraits<CType>::max_value_inv;
                constexpr float one_value = FloatTraits<CType>::one_value;

                amax0 = amax0 * max_inv $$reg_scale_name$$;
                amax1 = amax1 * max_inv $$reg_scale_name$$;

                if (lane % 2 == 0) {
                  int32_t group = ((lane) % 4) == 0;
                  scale_base[(wg_idx * scale_bwgm + m_it * scale_wgmma_m + warp * warp_m + q_row
                              + group * 8 / c_q_m)
                                 * scale_n
                             + n_it * q_store_stride + s_it_q]
                      = group ? amax1 : amax0;
                }

                auto scaling0 = $$one_or_ab_scale$$ / amax0;
                auto scaling1 = $$one_or_ab_scale$$ / amax1;

                ////

                //// perform quantization
#pragma unroll
                for (int w = s_it; w < s_it + c_q_n; w += 16) {
#define D(i) d[m_it][w / 16][i]

using b16 = utils::b16;

                  union {
                    __nv_fp8x2_storage_t fp8x2;
                    b16 value;
                  } reg[4];

                  reg[0].fp8x2 = t22u2<float, CType>::convert({D(0) * scaling0, D(1) * scaling0});
                  *utils::cast_ptr<b16>(&block_sC[(swizzle::apply(
                      (wg_idx * B_WG_M + m_it * WGMMA_M + warp * 16 + row) * store_stride
                      + w % store_stride + col))])
                      = reg[0].value;

                  reg[1].fp8x2 = t22u2<float, CType>::convert({D(2) * scaling1, D(3) * scaling1});
                  *utils::cast_ptr<b16>(&block_sC[(swizzle::apply(
                      (wg_idx * B_WG_M + m_it * WGMMA_M + warp * 16 + row + 8) * store_stride
                      + w % store_stride + col))])
                      = reg[1].value;

                  reg[2].fp8x2 = t22u2<float, CType>::convert({D(4) * scaling0, D(5) * scaling0});
                  *utils::cast_ptr<b16>(&block_sC[(swizzle::apply(
                      (wg_idx * B_WG_M + m_it * WGMMA_M + warp * 16 + row) * store_stride
                      + w % store_stride + col + 8))])
                      = reg[2].value;

                  reg[3].fp8x2 = t22u2<float, CType>::convert({D(6) * scaling1, D(7) * scaling1});
                  *utils::cast_ptr<b16>(&block_sC[(swizzle::apply(
                      (wg_idx * B_WG_M + m_it * WGMMA_M + warp * 16 + row + 8) * store_stride
                      + w % store_stride + col + 8))])
                      = reg[3].value;
#undef D
                }
              }
            }
            block_sC += store_stride * BM;
          }
"""
            ).format(
                reg_scale_name=f" * ab_scale" if not scaled_wgmma_regs else "",
                one_or_ab_scale="ab_scale" if not scaled_wgmma_regs else "one_value",
            )

        return code

    def pre_c_smem_code(self, c_layout: Layout) -> str:
        code = r"""
        using swizzle = ToSwizzleType<c_swizzle, CType>;
        

        asm volatile("cp.async.bulk.wait_group 0;");
"""

        if c_layout == Layout.COLUMN_MAJOR:
            code += r"""

        constexpr int32_t store_stride
            = TMASwizzleInfo<c_swizzle>::template swizzle_stride<CType>();
        static_assert(IsSupportedFloat16Type<CType>::value);

        static_assert(store_stride == WGMMA_M);
"""
        else:
            code += r"""
        constexpr int32_t store_stride
            = TMASwizzleInfo<c_swizzle>::template swizzle_stride_or<CType, BN>();


        static_assert(WGMMA_N % store_stride == 0);
        static_assert(store_stride % 16 == 0);
"""

        return code

    def post_c_smem_code(self, c_layout: Layout) -> str:
        code = F(
            r"""
        asm volatile("fence.proxy.async.shared::cta;");
        asm volatile("bar.sync 10, %0;\n" ::"n"(num_consumers * 128));
        if (threadIdx.x == 128) {
          store_async<store_stride>(&tensor_map_c, &sC[0], $$addr$$);
          asm volatile("cp.async.bulk.commit_group;");
        }
"""
        ).format(
            addr=(
                "num_block_m * BM, num_block_n * BN"
                if c_layout == Layout.COLUMN_MAJOR
                else "num_block_n * BN, num_block_m * BM"
            )
        )

        return code

    def epilgogue_code(self, wgmma_regs_name, scaled_wgmma_regs: bool = False) -> str:
        code = F(
            r"""

        $$pre_c_smem_code$$

        $$storing_smem_code$$

        $$post_c_smem_code$$

"""
        ).format(
            pre_c_smem_code=self.pre_c_smem_code(self.c_layout),
            storing_smem_code=self.storing_to_smem_code(
                wgmma_regs_name,
                scaled_wgmma_regs,
                self.c_quant,
                self.c_layout,
            ),
            post_c_smem_code=self.post_c_smem_code(self.c_layout),
        )

        return code

    def generate(self):

        promotion_code, scaled_regs = self.assign_promotion_registers()

        return """
    warpgroup_reg_alloc<NUM_MATH_REGISTERS>();

    --wg_idx;  // 0-based

    int p = 0;
    int qidx = 0;
    int num_block_m, num_block_n;

    while (schedule.template next<LEAD_DIM>(num_block_m, num_block_n)) {{
      num_block_n = num_block_n * CLUSTER_N + rank_n;
      num_block_m = num_block_m * CLUSTER_M + rank_m;

      const int lane = tid % 32, warp = tid / 32;

      {promotion_code}
      {epilogue_code}
    }}
""".format(
            promotion_code=promotion_code,
            epilogue_code=self.epilgogue_code("d", scaled_wgmma_regs=scaled_regs),
        )


class HostLauncherGenerator(CodeGenerator):
    def __init__(self, key_types: Collection[Key_T]):
        self.key_types = key_types

    def generate(self) -> str:
        code = F(
            r"""
  static void run_kernel(const $$ab_type$$ *__restrict__ A,
                         const $$ab_type$$ *__restrict__ B,
                         $$c_type$$ *__restrict__ C,
                         const void *__restrict__ a_scales,
                         const void *__restrict__ b_scales,
                         void *__restrict__ c_scales,
                         cudaStream_t stream) {

    namespace ul = utils;


    constexpr int M = $$m$$;
    constexpr int N = $$n$$;
    constexpr int K = $$k$$;

    constexpr int BM = $$block_m$$;
    constexpr int BN = $$block_n$$;
    constexpr int BK = $$block_k$$;


    auto tma_map_A = create_mx_gemm_tensor_map<M, K, $$a_layout$$, BM, BK, $$ab_smem_swizzle$$>(A);
    auto tma_map_B = create_mx_gemm_tensor_map<K, N, $$b_layout$$, BK, BN, $$ab_smem_swizzle$$>(B);
    auto tma_map_C = create_mx_gemm_tensor_map<M, N, $$c_layout$$, BM, BN, $$c_smem_swizzle$$>(C);

    auto *kernel = mx_gemm_kernel<$$forward_templates$$>;
    constexpr size_t SMEM_SIZE = sizeof(SMem<BM,
                                             BN,
                                             BK,
                                             $$num_stages$$,
                                             $$ab_type$$,
                                             $$ab_type$$,
                                             $$c_type$$>);

    constexpr size_t MaxDynamicSharedMemorySize = 227 * 1024;
    static_assert(SMEM_SIZE <= MaxDynamicSharedMemorySize);

    CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterWidth, $$cluster_m$$ * $$cluster_n$$));

    kernel<<<$$num_sms$$, $$num_threads$$, SMEM_SIZE, stream>>>(ul::cast_ptr<$$ab_scale_t$$>(a_scales),
                                                       ul::cast_ptr<$$ab_scale_t$$>(b_scales),
                                                       ul::cast_ptr<$$c_scale_t$$>(c_scales),
                                                       tma_map_A,
                                                       tma_map_B,
                                                       tma_map_C);

    ul::check_error();
  }
"""
        ).format(
            ab_type=K_AB_Type,
            c_type=K_C_Type,
            m=K_M,
            n=K_N,
            k=K_K,
            a_layout=K_A_Layout,
            b_layout=K_B_Layout,
            c_layout=K_C_Layout,
            num_stages=K_Num_Stages,
            block_m=K_BM,
            block_n=K_BN,
            block_k=K_BK,
            ab_scale_t=K_AB_Scale_Type,
            c_scale_t=K_C_Scale_Type,
            ab_smem_swizzle=K_AB_SMem_Swizzle,
            c_smem_swizzle=K_C_SMem_Swizzle,
            cluster_m=K_Cluster_M,
            cluster_n=K_Cluster_N,
            num_sms=K_Num_SMs,
            num_threads=K_Num_Threads,
            forward_templates=", ".join(str(key_type) for key_type in self.key_types),
        )

    
        return code


class NaiveGenerator(CodeGenerator):

    def __init__(
        self,
        include_generator: CodeGenerator,
        predefined_generator: CodeGenerator,
        shared_memory_generator: CodeGenerator,
        signature_generator: CodeGenerator,
        producer_generator: CodeGenerator,
        consumer_generator: CodeGenerator,
        host_launcher_generator: CodeGenerator,
        key_types: Collection[Key_T],
    ):
        self.include_generator = include_generator
        self.predefined_generator = predefined_generator
        self.shared_memory_generator = shared_memory_generator
        self.signature_generator = signature_generator
        self.producer_generator = producer_generator
        self.consumer_generator = consumer_generator
        self.host_launcher_generator = host_launcher_generator
        self.key_types = key_types


    def generate_template_keys(self, *key_types: Key_T) -> str:
        code = "template<"
        code += ", ".join(
            f"{key_type_to_cpp_Ttype(key_type)} {key_type}" for key_type in key_types
        )
        code += ">\n"

        return code

    def generate_body(self, code) -> str:
        code += self.signature_generator.generate() + "{\n"

        code += """

constexpr int32_t M = {m};
constexpr int32_t N = {n};
constexpr int32_t K = {k};

constexpr int32_t SM = {sm};
constexpr int32_t SN = {sn};
constexpr int32_t SK = {sk};

constexpr int32_t QM = {qm};
constexpr int32_t QN = {qn};

constexpr int32_t BM = {block_m};
constexpr int32_t BN = {block_n};
constexpr int32_t BK = {block_k};

constexpr int32_t NUM_SM = {num_sms};
constexpr int32_t NUM_THREADS = {num_threads};
constexpr int32_t QSIZE = {num_stages};
constexpr int32_t CLUSTER_M = {cluster_m};
constexpr int32_t CLUSTER_N = {cluster_n};
constexpr int32_t CLUSTERS = CLUSTER_M * CLUSTER_N;

constexpr int32_t WGMMA_M = {wgmma_m};
constexpr int32_t WGMMA_N = {wgmma_n};
constexpr int32_t WGMMA_K = {wgmma_k};
  
constexpr int32_t num_consumers = (NUM_THREADS / 128) - 1;
constexpr int32_t num_blocks_k = K / BK;
constexpr int32_t B_WG_M = BM / num_consumers;

static_assert(BN == WGMMA_N);
static_assert(BM % B_WG_M == 0);
static_assert(B_WG_M % WGMMA_M == 0);



using WGMMA = WGMMA<WGMMA_N>;
using ABType = {input_t};
using CType = {output_t};
using ABScaleType = {scale_t};
using CScaleType = {scale_t};

constexpr SMemSwizzleBits ab_swizzle = {ab_smem_swizzle};
constexpr SMemSwizzleBits c_swizzle = {c_smem_swizzle};
constexpr int32_t load_stride = TMASwizzleInfo<ab_swizzle>::template swizzle_stride<ABType>();

constexpr int32_t NUM_TMA_REGISTERS = {num_tma_registers};
constexpr int32_t NUM_MATH_REGISTERS = {num_math_registers};


extern __shared__ __align__(1024) uint8_t smem[];
using SMemType = SMem<BM, BN, BK, QSIZE, ABType, ABType, CType>;
SMemType &s = *reinterpret_cast<SMemType *>(smem);

ABType *sA = s.A, *sB = s.B;
CType *sC = s.C;
uint64_t *full = s.full, *empty = s.empty;

int32_t wg_idx = threadIdx.x / 128;
const int32_t tid = threadIdx.x % 128;

if (threadIdx.x == 0) {{
  asm volatile("prefetch.tensormap [%0];" : : "l"(&tensor_map_a) : "memory");
  asm volatile("prefetch.tensormap [%0];" : : "l"(&tensor_map_b) : "memory");
  asm volatile("prefetch.tensormap [%0];" : : "l"(&tensor_map_c) : "memory");
}}

if (threadIdx.x == 0) {{
#pragma unroll
  for (int32_t i = 0; i < QSIZE; ++i) {{
    init_barrier(&full[i], 0, 1);
    init_barrier(&empty[i], 0, num_consumers * CLUSTERS);
  }}
}}

int32_t rank;
asm volatile("mov.u32 %0, %clusterid.x;\\n" : "=r"(rank) :);

static_assert(NUM_SM % CLUSTERS == 0);
constexpr int32_t LEAD_SIZE = {cta_swizzle_lead_size};
constexpr int32_t LEAD_DIM = {cta_swizzle_lead_dim};
static_assert(LEAD_SIZE % (LEAD_DIM == 0 ? CLUSTER_N : CLUSTER_M) == 0);

auto schedule = create_scheduler<NUM_SM / CLUSTERS,
                                   M,
                                   N,
                                   BM * CLUSTER_M,
                                   BN * CLUSTER_N,
                                   LEAD_SIZE / (LEAD_DIM == 0 ? CLUSTER_N : CLUSTER_M),
                                   LEAD_DIM>(rank);

asm volatile("mov.u32 %0, %cluster_ctarank;\\n" : "=r"(rank) :);
int32_t rank_m = rank / CLUSTER_N;
int32_t rank_n = rank % CLUSTER_N;

asm volatile("barrier.cluster.arrive;\\n" : :);
asm volatile("barrier.cluster.wait;\\n" : :);


""".format(
            m=K_M,
            n=K_N,
            k=K_K,
            sm=K_SM,
            sn=K_SN,
            sk=K_SK,
            qm=K_QM,
            qn=K_QN,
            input_t=K_AB_Type,
            output_t=K_C_Type,
            scale_t=K_C_Scale_Type,
            block_m=K_BM,
            block_n=K_BN,
            block_k=K_BK,
            num_sms=K_Num_SMs,
            num_threads=K_Num_Threads,
            num_stages=K_Num_Stages,
            cluster_m=K_Cluster_M,
            cluster_n=K_Cluster_N,
            wgmma_m=K_WGMMA_M,
            wgmma_n=K_WGMMA_N,
            wgmma_k=K_WGMMA_K,
            num_tma_registers=K_Num_TMA_Registers,
            num_math_registers=K_Num_Math_Registers,
            ab_smem_swizzle=K_AB_SMem_Swizzle,
            c_smem_swizzle=K_C_SMem_Swizzle,
            cta_swizzle_lead_size=K_CTA_Swizzle_Lead_Size,
            cta_swizzle_lead_dim=K_CTA_Swizzle_Lead_Dim,
        )

        code += "if (wg_idx == 0) {\n // Producer do\n"
        code += self.producer_generator.generate()
        code += "} else {\n // Consumer do\n"
        code += self.consumer_generator.generate()
        code += "}\n"

        code += "}\n"
        return code

    def generate(self) -> str:
        code = self.include_generator.generate()
        code += "namespace mxblas {\n"
        code += self.predefined_generator.generate()
        code += self.shared_memory_generator.generate()
        code += self.generate_template_keys(*self.key_types)
        code = self.generate_body(code)

        code += self.generate_template_keys(*self.key_types)
        code += self.host_launcher_generator.generate()

        code += "}  // namespace mxblas \n"

        return code
