from enum import Enum
from typing import Collection

from mxblas.gemm.descriptor import Layout
from mxblas.gemm.keys import (
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
    K_AS_Layout,
    K_B_Layout,
    K_BS_Layout,
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

from .generator import CodeGenerator, F


class ScalesUsingTMA(Enum):
    A_SCALES = "TMA_A_SCALES"
    B_SCALES = "TMA_B_SCALES"
    BOTH_SCALES = "TMA_BOTH_SCALES"


class SharedMemoryStructGenerator(CodeGenerator):

    def __init__(self, scales_using_tma: ScalesUsingTMA):
        self.scales_using_tma = scales_using_tma

    def generate(self):
        code = r"""
template <int32_t BM,
          int32_t BN,
          int32_t BK,
          int32_t QSIZE,
          typename AType,
          typename BType,
          typename CType,
          typename ABScaleType,
          int32_t SM,
          int32_t SN,
          int32_t SK>
struct SMem {
  alignas(1024) AType A[BM * BK * QSIZE];
  alignas(1024) BType B[BK * BN * QSIZE];
  alignas(1024) CType C[BN * BM];

  alignas(16) alignas(8) uint64_t full[QSIZE], empty[QSIZE];

  static constexpr size_t A_SMEM_SIZE = BM * BK * sizeof(AType);
  static constexpr size_t B_SMEM_SIZE = BK * BN * sizeof(BType);
  static constexpr size_t C_SMEM_SIZE = BM * BN * sizeof(CType);
  static_assert(A_SMEM_SIZE % 1024 == 0);
  static_assert(B_SMEM_SIZE % 1024 == 0);
  static_assert(C_SMEM_SIZE % 1024 == 0);
"""
        if self.scales_using_tma == ScalesUsingTMA.A_SCALES:
            code += r"""
alignas(128) ABScaleType a_scales[BM / SM * QSIZE];

static constexpr size_t AS_SMEM_SIZE_PER_STAGE = (BM / SM) * sizeof(ABScaleType);
static constexpr size_t SCALES_SIZE = AS_SMEM_SIZE_PER_STAGE;
static_assert(BM % SM == 0);
"""
        elif self.scales_using_tma == ScalesUsingTMA.B_SCALES:
            code += r"""
alignas(128) ABScaleType b_scales[BN / SN * QSIZE];

static constexpr size_t BS_SMEM_SIZE_PER_STAGE = (BN / SN) * sizeof(ABScaleType);
static constexpr size_t SCALES_SIZE = BS_SMEM_SIZE_PER_STAGE;
static_assert(BN % SN == 0);
"""
        elif self.scales_using_tma == ScalesUsingTMA.BOTH_SCALES:
            code += r"""
alignas(128) ABScaleType a_scales[BM / SM * QSIZE];
alignas(128) ABScaleType b_scales[BN / SN * QSIZE];

static constexpr size_t AS_SMEM_SIZE_PER_STAGE = (BM / SM) * sizeof(ABScaleType);
static constexpr size_t BS_SMEM_SIZE_PER_STAGE = (BN / SN) * sizeof(ABScaleType);
static constexpr size_t SCALES_SIZE = AS_SMEM_SIZE_PER_STAGE + BS_SMEM_SIZE_PER_STAGE;
static_assert(BM % SM == 0);
static_assert(BN % SN == 0);
"""
        else:
            raise ValueError("Invalid scales_using_tma value")

        code += r"""
};
"""
        return code


class SignatureGenerator(CodeGenerator):

    def __init__(self, scales_using_tma: ScalesUsingTMA):
        self.scales_using_tma = scales_using_tma

    def parse_arguments(self):

        if self.scales_using_tma == ScalesUsingTMA.A_SCALES:
            code = """
const __grid_constant__ CUtensorMap tensor_map_a_scales,
const {ab_scale_t} *__restrict__ b_scales_gptr,
""".format(
                ab_scale_t=K_AB_Scale_Type
            )
        elif self.scales_using_tma == ScalesUsingTMA.B_SCALES:
            code = """
const {ab_scale_t} *__restrict__ a_scales_gptr,
const __grid_constant__ CUtensorMap tensor_map_b_scales,
""".format(
                ab_scale_t=K_AB_Scale_Type
            )
        elif self.scales_using_tma == ScalesUsingTMA.BOTH_SCALES:
            code = """
const __grid_constant__ CUtensorMap tensor_map_a_scales,
const __grid_constant__ CUtensorMap tensor_map_b_scales,
"""
        else:
            raise ValueError("Invalid scales_using_tma value")
        code += """
{c_scale_t} *__restrict__ c_scales,
const __grid_constant__ CUtensorMap tensor_map_a,
const __grid_constant__ CUtensorMap tensor_map_b,
const __grid_constant__ CUtensorMap tensor_map_c
""".format(
            c_scale_t=K_C_Scale_Type
        )

        return code

    def generate(self) -> str:
        return f"""
__global__ __launch_bounds__({K_Num_Threads}) void __cluster_dims__(
{K_Cluster_M}*{K_Cluster_N}, 1, 1) mx_gemm_kernel({self.parse_arguments()}) """


class ProducerGenerator(CodeGenerator):

    def __init__(self, scales_using_tma: ScalesUsingTMA):
        self.scales_using_tma = scales_using_tma

    def generate(self) -> str:
        code = F(
            r"""
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

          expect_bytes(&full[qidx],
                       (BK * BN + BK * BM) * sizeof(ABType) + SMemType::SCALES_SIZE);

          if constexpr (CLUSTER_N > 1) {
            uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
            if (rank_n == 0) {
              load_async_multicast<load_stride>(&sA[qidx * BK * BM],
                                                &tensor_map_a,
                                                &full[qidx],
                                                block_k_iter * BK,
                                                num_block_m * BM,
                                                mask);
              $$load_async_multicast_a_scales_or_not$$
            }
          } else {
            load_async<load_stride>(&sA[qidx * BK * BM],
                                    &tensor_map_a,
                                    &full[qidx],
                                    block_k_iter * BK,
                                    num_block_m * BM);
            $$load_async_a_scales_or_not$$
          }

          if constexpr (CLUSTER_M > 1) {
            if (rank_m == 0) {
              load_async_multicast<load_stride>(&sB[qidx * BK * BN],
                                                &tensor_map_b,
                                                &full[qidx],
                                                block_k_iter * BK,
                                                num_block_n * BN,
                                                col_mask << rank_n);
              $$load_async_multicast_b_scales_or_not$$
            }
          } else {
            load_async<load_stride>(&sB[qidx * BK * BN],
                                    &tensor_map_b,
                                    &full[qidx],
                                    block_k_iter * BK,
                                    num_block_n * BN);
            $$load_async_b_scales_or_not$$
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
        ).format(
            load_async_multicast_a_scales_or_not=(
                """
load_async_multicast<BM / SM>(&a_scales_sptr[qidx * (BM / SM)],
                            &tensor_map_a_scales,
                            &full[qidx],
                            num_block_m * BM / SM,
                            block_k_iter * BK / SK,
                            mask);
"""
                if self.scales_using_tma
                in (ScalesUsingTMA.A_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
            load_async_a_scales_or_not=(
                """
load_async<BM / SM>(&a_scales_sptr[qidx * (BM / SM)],
                    &tensor_map_a_scales,
                    &full[qidx],
                    num_block_m * BM / SM,
                    block_k_iter * BK / SK);
"""
                if self.scales_using_tma
                in (ScalesUsingTMA.A_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
            load_async_multicast_b_scales_or_not=(
                """
load_async_multicast<BN / SN>(&b_scales_sptr[qidx * (BN / SN)],
                             &tensor_map_b_scales,
                             &full[qidx],
                             num_block_n * BN / SN,
                             block_k_iter * BK / SK,
                             col_mask << rank_n);
"""
                if self.scales_using_tma
                in (ScalesUsingTMA.B_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
            load_async_b_scales_or_not=(
                """
load_async<BN / SN>(&b_scales_sptr[qidx * (BN / SN)],
                   &tensor_map_b_scales,
                   &full[qidx],
                   num_block_n * BN / SN,
                   block_k_iter * BK / SK);
"""
                if self.scales_using_tma
                in (ScalesUsingTMA.B_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
        )

        return code


class ConsumerGenerator(CodeGenerator):

    def __init__(
        self, scales_using_tma: ScalesUsingTMA, c_quant: bool, c_layout: Layout
    ):
        self.scales_using_tma = scales_using_tma
        self.c_quant = c_quant
        self.c_layout = c_layout

    def assign_promotion_registers(self):
        code = F(
            r"""
      for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
        if (qidx == QSIZE) {
          qidx = 0;
          p ^= 1;
        };

        wait(&full[qidx], p);
        // static_assert(BN == SN);

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

        // // static_assert(BN % SN == 0);  // TODO  // TODO
        /// static_assert(SK % BK == 0);

        constexpr int32_t scale_stride = K / SK;
        constexpr int32_t NumAScalesIn16x16Tile = 2;
        constexpr int32_t NumBScalesIn16x16Tile = 4;

#pragma unroll
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; m_it++) {
          ABScaleType a_scale_regs[NumAScalesIn16x16Tile];

          $$load_a_scales$$

#pragma unroll
          for (int w = 0; w < WGMMA_N; w += 16) {
            ABScaleType b_scale_regs[NumBScalesIn16x16Tile];

            $$load_b_scales$$

#define D(i) d[m_it][w / 16][i]
#define ACC(i) acc[m_it][w / 16][i]
            D(0) += ACC(0) * static_cast<float>(a_scale_regs[0] * b_scale_regs[0]);
            D(1) += ACC(1) * static_cast<float>(a_scale_regs[0] * b_scale_regs[1]);
            D(2) += ACC(2) * static_cast<float>(a_scale_regs[1] * b_scale_regs[0]);
            D(3) += ACC(3) * static_cast<float>(a_scale_regs[1] * b_scale_regs[1]);
            D(4) += ACC(4) * static_cast<float>(a_scale_regs[0] * b_scale_regs[2]);
            D(5) += ACC(5) * static_cast<float>(a_scale_regs[0] * b_scale_regs[3]);
            D(6) += ACC(6) * static_cast<float>(a_scale_regs[1] * b_scale_regs[2]);
            D(7) += ACC(7) * static_cast<float>(a_scale_regs[1] * b_scale_regs[3]);
#undef ACC
#undef D
          }
        }
      }

"""
        ).format(
            load_a_scales=(
                """
          int32_t s_row_0 = wg_idx * B_WG_M + m_it * WGMMA_M + warp * 16 + lane / 4;
          a_scale_regs[0] = a_scales_sptr[(s_row_0 + 0) / SM];
          a_scale_regs[1] = a_scales_sptr[(s_row_0 + 8) / SM];
"""
                if self.scales_using_tma
                in (ScalesUsingTMA.A_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else """
          int32_t s_row_0 = num_block_m * BM + wg_idx * B_WG_M + m_it * WGMMA_M + warp * 16 + lane / 4;
          a_scale_regs[0] = __ldg(&a_scales_gptr[(s_row_0 + 0) / SM * scale_stride + block_k_iter * BK / SK]);
          // a_scale_regs[1] = __ldg(&a_scales_gptr[(s_row_0 + 8) / SM * scale_stride + block_k_iter * BK / SK]);
"""
            ),
            load_b_scales=(
                """
            int32_t s_col = w + 2 * (lane % 4);
            b_scale_regs[0] = b_scales_sptr[(s_col + 0) / SN];
            b_scale_regs[1] = b_scales_sptr[(s_col + 1) / SN];
            b_scale_regs[2] = b_scales_sptr[(s_col + 8) / SN];
            b_scale_regs[3] = b_scales_sptr[(s_col + 9) / SN];
"""
                if self.scales_using_tma
                in (ScalesUsingTMA.B_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else """
            int32_t s_col = num_block_n * BN + w + 2 * (lane % 4);
            b_scale_regs[0] = __ldg(&b_scales_gptr[(s_col + 0) / SN * scale_stride + block_k_iter * BK / SK]);
            // b_scale_regs[1] = __ldg(&b_scales_gptr[(s_col + 1) / SN * scale_stride + block_k_iter * BK / SK]);
            // b_scale_regs[2] = __ldg(&b_scales_gptr[(s_col + 8) / SN * scale_stride + block_k_iter * BK / SK]);
            // b_scale_regs[3] = __ldg(&b_scales_gptr[(s_col + 9) / SN * scale_stride + block_k_iter * BK / SK]);
"""
            ),
        )

        return code, True  # True for scaled accumulation registers

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

    def generate(self) -> str:

        promotion_code, scaled_regs = self.assign_promotion_registers()
        code = F(
            r"""
    warpgroup_reg_alloc<NUM_MATH_REGISTERS>();

    --wg_idx;  // 0-based

    int p = 0;
    int qidx = 0;
    int num_block_m, num_block_n;

    const int32_t lane = tid % 32, warp = tid / 32;

    while (schedule.template next<LEAD_DIM>(num_block_m, num_block_n)) {
      num_block_n = num_block_n * CLUSTER_N + rank_n;
      num_block_m = num_block_m * CLUSTER_M + rank_m;

      float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8] = {0};
      float acc[B_WG_M / WGMMA_M][WGMMA_N / 16][8];

      $$promotion_code$$
      $$epilogue_code$$
}
"""
        ).format(
            promotion_code=promotion_code,
            epilogue_code=self.epilgogue_code("d", scaled_wgmma_regs=scaled_regs),
        )

        return code


class HostLauncherGenerator(CodeGenerator):
    def __init__(self, scales_using_tma: ScalesUsingTMA, key_types: Collection[Key_T]):
        self.scales_using_tma = scales_using_tma
        self.key_types = key_types

    def generate(self) -> str:
        code = F(
            r"""
  static void run_kernel(const $$ab_type$$ *__restrict__ A,
                         const $$ab_type$$ *__restrict__ B,
                         $$c_type$$ *__restrict__ C,
                         const $$ab_scale_t$$ *__restrict__ a_scales,
                         const $$ab_scale_t$$ *__restrict__ b_scales,
                         void *__restrict__ c_scales,
                         cudaStream_t stream) {

    namespace ul = utils;


    constexpr int M = $$m$$;
    constexpr int N = $$n$$;
    constexpr int K = $$k$$;

    constexpr int SM = $$sm$$;
    constexpr int SN = $$sn$$;
    constexpr int SK = $$sk$$;

    constexpr int BM = $$block_m$$;
    constexpr int BN = $$block_n$$;
    constexpr int BK = $$block_k$$;

    constexpr Layout AS_LAYOUT = $$as_layout$$;
    constexpr Layout BS_LAYOUT = $$bs_layout$$;


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
                                             $$c_type$$,
                                             $$ab_scale_t$$,
                                             $$sm$$,
                                             $$sn$$,
                                             $$sk$$>);

    constexpr size_t MaxDynamicSharedMemorySize = 227 * 1024;
    static_assert(SMEM_SIZE <= MaxDynamicSharedMemorySize);

    CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterWidth, $$cluster_m$$ * $$cluster_n$$));

    $$make_a_scales_tensor_map_or_not$$
    $$make_b_scales_tensor_map_or_not$$


    kernel<<<$$num_sms$$, $$num_threads$$, SMEM_SIZE, stream>>>(
                                                       $$take_tensor_map_a_scales_or_not$$,
                                                       $$take_tensor_map_b_scales_or_not$$,
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
            sm=K_SM,
            sn=K_SN,
            sk=K_SK,
            a_layout=K_A_Layout,
            b_layout=K_B_Layout,
            c_layout=K_C_Layout,
            as_layout=K_AS_Layout,
            bs_layout=K_BS_Layout,
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
            make_a_scales_tensor_map_or_not=(
                """
    static_assert(AS_LAYOUT == Layout::COLUMN_MAJOR);
    auto tensor_map_a_scales = create_mx_gemm_tensor_map<M / SM,
                                                K / SK,
                                                AS_LAYOUT,
                                                BM / SM,
                                                1,
                                                SMemSwizzleBits::DISABLE>(a_scales);
"""
                if self.scales_using_tma
                in (ScalesUsingTMA.A_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
            make_b_scales_tensor_map_or_not=(
                """
    static_assert(BS_LAYOUT == Layout::ROW_MAJOR);
    auto tensor_map_b_scales = create_mx_gemm_tensor_map<K / SK,
                                                N / SN,
                                                BS_LAYOUT,
                                                1,
                                                BN / SN,
                                                SMemSwizzleBits::DISABLE>(b_scales);
"""
                if self.scales_using_tma
                in (ScalesUsingTMA.B_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
            take_tensor_map_a_scales_or_not=(
                "tensor_map_a_scales"
                if self.scales_using_tma
                in (ScalesUsingTMA.A_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else "a_scales"
            ),
            take_tensor_map_b_scales_or_not=(
                "tensor_map_b_scales"
                if self.scales_using_tma
                in (ScalesUsingTMA.B_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else "b_scales"
            ),
        )

        return code


class TMAScalesGenerator(CodeGenerator):

    def __init__(
        self,
        scales_using_tma: ScalesUsingTMA,
        include_generator: CodeGenerator,
        predefined_generator: CodeGenerator,
        shared_memory_generator: CodeGenerator,
        signature_generator: CodeGenerator,
        producer_generator: CodeGenerator,
        consumer_generator: CodeGenerator,
        host_launcher_generator: CodeGenerator,
        key_types: Collection[Key_T],
    ):
        super().__init__()

        self.scales_using_tma = scales_using_tma

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

    def generate_body(self):

        code = self.signature_generator.generate() + "{\n"

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
using SMemType
    = SMem<BM, BN, BK, QSIZE, ABType, ABType, CType, ABScaleType, SM, SN, SK>;

SMemType &s = *reinterpret_cast<SMemType *>(smem);

ABType *sA = s.A, *sB = s.B;
CType *sC = s.C;
uint64_t *full = s.full, *empty = s.empty;

{parse_smem_a_scales_or_not}
{parse_smem_b_scales_or_not}


int32_t wg_idx = threadIdx.x / 128;
const int32_t tid = threadIdx.x % 128;


if (threadIdx.x == 0) {{
  asm volatile("prefetch.tensormap [%0];" : : "l"(&tensor_map_a) : "memory");
  asm volatile("prefetch.tensormap [%0];" : : "l"(&tensor_map_b) : "memory");
  asm volatile("prefetch.tensormap [%0];" : : "l"(&tensor_map_c) : "memory");
  {prefetch_a_scales_or_not}
  {prefetch_b_scales_or_not}
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
            parse_smem_a_scales_or_not=(
                "ABScaleType *a_scales_sptr = s.a_scales;"
                if self.scales_using_tma
                in (ScalesUsingTMA.A_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
            parse_smem_b_scales_or_not=(
                "ABScaleType *b_scales_sptr = s.b_scales;"
                if self.scales_using_tma
                in (ScalesUsingTMA.B_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
            prefetch_a_scales_or_not=(
                'asm volatile("prefetch.tensormap [%0];" : : "l"(&tensor_map_a_scales) : "memory");'
                if self.scales_using_tma
                in (ScalesUsingTMA.A_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
            prefetch_b_scales_or_not=(
                'asm volatile("prefetch.tensormap [%0];" : : "l"(&tensor_map_b_scales) : "memory");'
                if self.scales_using_tma
                in (ScalesUsingTMA.B_SCALES, ScalesUsingTMA.BOTH_SCALES)
                else ""
            ),
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
        code += self.generate_body()

        code += self.generate_template_keys(*self.key_types)
        code += self.host_launcher_generator.generate()

        code += "}  // namespace mxblas \n"

        return code
