#ifndef MX_BLAS_PTX_WRAPPER_CUH_
#define MX_BLAS_PTX_WRAPPER_CUH_

#include "com_utils.cuh"

namespace mxblas {

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

__device__ static __forceinline__ void init_barrier(uint64_t *bar,
                                                    int thread_count,
                                                    int transaction_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(thread_count + transaction_count));
}


__device__ __forceinline__ void invalidate(uint64_t *bar) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n\t"
      "mbarrier.inval.shared::cta.b64 [%0]; \n\t"
      "}"
      :
      : "r"(bar_ptr));
}


__device__ void arrive_cluster(uint64_t *bar, uint32_t cta_id, uint32_t count = 1) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n\t"
      ".reg .b32 remAddr32;\n\t"
      "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
      "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n\t"
      "}"
      :
      : "r"(smem_addr), "r"(cta_id), "r"(count));
}

__device__ static __forceinline__ void wait(uint64_t *bar, int kPhaseBit) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(mbar_ptr),
      "r"(kPhaseBit));
}

__device__ static __forceinline__ void expect_bytes(uint64_t *bar, uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
               "r"(bytes));
}

template <int32_t stride, typename T>
DEVICE void load_async_multicast(T *dst,
                                 void const *const src_tma_map,
                                 uint64_t *bar,
                                 int global_col_idx,
                                 int global_row_idx,
                                 uint16_t cluster_mask) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
      "complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :
      : "r"(dst_ptr),
        "l"(tma_ptr),
        "r"(mbar_ptr),
        "n"(0),
        "r"(global_row_idx),
        "r"(global_col_idx / stride),
        "h"(cluster_mask)
      : "memory");
}

template <int32_t stride, typename T>
DEVICE void load_async(T *dst,
                       void const *const src_tma_map,
                       uint64_t *bar,
                       int global_col_idx,
                       int global_row_idx) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
      "complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(dst_ptr),
        "l"(tma_ptr),
        "r"(mbar_ptr),
        "n"(0),
        "r"(global_row_idx),
        "r"(global_col_idx / stride)
      : "memory");
}

__device__ void warpgroup_arrive() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }

__device__ void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
  static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <int32_t stride, typename T>
__device__ static inline void store_async(void const *dst_tma_map,
                                          T *src,
                                          int global_col_idx,
                                          int global_row_idx) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst_tma_map);
  uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
      " [%0, {%2, %3, %4}], [%1];"
      :
      : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(global_row_idx), "r"(global_col_idx / stride)
      : "memory");
}

template <bool transpose = false>
__device__ static inline void stmatrix(void *addr, uint32_t reg) {
  uint32_t addr_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  if constexpr (transpose) {
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x1.trans.shared::cta.b16"
        " [%0], {%1};" ::"r"(addr_ptr),
        "r"(reg));
  } else {
    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};" ::"r"(addr_ptr), "r"(reg));
  }
}

template <bool transpose = false>
__device__ static inline void stmatrix(void *addr, uint32_t reg0, uint32_t reg1) {
  uint32_t addr_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  if constexpr (transpose) {
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16"
        " [%0], {%1, %2};" ::"r"(addr_ptr),
        "r"(reg0),
        "r"(reg1));
  } else {
    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" ::"r"(addr_ptr),
                 "r"(reg0),
                 "r"(reg1));
  }
}

template <bool transpose = false, typename T>
__device__ static inline void stmatrix(void *addr, T reg0, T reg1, T reg2, T reg3) {
  static_assert(sizeof(T) == 4, "T must be 32-bit register");

  uint32_t addr_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  if constexpr (transpose) {
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16"
        " [%0], {%1, %2, %3, %4};" ::"r"(addr_ptr),
        "r"(reg0),
        "r"(reg1),
        "r"(reg2),
        "r"(reg3));
  } else {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr_ptr),
                 "r"(reg0),
                 "r"(reg1),
                 "r"(reg2),
                 "r"(reg3));
  }
}


__device__  __forceinline__ float ld_shared(const float* __restrict__ ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}


}

#endif  // MX_BLAS_PTX_WRAPPER_CUH_