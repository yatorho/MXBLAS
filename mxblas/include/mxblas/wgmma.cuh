#ifndef MXBLAS_WGMMMA_CUH_
#define MXBLAS_WGMMMA_CUH_

#include <cuda_fp8.h>

#include "com_utils.cuh"
#include "swizzle.cuh"

namespace mxblas {

DEVICE constexpr uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

template <SMemSwizzleBits swizzle>
struct SmemDesc {};

template <>
struct SmemDesc<SMemSwizzleBits::B32> {
  static constexpr DEVICE uint64_t base_value() {
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode((uint64_t)1) << 16;  // not used
    desc |= matrix_descriptor_encode((uint64_t)256) << 32;
    desc |= 3llu << 62;  // 32B swizzle
    return desc;
  }
};

template <>
struct SmemDesc<SMemSwizzleBits::B64> {
  static constexpr DEVICE uint64_t base_value() {
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode((uint64_t)1) << 16;  // not used
    desc |= matrix_descriptor_encode((uint64_t)512) << 32;
    desc |= 2llu << 62;  // 64B swizzle
    return desc;
  }
};

template <>
struct SmemDesc<SMemSwizzleBits::B128> {
  static constexpr DEVICE uint64_t base_value() {
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode((uint64_t)1) << 16;  // not used
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62;  // 64B swizzle
    return desc;
  }
};

template <SMemSwizzleBits swizzle = SMemSwizzleBits::B128, typename T>
DEVICE uint64_t make_smem_desc(T *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = SmemDesc<swizzle>::base_value();
  desc |= matrix_descriptor_encode(addr);
  return desc;
}

template <int32_t GN>
struct WGMMABase {
  static_assert(GN % 8 == 0, "GN must be multiple of 8.");

  static constexpr int WGMMA_N = GN;
  static constexpr int WGMMA_M = 64;
  static constexpr int WGMMA_K = 32;

  static constexpr int NUM_ELEMENTS = WGMMA_M * WGMMA_N / 128;
};

template <int32_t GN>
struct WGMMA : public WGMMABase<GN> {};

template <int32_t v>
struct add_one {
  static constexpr int32_t value = v + 1;
};

#define UNROLL_REGISTER_1x8(i, d)                                                      \
  "+f"(d[(i)][0]), "+f"(d[(i)][1]), "+f"(d[(i)][2]), "+f"(d[(i)][3]), "+f"(d[(i)][4]), \
      "+f"(d[(i)][5]), "+f"(d[(i)][6]), "+f"(d[(i)][7])

#define UNROLL_REGISTER_2x8(i, d)                                                                  \
  "+f"(d[(i)][0]), "+f"(d[(i)][1]), "+f"(d[(i)][2]), "+f"(d[(i)][3]), "+f"(d[(i)][4]),             \
      "+f"(d[(i)][5]), "+f"(d[(i)][6]), "+f"(d[(i)][7]), "+f"(d[add_one<i>::value][0]),            \
      "+f"(d[add_one<i>::value][1]), "+f"(d[add_one<i>::value][2]), "+f"(d[add_one<i>::value][3]), \
      "+f"(d[add_one<i>::value][4]), "+f"(d[add_one<i>::value][5]), "+f"(d[add_one<i>::value][6]), \
      "+f"(d[add_one<i>::value][7])

#define UNROLL_REGISTER_3x8(d) UNROLL_REGISTER_2x8(0, d), UNROLL_REGISTER_1x8(2, d)

#define UNROLL_REGISTER_4x8(d) UNROLL_REGISTER_2x8(0, d), UNROLL_REGISTER_2x8(2, d)

#define UNROLL_REGISTER_5x8(d) UNROLL_REGISTER_4x8(d), UNROLL_REGISTER_1x8(4, d)

#define UNROLL_REGISTER_6x8(d) \
  UNROLL_REGISTER_2x8(0, d), UNROLL_REGISTER_2x8(2, d), UNROLL_REGISTER_2x8(4, d)

#define UNROLL_REGISTER_7x8(d) UNROLL_REGISTER_6x8(d), UNROLL_REGISTER_1x8(6, d)

#define UNROLL_REGISTER_8x8(d)                                                     \
  UNROLL_REGISTER_2x8(0, d), UNROLL_REGISTER_2x8(2, d), UNROLL_REGISTER_2x8(4, d), \
      UNROLL_REGISTER_2x8(6, d)

#define UNROLL_REGISTER_9x8(d) UNROLL_REGISTER_8x8(d), UNROLL_REGISTER_1x8(8, d)

#define UNROLL_REGISTER_10x8(d)                                                    \
  UNROLL_REGISTER_2x8(0, d), UNROLL_REGISTER_2x8(2, d), UNROLL_REGISTER_2x8(4, d), \
      UNROLL_REGISTER_2x8(6, d), UNROLL_REGISTER_2x8(8, d)

#define UNROLL_REGISTER_11x8(d) UNROLL_REGISTER_10x8(d), UNROLL_REGISTER_1x8(10, d)

#define UNROLL_REGISTER_12x8(d)                                                    \
  UNROLL_REGISTER_2x8(0, d), UNROLL_REGISTER_2x8(2, d), UNROLL_REGISTER_2x8(4, d), \
      UNROLL_REGISTER_2x8(6, d), UNROLL_REGISTER_2x8(8, d), UNROLL_REGISTER_2x8(10, d)

#define UNROLL_REGISTER_13x8(d) UNROLL_REGISTER_12x8(d), UNROLL_REGISTER_1x8(12, d)

#define UNROLL_REGISTER_14x8(d)                                                         \
  UNROLL_REGISTER_2x8(0, d), UNROLL_REGISTER_2x8(2, d), UNROLL_REGISTER_2x8(4, d),      \
      UNROLL_REGISTER_2x8(6, d), UNROLL_REGISTER_2x8(8, d), UNROLL_REGISTER_2x8(10, d), \
      UNROLL_REGISTER_2x8(12, d)

#define UNROLL_REGISTER_15x8(d) UNROLL_REGISTER_14x8(d), UNROLL_REGISTER_1x8(14, d)

#define UNROLL_REGISTER_16x8(d)                                                         \
  UNROLL_REGISTER_2x8(0, d), UNROLL_REGISTER_2x8(2, d), UNROLL_REGISTER_2x8(4, d),      \
      UNROLL_REGISTER_2x8(6, d), UNROLL_REGISTER_2x8(8, d), UNROLL_REGISTER_2x8(10, d), \
      UNROLL_REGISTER_2x8(12, d), UNROLL_REGISTER_2x8(14, d)

template <>
struct WGMMA<32> : public WGMMABase<32> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,  "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},  "
          " %16,"
          " %17,"
          " %18,  %19,  %20;\n"
          "}\n"
          : UNROLL_REGISTER_2x8(0, d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));

    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  }
};

template <>
struct WGMMA<48> : public WGMMABase<48> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23},  "
          " %24,"
          " %25,"
          " %26,  %27,  %28;\n"
          "}\n"
          : UNROLL_REGISTER_3x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<64> : public WGMMABase<64> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},  "
          " %32,"
          " %33,"
          " %34,  %35,  %36;\n"
          "}\n"
          : UNROLL_REGISTER_4x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<80> : public WGMMABase<80> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39},  "
          " %40,"
          " %41,"
          " %42,  %43,  %44;\n"
          "}\n"
          : UNROLL_REGISTER_5x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<96> : public WGMMABase<96> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n96k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},  "
          " %48,"
          " %49,"
          " %50,  %51,  %52;\n"
          "}\n"
          : UNROLL_REGISTER_6x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<112> : public WGMMABase<112> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n112k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55},  "
          " %56,"
          " %57,"
          " %58,  %59,  %60;\n"
          "}\n"
          : UNROLL_REGISTER_7x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<128> : public WGMMABase<128> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},  "
          " %64,"
          " %65,"
          " %66,    %67,  %68;\n"
          "}\n"
          : UNROLL_REGISTER_8x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<144> : public WGMMABase<144> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n144k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
          " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71},  "
          " %72,"
          " %73,"
          " %74,    %75,  %76;\n"
          "}\n"
          : UNROLL_REGISTER_9x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<160> : public WGMMABase<160> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n160k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
          " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
          " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79},  "
          " %80,"
          " %81,"
          " %82,    %83,  %84;\n"
          "}\n"
          : UNROLL_REGISTER_10x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<176> : public WGMMABase<176> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n176k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
          " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
          " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
          " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87},  "
          " %88,"
          " %89,"
          " %90, %91, %92;\n"
          "}\n"
          : UNROLL_REGISTER_11x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<192> : public WGMMABase<192> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n192k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
          " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
          " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
          " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
          " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},  "
          " %96,"
          " %97,"
          " %98, %99,  %100;\n"
          "}\n"
          : UNROLL_REGISTER_12x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<208> : public WGMMABase<208> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n208k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
          " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
          " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
          " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
          " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
          " %96,  %97,  %98,  %99,  %100, %101, %102, %103}, "
          " %104,"
          " %105,"
          " %106, %107, %108;\n"
          "}\n"
          : UNROLL_REGISTER_13x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<224> : public WGMMABase<224> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n224k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
          " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
          " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
          " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
          " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
          " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
          " %104, %105, %106, %107, %108, %109, %110, %111}, "
          " %112,"
          " %113,"
          " %114, %115, %116;\n"
          "}\n"
          : UNROLL_REGISTER_14x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<240> : public WGMMABase<240> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n240k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
          " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
          " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
          " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
          " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
          " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
          " %104, %105, %106, %107, %108, %109, %110, %111, "
          " %112, %113, %114, %115, %116, %117, %118, %119}, "
          " %120,"
          " %121,"
          " %122, %123, %124;\n"
          "}\n"
          : UNROLL_REGISTER_15x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  };

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

template <>
struct WGMMA<256> : public WGMMABase<256> {
  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(float d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    uint64_t desc_a = make_smem_desc<a_swizzle>(a_smem);
    uint64_t desc_b = make_smem_desc<b_swizzle>(b_smem);

    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "{\n"
          "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
          " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
          " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
          " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
          " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
          " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
          " %104, %105, %106, %107, %108, %109, %110, %111,  "
          " %112, %113, %114, %115, %116, %117, %118, %119,  "
          " %120, %121, %122, %123, %124, %125, %126, %127},"
          " %128,"
          " %129,"
          " %130, %131, %132;\n"
          "}\n"
          : UNROLL_REGISTER_16x8(d)
          : "l"(desc_a),
            "l"(desc_b),
            "n"(int32_t(ScaleD)),
            "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      static_assert(utils::always_false<T>::value, "Not implemented");
    } else {
      static_assert(utils::always_false<T>::value, "Not implemented");
    }
  }

  template <int32_t ScaleD,
            int ScaleA,
            int ScaleB,
            SMemSwizzleBits a_swizzle,
            SMemSwizzleBits b_swizzle = a_swizzle,
            typename T>
  static DEVICE void wgmma(uint32_t d[WGMMA_N / 16][8], T *a_smem, T *b_smem) {
    static_assert(utils::always_false<T>::value, "Not implemented");
  };
};

#undef UNROLL_REGISTER_2x8
#undef UNROLL_REGISTER_4x8
#undef UNROLL_REGISTER_6x8
#undef UNROLL_REGISTER_8x8
#undef UNROLL_REGISTER_10x8
#undef UNROLL_REGISTER_12x8
#undef UNROLL_REGISTER_14x8
#undef UNROLL_REGISTER_16x8

}  // namespace mxblas

#endif  //  MXBLAS_WGMMMA_CUH_
