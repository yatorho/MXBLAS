#ifndef MXBLAS_SWIZZLE_CUH_
#define MXBLAS_SWIZZLE_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "com_utils.cuh"

namespace mxblas {

template <auto v>
struct C {
  using type = C<v>;
  static constexpr auto value = v;
  using value_type = decltype(v);
  HOST_DEVICE constexpr operator value_type() const noexcept { return value; }
  HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

template <class T, T v>
using constant = C<v>;

template <int32_t BBits, int32_t MBase, int32_t SShift = BBits>
struct Swizzle {
  static constexpr int32_t num_bits = BBits;
  static constexpr int32_t num_base = MBase;
  static constexpr int32_t num_shft = SShift;

  // To avoid clangd's no constexpr abs warning
  struct AvoidClangdWarning {
    template <typename T>
    static constexpr T abs(T n) {
      return n < 0 ? -n : n;
    }
  };

  static_assert(num_base >= 0, "MBase must be positive.");
  static_assert(num_bits >= 0, "BBits must be positive.");
  static_assert(AvoidClangdWarning::abs(num_shft) >= num_bits,
                "abs(SShift) must be more than BBits.");

  // using 'int' type here to avoid unintentially casting to unsigned... unsure.
  using bit_msk = constant<int32_t, (1 << num_bits) - 1>;
  using yyy_msk = constant<int32_t, bit_msk{} << (num_base + cnst::_max(0, num_shft))>;
  using zzz_msk = constant<int32_t, bit_msk{} << (num_base - cnst::_min(0, num_shft))>;
  using msk_sft = constant<int32_t, num_shft>;

  static constexpr uint32_t swizzle_code = uint32_t(yyy_msk{} | zzz_msk{});

  template <class Offset>
  HOST_DEVICE constexpr static auto apply(Offset const &offset) {
    return offset ^ cnst::shiftr(offset & yyy_msk{}, msk_sft{});  // ZZZ ^= YYY
  }

  template <class Offset>
  HOST_DEVICE constexpr auto operator()(Offset const &offset) const {
    return apply(offset);
  }
};

enum class SMemSwizzleBits : uint8_t {
  DISABLE,
  B32,
  B64,
  B128,
};

template <int32_t B, int32_t M, int32_t S>
HOST_DEVICE constexpr SMemSwizzleBits get_tma_swizzle_bits(Swizzle<B, M, S>) {
  if constexpr (M == 4 && S == 3) {
    switch (B) {
      default:
        static_assert(0 <= B && B <= 3,
                      "Expected B = 0,1,2, or 3 when M == 4. Unsupported "
                      "layout swizzle.");
      case 3:
        return SMemSwizzleBits::B128;
      case 2:
        return SMemSwizzleBits::B64;
      case 1:
        return SMemSwizzleBits::B32;
      case 0:
        return SMemSwizzleBits::DISABLE;
    }
  } else {
    static_assert(M < 0, "Unsupported layout swizzle.");
  }
}

template <SMemSwizzleBits SwizzleBits>
HOST constexpr CUtensorMapSwizzle to_CUtensorMapSwizzle() {
  // switch (SwizzleBits) {
  //   default:
  //   //   assert(false && "Unknown SMemSwizzleBits!");
  //     static_assert(always_false<SMemSwizzleBits>::value);
  //   case SMemSwizzleBits::DISABLE:
  //     return CU_TENSOR_MAP_SWIZZLE_NONE;
  //   case SMemSwizzleBits::B32:
  //     return CU_TENSOR_MAP_SWIZZLE_32B;
  //   case SMemSwizzleBits::B64:
  //     return CU_TENSOR_MAP_SWIZZLE_64B;
  //   case SMemSwizzleBits::B128:
  //     return CU_TENSOR_MAP_SWIZZLE_128B;
  // }
  if constexpr (SwizzleBits == SMemSwizzleBits::DISABLE) {
    return CU_TENSOR_MAP_SWIZZLE_NONE;
  } else if constexpr (SwizzleBits == SMemSwizzleBits::B32) {
    return CU_TENSOR_MAP_SWIZZLE_32B;
  } else if constexpr (SwizzleBits == SMemSwizzleBits::B64) {
    return CU_TENSOR_MAP_SWIZZLE_64B;
  } else if constexpr (SwizzleBits == SMemSwizzleBits::B128) {
    return CU_TENSOR_MAP_SWIZZLE_128B;
  }
}

template <typename T>
HOST constexpr CUtensorMapDataType to_CUtensorMapDataType() {
  if constexpr (std::is_same<T, float>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if constexpr (std::is_same<T, half>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, __nv_fp8_e5m2>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else {
    static_assert(cnst::always_false<T>::value, "Unsupported data type.");
  }
}

template <SMemSwizzleBits swizzle_bits>
struct SwizzleBitsToNum {};

template <>
struct SwizzleBitsToNum<SMemSwizzleBits::DISABLE> {
  static constexpr int32_t value = 0;
};

template <>
struct SwizzleBitsToNum<SMemSwizzleBits::B32> {
  static constexpr int32_t value = 32;
};

template <>
struct SwizzleBitsToNum<SMemSwizzleBits::B64> {
  static constexpr int32_t value = 64;
};

template <>
struct SwizzleBitsToNum<SMemSwizzleBits::B128> {
  static constexpr int32_t value = 128;
};

template <SMemSwizzleBits swizzle_bits>
struct TMASwizzleInfo {
  template <typename T>
  static constexpr int32_t swizzle_stride() {
    static_assert(swizzle_bits != SMemSwizzleBits::DISABLE, "Swizzle bits must be enabled.");
    return SwizzleBitsToNum<swizzle_bits>::value / sizeof(T);
  };

  template <typename T, int32_t default_stride>
  static constexpr int32_t swizzle_stride_or() {
    if constexpr (swizzle_bits == SMemSwizzleBits::DISABLE) {
      return default_stride;
    } else {
      return swizzle_stride<T>();
    }
  };

  static constexpr int32_t cols_per_phase = (swizzle_bits != SMemSwizzleBits::DISABLE)? 8 : 1;

  constexpr int32_t bytes_per_phase() {
    static_assert(swizzle_bits != SMemSwizzleBits::DISABLE, "Swizzle bits must be enabled.");
    return SwizzleBitsToNum<swizzle_bits>::value * cols_per_phase;
  }

  static constexpr int32_t alignment() {
    if constexpr (swizzle_bits == SMemSwizzleBits::DISABLE) {
      return 16;
    } else {
      return SwizzleBitsToNum<swizzle_bits>::value;
    }
  }
};

template <typename T>
struct SwizzleBase {
  static constexpr int32_t MBaseNormal = 4;

  static constexpr int32_t ElementByteSize = sizeof(T);
  static constexpr int32_t MBase = cnst::log2<int>(cnst::pow2(MBaseNormal) / ElementByteSize);
};
template <SMemSwizzleBits swizzle, typename T>
struct SwizzleBitsToSwizzle {};

template <typename T>
struct SwizzleBitsToSwizzle<SMemSwizzleBits::DISABLE, T> {
  using swizzle = Swizzle<0, SwizzleBase<T>::MBase, 3>;
};

template <typename T>
struct SwizzleBitsToSwizzle<SMemSwizzleBits::B32, T> {
  using swizzle = Swizzle<1, SwizzleBase<T>::MBase, 3>;
};

template <typename T>
struct SwizzleBitsToSwizzle<SMemSwizzleBits::B64, T> {
  using swizzle = Swizzle<2, SwizzleBase<T>::MBase, 3>;
};

template <typename T>
struct SwizzleBitsToSwizzle<SMemSwizzleBits::B128, T> {
  using swizzle = Swizzle<3, SwizzleBase<T>::MBase, 3>;
};

template <SMemSwizzleBits swizzle_bits, typename T>
using ToSwizzleType = typename SwizzleBitsToSwizzle<swizzle_bits, T>::swizzle;

}  // namespace mxblas

#endif  // MXBLAS_SWIZZLE_CUH_
