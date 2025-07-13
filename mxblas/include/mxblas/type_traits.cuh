#ifndef MX_BLAS_TYPE_TRAITS_CUH_
#define MX_BLAS_TYPE_TRAITS_CUH_

#include <cuda_fp8.h>

namespace mxblas {

constexpr float SCALE_MIN_THRES = 1e-10;

template <typename T, typename U = float>
struct FloatTraits {};

template <typename U>
struct FloatTraits<__nv_fp8_e4m3, U> {
  static constexpr float max_value_f = 448.f;
  static constexpr U max_value = static_cast<U>(max_value_f);
  static constexpr U max_value_inv = static_cast<U>(1.f) / max_value;

  static constexpr U one_value = static_cast<U>(1.f);
};

template <>
struct FloatTraits<__nv_fp8_e4m3, __nv_bfloat16> {
  using U = __nv_bfloat16;

  static constexpr float max_value_f = 448.f;
  static constexpr float one_value_f = 1.f;

  static constexpr U max_value = __nv_bfloat16_raw{0x43E0};
  static constexpr U max_value_inv = __nv_bfloat16_raw{0x3B12};

  static constexpr U one_value = __nv_bfloat16_raw{0x3F80};
};

}  // namespace mxblas

#endif  // MX_BLAS_TYPE_TRAITS_CUH_
