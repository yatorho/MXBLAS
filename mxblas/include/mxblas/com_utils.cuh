#ifndef MXBLAS_COM_UTILS_CUH_
#define MXBLAS_COM_UTILS_CUH_

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <type_traits>

namespace utils {

#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      std::string msg =                                                        \
          "CUDA error: " + std::string(cudaGetErrorString(error)) + " at " +   \
          std::string(__FILE__) + ":" + std::to_string(__LINE__);              \
      throw std::runtime_error(msg);                                           \
    }                                                                          \
  } while (0)

#define CHECK_CUDA_ERRORS() CUDA_CHECK(cudaGetLastError())

#define CHECK_ALIGNMENT(PTR, ALIGNMENT)                                        \
  do {                                                                         \
    if ((reinterpret_cast<uintptr_t>(PTR) % ALIGNMENT) != 0) {                 \
      std::cerr << "Alignment error: " << reinterpret_cast<uintptr_t>(PTR)     \
                << " is not aligned to " << ALIGNMENT << std::endl;            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define DEVICE_CHECK_ALIGNMENT(PTR, ALIGNMENT)                                 \
  do {                                                                         \
    if ((reinterpret_cast<uintptr_t>(PTR) % ALIGNMENT) != 0) {                 \
      printf("Alignment error: %lu is not aligned to %d\n",                    \
             reinterpret_cast<uintptr_t>(PTR), ALIGNMENT);                     \
      asm("trap;");                                                            \
    }                                                                          \
  } while (0)

#define DEVICE_ASSERT(condition)                                               \
  do {                                                                         \
    if (!(condition)) {                                                        \
      printf("[%d, %d, %d],[%d, %d, %d] Assertion failed: %s at %s:%d\n",      \
             blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,     \
             threadIdx.z, #condition, __FILE__, __LINE__);                     \
      asm("trap;");                                                            \
    }                                                                          \
  } while (0)

#define HOST_ASSERT(condition)                                                 \
  do {                                                                         \
    if (!(condition)) {                                                        \
      printf("Assertion failed: %s at %s:%d\n", #condition, __FILE__,          \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define DEVICE __device__ __forceinline__
#define HOST __host__ inline
#define HOST_DEVICE __host__ __device__ __forceinline__

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

namespace cnst {

template <class DType> HOST_DEVICE constexpr DType _max(DType a, DType b) {
  return a < b ? b : a;
}

template <class DType> HOST_DEVICE constexpr DType _min(DType a, DType b) {
  return a < b ? a : b;
}

template <class T> HOST_DEVICE constexpr T shiftr(T x, int32_t s) {
  return s >= 0 ? (x >> s) : (x << -s);
}

template <class T> HOST_DEVICE constexpr T pow(T base, int32_t exp) {
  return exp == 0 ? 1 : base * pow(base, exp - 1);
}

template <class T> HOST_DEVICE constexpr T pow2(T exp) {
  return pow(T(2), exp);
}

template <class T> HOST_DEVICE constexpr T log(T base, T value) {
  return (value < base) ? 0 : 1 + log(base, value / base);
}

template <class T> HOST_DEVICE constexpr T log2(T value) {
  return log(T(2), value);
}

template <typename T> struct always_false : std::false_type {};

} // namespace cnst

template <typename T> HOST T *cpu_ptr(int32_t N) {
  T *ptr;
  CUDA_CHECK(cudaMallocHost(&ptr, N * sizeof(T)));
  return ptr;
}

template <typename T> HOST void cpu_free(T *ptr) {
  CUDA_CHECK(cudaFreeHost(ptr));
}

template <typename T> HOST T *gpu_ptr(int32_t N) {
  T *ptr;
  CUDA_CHECK(cudaMalloc(&ptr, N * sizeof(T)));
  return ptr;
}

template <typename T> HOST void gpu_free(T *ptr) { CUDA_CHECK(cudaFree(ptr)); }

template <typename T> HOST void to_gpu(T *dst, const T *src, int32_t N) {
  CUDA_CHECK(cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> HOST void to_cpu(T *dst, const T *src, int32_t N) {
  CUDA_CHECK(cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyDeviceToHost));
}

void set_device(int32_t device_id) { CUDA_CHECK(cudaSetDevice(device_id)); }

int32_t get_device() {
  int32_t device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  return device_id;
}

HOST void device_sync() { CUDA_CHECK(cudaDeviceSynchronize()); }

void check_error(const char *message = "") {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << message << ": " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
};

struct Timer {
  virtual void tick() = 0;
  virtual double report_last_ms() = 0;
};

struct CPUTimer : public Timer {
  void tick() final {
    trace_[cur_] = std::chrono::high_resolution_clock::now();
    cur_ = 1 - cur_;
  }

  double report_last_ms() final {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        trace_[1 - cur_] - trace_[cur_]);

    return duration.count() / 1e3;
  }

private:
  decltype(std::chrono::high_resolution_clock::now()) trace_[2];
  int32_t cur_ = 0;
};

struct GPUTimer : public Timer {
  GPUTimer() {
    cudaEventCreate(&events_[0]);
    cudaEventCreate(&events_[1]);
  }

  ~GPUTimer() {
    cudaEventDestroy(events_[0]);
    cudaEventDestroy(events_[1]);
  }

  void tick() final {
    cudaEventRecord(events_[cur_]);
    cur_ = 1 - cur_;
  }

  double report_last_ms() final {
    float ms;
    cudaEventElapsedTime(&ms, events_[cur_], events_[1 - cur_]);
    return ms;
  }

  void sync_all() { device_sync(); }

private:
  cudaEvent_t events_[2];
  int32_t cur_ = 0;
};

template <typename F>
double GPU_bench(F &&f, int32_t warmup_iters = 10, int32_t repeats = 20) {
  GPUTimer timer;
  double total_time = 0;

  for (int32_t i = 0; i < warmup_iters; ++i) {
    f();
  }

  timer.tick();
  for (int32_t i = 0; i < repeats; ++i) {
    f();
  }

  timer.tick();
  timer.sync_all();

  return timer.report_last_ms() / repeats;
}

constexpr int32_t THREADS_PER_WARP = 32;
constexpr int32_t WARPS_PER_WG = 4;
constexpr int32_t THREADS_PER_WG = THREADS_PER_WARP * WARPS_PER_WG;

auto default_dim3(int32_t x, int32_t y = 1, int32_t z = 1) {
  return dim3(x, y, z);
}

template <typename F, typename... Args> HOST void launch(F &&f, Args... args) {
  f<<<1, 1>>>(args...);
}

template <typename F, typename... Args>
HOST void launch_warp(F &&f, Args... args) {
  f<<<1, THREADS_PER_WARP>>>(args...);
}

template <typename F, typename... Args>
HOST void launch_2(F &&f, Args... args) {
  f<<<1, 2>>>(args...);
}

template <typename F, typename... Args>
HOST void launch_4(F &&f, Args... args) {
  f<<<1, 2>>>(args...);
}

template <typename F, typename... Args>
HOST void launch_wg(F &&f, Args... args) {
  f<<<1, THREADS_PER_WG>>>(args...);
}

// template <typename F, typename C, typename... Args>
// HOST void launch(std::tuple<F, C> tuple, Args... args) {
//   auto f = std::get<0>(tuple);
//   auto gd = std::get<1>(tuple);
//   auto [grid, block] = gd();

//   printf("grid: (%d, %d, %d), block: (%d, %d, %d)\n", grid.x, grid.y, grid.z,
//          block.x, block.y, block.z);

//   f<<<grid, block>>>(args...);
// }

template <size_t N> struct alignas(N) AlignedBase {
  static_assert(N > 0, "N must be greater than 0!");
  uint8_t data[N];
};

using b8 = AlignedBase<1>;
using b16 = AlignedBase<2>;
using b32 = AlignedBase<4>;
using b64 = AlignedBase<8>;
using b128 = AlignedBase<16>;

template <typename T>
HOST_DEVICE void print_bin(const T &val, const char *prefix = "") {
  union {
    T val;
    uint8_t bytes[sizeof(T)];
  } u;

  u.val = val;

  printf("%s0b", prefix);
  for (int32_t i = sizeof(T) - 1; i >= 0; --i) {
    for (int32_t j = 7; j >= 0; --j) {
      printf("%d", (u.bytes[i] >> j) & 1);
    }
  }
  printf("\n");
}

template <typename T>
HOST_DEVICE void print_hex(const T &val, const char *prefix = "") {
  union {
    T val;
    uint8_t bytes[sizeof(T)];
  } u;

  u.val = val;

  printf("%s0x", prefix);
  for (int32_t i = sizeof(T) - 1; i >= 0; --i) {
    printf("%02X", u.bytes[i]);
  }
  printf("\n");
}

template <typename T, typename U,
          typename = std::enable_if_t<std::is_pointer_v<U>>>
HOST_DEVICE auto cast_ptr(U ptr) {
  using TargetType =
      std::conditional_t<std::is_const_v<std::remove_pointer_t<U>>, const T, T>;
  return reinterpret_cast<TargetType *>(ptr);
}

template <typename T> using always_false = typename cnst::always_false<T>;

template <typename T> HOST_DEVICE T safe_mul(T a, T b) {
  if constexpr (std::is_integral<T>::value) {
    if constexpr (std::is_signed<T>::value) {
      if (a > 0) {
        if (b > 0) {
          if (a > std::numeric_limits<T>::max() / b)
            throw std::overflow_error("Multiplication overflow");
        } else {
          if (b < std::numeric_limits<T>::min() / a)
            throw std::overflow_error("Multiplication overflow");
        }
      } else {
        if (b > 0) {
          if (a < std::numeric_limits<T>::min() / b)
            throw std::overflow_error("Multiplication overflow");
        } else {
          if (a != 0 && b < std::numeric_limits<T>::max() / a)
            throw std::overflow_error("Multiplication overflow");
        }
      }
    } else {
      if (b != 0 && a > std::numeric_limits<T>::max() / b)
        throw std::overflow_error("Multiplication overflow");
    }
    return a * b;
  } else if constexpr (std::is_floating_point<T>::value) {
    T result = a * b;
    if (!std::isfinite(result)) {
      throw std::overflow_error("Floating-point multiplication overflow");
    }
    return result;
  } else {
    static_assert(std::is_arithmetic<T>::value,
                  "safe_mul only supports arithmetic types");
  }
}

} // namespace utils

namespace cnst = utils::cnst;

#endif // MXBLAS_COM_UTILS_CUH_
