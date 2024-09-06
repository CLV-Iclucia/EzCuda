//
// Created by creeper on 9/2/24.
//

#ifndef GSHOP_GSHOP_INCLUDE_GSHOP_BACKENDS_CUDA_MACROS_H
#define GSHOP_GSHOP_INCLUDE_GSHOP_BACKENDS_CUDA_MACROS_H

#define EZ_CALLABLE __host__ __device__
#define EZ_FORCEINLINE __forceinline__
#define EZ_INLINE __inline__
#define EZ_SHARED __shared__

#ifndef NDEBUG
#define cudaSafeCheck(kernel) do { \
  kernel;                          \
  cudaDeviceSynchronize();         \
  cudaError_t error = cudaGetLastError(); \
  if (error != cudaSuccess) { \
    printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    assert(false); \
  } \
} while (0)
#else
#define cudaSafeCheck(kernel) kernel
#endif

#define tid(axis) (blockIdx.axis * blockDim.axis + threadIdx.axis)


#define EZ_DEVICE __device__
#define EZ_HOST __host__
#define EZ_GLOBAL __global__
#define EZ_CONSTANT __constant__

#endif //GSHOP_GSHOP_INCLUDE_GSHOP_BACKENDS_CUDA_MACROS_H
