//
// Created by creeper on 9/5/24.
//

#ifndef EZCUDA_INCLUDE_EZ_CUDA_MEMORY_H
#define EZCUDA_INCLUDE_EZ_CUDA_MEMORY_H
#include <cuda_runtime.h>
namespace ezcuda {

struct Memory {
  template<typename T>
  T *alloc() const {
    T *ptr;
    cudaMalloc(&ptr, sizeof(T));
    return ptr;
  }

  template<typename T>
  T *alloc(size_t size) const {
    T *ptr;
    cudaMalloc(&ptr, size * sizeof(T));
    return ptr;
  }

  template<typename T>
  T* realloc(T *ptr, size_t size) const {
    T *new_ptr;
    cudaMalloc(&new_ptr, size * sizeof(T));
    cudaMemcpy(new_ptr, ptr, size * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaFree(ptr);
    return new_ptr;
  }

  Memory& free(void *ptr) {
    cudaFree(ptr);
    return *this;
  }
private:

};

}
#endif //EZCUDA_INCLUDE_EZ_CUDA_MEMORY_H
